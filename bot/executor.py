from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from typing import Dict, Any, Tuple, List

from bot.config import venue_config_for, risk_config, default_trading_rules
from bot.hummingbot_client import HummingbotAPIClient
from bot.risk_engine import (
    MarketSnapshot,
    TradingRules,
    funding_guard_ok,
    compute_notional_from_stop,
    apply_exposure_caps,
    quantize_order,
)
from bot.state_store import load_portfolio_state, save_portfolio_state, apply_notional_delta
from bot.trade_intent import TradeIntent
from bot.trade_state import load_trade_state, save_trade_state, ActiveTrade, record_trade_close, reset_daily_state
import requests


def _record_receipt(intent: TradeIntent, status: str, details: Dict[str, Any]) -> str:
    out_dir = os.getenv("RECEIPTS_DIR", "state/receipts")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{intent.msg_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": status,
                "version": os.getenv("BOT_VERSION", "v2"),
                "intent": asdict(intent),
                "details": details,
            },
            f,
            indent=2,
        )
    return path


def _resolve_trading_rules(connector: str, symbol: str, dry_run: bool) -> TradingRules:
    if dry_run:
        return default_trading_rules()

    client = HummingbotAPIClient()
    rules_map = client.get_trading_rules(connector_name=connector, trading_pairs=[symbol])

    if symbol not in rules_map:
        raise ValueError(f"Trading rules not found for {symbol}")

    rules = rules_map[symbol]
    return TradingRules(
        min_order_size=float(rules["min_order_size"]),
        max_order_size=float(rules["max_order_size"]),
        min_price_increment=float(rules["min_price_increment"]),
        min_base_amount_increment=float(rules["min_base_amount_increment"]),
        min_notional_size=float(rules.get("min_notional_size", 0)),
    )


def _build_order_payload(intent: TradeIntent, qty: float, price: float) -> Dict[str, Any]:
    payload = {
        "account_name": intent.execution.account,
        "connector_name": intent.execution.connector,
        "trading_pair": intent.symbol,
        "trade_type": intent.side,
        "amount": qty,
        "order_type": intent.execution.order_type,
        "position_action": intent.intent,
    }
    if intent.execution.order_type.upper() == "LIMIT":
        payload["price"] = price
    if intent.execution.time_in_force:
        payload["time_in_force"] = intent.execution.time_in_force
    return payload


def _normalize_symbol_for_binance(symbol: str) -> str:
    return symbol.replace("-", "")


def _fetch_last_price(symbol: str) -> float:
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": _normalize_symbol_for_binance(symbol)}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return float(response.json()["price"])


def _fetch_spread_bps(symbol: str) -> float:
    url = "https://api.binance.com/api/v3/ticker/bookTicker"
    params = {"symbol": _normalize_symbol_for_binance(symbol)}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    bid = float(data["bidPrice"])
    ask = float(data["askPrice"])
    mid = (bid + ask) / 2
    if mid <= 0:
        return 0.0
    return (ask - bid) / mid * 10_000.0


def _place_tp_orders(
    client: HummingbotAPIClient,
    intent: TradeIntent,
    qty: float,
    entry: float,
) -> List[Dict[str, Any]]:
    tp_pcts = [float(x) for x in os.getenv("TP_LADDER_PCTS", "0.5,0.5").split(",")]
    tp_rs = [float(x) for x in os.getenv("TP_LADDER_R_MULTS", "1.0,2.0").split(",")]
    stop_bps = float(intent.risk.stop_bps)
    stop_dist = entry * stop_bps / 10_000.0

    orders = []
    for pct, r_mult in zip(tp_pcts, tp_rs):
        tp_qty = qty * pct
        if tp_qty <= 0:
            continue
        if intent.side == "BUY":
            tp_price = entry + stop_dist * r_mult
        else:
            tp_price = entry - stop_dist * r_mult
        payload = {
            "account_name": intent.execution.account,
            "connector_name": intent.execution.connector,
            "trading_pair": intent.symbol,
            "trade_type": "SELL" if intent.side == "BUY" else "BUY",
            "amount": tp_qty,
            "order_type": "LIMIT",
            "price": tp_price,
            "position_action": "CLOSE",
        }
        resp = client.place_order(payload)
        orders.append(resp)
    return orders


def _start_trailing_stop(
    client: HummingbotAPIClient,
    intent: TradeIntent,
    qty: float,
    entry: float,
    atr: float,
) -> None:
    trail_mult = float(os.getenv("TRAIL_STOP_ATR_MULT", "1.0"))
    poll_sec = int(os.getenv("TRAIL_POLL_SEC", "30"))
    trail_dist = atr * trail_mult

    def _run():
        peak = entry
        trough = entry
        while True:
            try:
                last = _fetch_last_price(intent.symbol)
            except Exception:
                time.sleep(poll_sec)
                continue

            if intent.side == "BUY":
                peak = max(peak, last)
                stop_level = peak - trail_dist
                if last <= stop_level:
                    payload = {
                        "account_name": intent.execution.account,
                        "connector_name": intent.execution.connector,
                        "trading_pair": intent.symbol,
                        "trade_type": "SELL",
                        "amount": qty,
                        "order_type": "MARKET",
                        "position_action": "CLOSE",
                    }
                    client.place_order(payload)
                    state = reset_daily_state(load_trade_state())
                    state = record_trade_close(state, exit_price=stop_level)
                    save_trade_state(state)
                    break
            else:
                trough = min(trough, last)
                stop_level = trough + trail_dist
                if last >= stop_level:
                    payload = {
                        "account_name": intent.execution.account,
                        "connector_name": intent.execution.connector,
                        "trading_pair": intent.symbol,
                        "trade_type": "BUY",
                        "amount": qty,
                        "order_type": "MARKET",
                        "position_action": "CLOSE",
                    }
                    client.place_order(payload)
                    state = reset_daily_state(load_trade_state())
                    state = record_trade_close(state, exit_price=stop_level)
                    save_trade_state(state)
                    break

            time.sleep(poll_sec)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def execute_intent(intent: TradeIntent) -> Tuple[bool, str]:
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    venue = venue_config_for(intent.execution.connector)
    risk = risk_config()

    snapshot = MarketSnapshot(
        price=float(intent.snapshot.price),
        funding_bps=intent.snapshot.funding_bps,
        next_funding_ts=intent.snapshot.next_funding_ts,
    )

    ok, reason = funding_guard_ok(intent.side, snapshot, venue)
    if not ok:
        _record_receipt(intent, "rejected", {"reason": reason})
        return False, reason

    portfolio = load_portfolio_state(intent.symbol)

    desired_notional = compute_notional_from_stop(
        equity=portfolio.equity_usd,
        risk_per_trade=risk.risk_per_trade,
        stop_bps=intent.risk.stop_bps,
        fee_bps=venue.fee_bps,
        slip_bps=venue.slip_bps,
    )

    if intent.risk.max_usd:
        desired_notional = min(desired_notional, intent.risk.max_usd)

    desired_notional, cap_reason = apply_exposure_caps(
        intent.symbol, desired_notional, portfolio, risk
    )
    size_multiplier = float((intent.metadata or {}).get("size_multiplier", 1.0))
    if size_multiplier > 0 and size_multiplier != 1.0:
        desired_notional *= size_multiplier
    if desired_notional <= 0:
        _record_receipt(intent, "rejected", {"reason": cap_reason})
        return False, cap_reason

    max_slip_bps = float(os.getenv("MAX_SLIPPAGE_BPS", "30"))
    max_spread_bps = float(os.getenv("MAX_SPREAD_BPS", "10"))
    try:
        last_price = _fetch_last_price(intent.symbol)
        slip_bps = abs(last_price - snapshot.price) / snapshot.price * 10_000.0
        if slip_bps > max_slip_bps:
            _record_receipt(intent, "rejected", {"reason": "slippage_guard", "slip_bps": slip_bps})
            return False, "slippage_guard"
        spread_bps = _fetch_spread_bps(intent.symbol)
        if spread_bps > max_spread_bps:
            _record_receipt(intent, "rejected", {"reason": "spread_guard", "spread_bps": spread_bps})
            return False, "spread_guard"
    except Exception:
        pass

    rules = _resolve_trading_rules(intent.execution.connector, intent.symbol, dry_run)
    qty, px, q_reason = quantize_order(desired_notional, snapshot, rules)
    if qty <= 0:
        _record_receipt(intent, "rejected", {"reason": q_reason})
        return False, q_reason

    order_payload = _build_order_payload(intent, qty, px)

    if dry_run:
        tp_orders = []
        tp_prices = []
        if os.getenv("ENABLE_TP_LADDER", "true").lower() == "true":
            stop_dist = px * float(intent.risk.stop_bps) / 10_000.0
            tp_pcts = [float(x) for x in os.getenv("TP_LADDER_PCTS", "0.5,0.5").split(",")]
            tp_rs = [float(x) for x in os.getenv("TP_LADDER_R_MULTS", "1.0,2.0").split(",")]
            for r_mult in tp_rs:
                if intent.side == "BUY":
                    tp_prices.append(px + stop_dist * r_mult)
                else:
                    tp_prices.append(px - stop_dist * r_mult)
        trail_settings = {
            "enabled": os.getenv("ENABLE_TRAIL_STOP", "true").lower() == "true",
            "trail_atr_mult": float(os.getenv("TRAIL_STOP_ATR_MULT", "1.0")),
            "trail_poll_sec": int(os.getenv("TRAIL_POLL_SEC", "30")),
        }
        tp_settings = {
            "tp_pcts": os.getenv("TP_LADDER_PCTS", "0.5,0.5"),
            "tp_r_mults": os.getenv("TP_LADDER_R_MULTS", "1.0,2.0"),
        }
        receipt_path = _record_receipt(
            intent,
            "dry_run",
            {
                "desired_notional": desired_notional,
                "qty": qty,
                "price": px,
                "order_payload": order_payload,
                "tp_settings": tp_settings,
                "trail_settings": trail_settings,
                "tp_orders": tp_orders,
                "tp_prices": tp_prices,
            },
        )
        return True, f"dry_run_receipt={receipt_path}"

    client = HummingbotAPIClient()
    response = client.place_order(order_payload)
    _record_receipt(intent, "executed", {"order_response": response})

    signed_notional = desired_notional if intent.side == "BUY" else -desired_notional
    new_state = apply_notional_delta(portfolio, intent.symbol, signed_notional)
    save_portfolio_state(new_state)

    trade_state = reset_daily_state(load_trade_state())
    tp_order_ids = []
    tp_prices = []
    if os.getenv("ENABLE_TP_LADDER", "true").lower() == "true":
        tp_orders = _place_tp_orders(client, intent, qty, px)
        for resp in tp_orders:
            order_id = resp.get("order_id") or resp.get("client_order_id")
            if order_id:
                tp_order_ids.append(str(order_id))
        stop_dist = px * float(intent.risk.stop_bps) / 10_000.0
        tp_rs = [float(x) for x in os.getenv("TP_LADDER_R_MULTS", "1.0,2.0").split(",")]
        for r_mult in tp_rs:
            tp_prices.append(px + stop_dist * r_mult if intent.side == "BUY" else px - stop_dist * r_mult)

    trade_state.active_trade = ActiveTrade(
        trade_id=intent.msg_id,
        symbol=intent.symbol,
        connector=intent.execution.connector,
        account=intent.execution.account,
        side=intent.side,
        qty=qty,
        entry_price=px,
        stop_price=px - (px * float(intent.risk.stop_bps) / 10_000.0)
        if intent.side == "BUY"
        else px + (px * float(intent.risk.stop_bps) / 10_000.0),
        tp_prices=tp_prices,
        tp_order_ids=tp_order_ids,
        opened_at=intent.ts,
    )
    save_trade_state(trade_state)
    if os.getenv("ENABLE_TRAIL_STOP", "true").lower() == "true":
        _start_trailing_stop(client, intent, qty, px, atr=abs(px * intent.risk.stop_bps / 10_000.0))

    return True, "executed"
