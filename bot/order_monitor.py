import os
from typing import Optional

from bot.hummingbot_client import HummingbotAPIClient
from bot.trade_state import load_trade_state, save_trade_state, record_trade_close, reset_daily_state, ActiveTrade
from bot.executor import _fetch_last_price


def reconcile_active_trade() -> Optional[str]:
    if os.getenv("ORDER_MONITOR_ENABLE", "true").lower() != "true":
        return None

    state = reset_daily_state(load_trade_state())
    if not state.active_trade:
        if os.getenv("POSITION_RECONCILE_ENABLE", "true").lower() != "true":
            return "no_active_trade"
        symbol = os.getenv("SYMBOL", "BTCUSDT")
        connector = os.getenv("HB_CONNECTOR", "binance_perpetual")
        account = os.getenv("HB_ACCOUNT", "master_account")
        try:
            client = HummingbotAPIClient()
            pos_resp = client.get_positions(
                {
                    "connector_name": connector,
                    "account_name": account,
                    "trading_pair": symbol,
                }
            )
        except Exception:
            return "no_active_trade"

        positions = pos_resp.get("positions") if isinstance(pos_resp, dict) else pos_resp
        if not positions:
            return "no_active_trade"

        for pos in positions:
            qty = float(pos.get("position_amt") or pos.get("amount") or pos.get("qty") or 0)
            if qty == 0:
                continue
            side = "BUY" if qty > 0 else "SELL"
            entry_price = float(pos.get("entry_price") or pos.get("avg_entry_price") or 0)
            if entry_price <= 0:
                entry_price = _fetch_last_price(symbol)
            state.active_trade = ActiveTrade(
                trade_id="external_position",
                symbol=symbol,
                connector=connector,
                account=account,
                side=side,
                qty=abs(qty),
                entry_price=entry_price,
                stop_price=0.0,
                tp_prices=[],
                tp_order_ids=[],
                opened_at=int(os.getenv("POSITION_OPENED_AT", "0") or 0),
            )
            save_trade_state(state)
            return "external_position_detected"
        return "no_active_trade"

    trade = state.active_trade
    client = HummingbotAPIClient()

    remaining_tp_ids = []
    for order_id in trade.tp_order_ids:
        try:
            status_resp = client.get_order_status(
                {
                    "order_id": order_id,
                    "connector_name": trade.connector,
                    "account_name": trade.account,
                    "trading_pair": trade.symbol,
                }
            )
        except Exception:
            remaining_tp_ids.append(order_id)
            continue

        status = (status_resp.get("status") or status_resp.get("order_status") or "").lower()
        if status in {"filled", "canceled"}:
            if status == "filled":
                filled_qty = float(status_resp.get("filled_amount", trade.qty))
                trade.qty = max(trade.qty - filled_qty, 0)
        else:
            remaining_tp_ids.append(order_id)

    trade.tp_order_ids = remaining_tp_ids
    if trade.qty <= 0:
        try:
            exit_price = float(status_resp.get("price", 0)) if "status_resp" in locals() else 0
            if exit_price <= 0:
                exit_price = _fetch_last_price(trade.symbol)
        except Exception:
            exit_price = trade.entry_price
        state = record_trade_close(state, exit_price=exit_price)
    else:
        state.active_trade = trade

    save_trade_state(state)
    return "reconciled"
