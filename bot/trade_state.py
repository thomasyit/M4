import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


STATE_FILE = os.getenv("TRADE_STATE_FILE", "state/trade_state.json")


@dataclass
class ActiveTrade:
    trade_id: str
    symbol: str
    connector: str
    account: str
    side: str
    qty: float
    entry_price: float
    stop_price: float
    tp_prices: list[float]
    tp_order_ids: list[str]
    opened_at: int


@dataclass
class TradeState:
    day: str
    day_start_equity: float
    equity_usd: float
    loss_streak: int
    last_result: Optional[str]
    active_trade: Optional[ActiveTrade]


def _default_state() -> TradeState:
    equity = float(os.getenv("EQUITY_USD", "10000"))
    return TradeState(
        day=datetime.utcnow().strftime("%Y-%m-%d"),
        day_start_equity=equity,
        equity_usd=equity,
        loss_streak=0,
        last_result=None,
        active_trade=None,
    )


def load_trade_state() -> TradeState:
    if not os.path.exists(STATE_FILE):
        return _default_state()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    active = data.get("active_trade")
    active_trade = ActiveTrade(**active) if active else None
    return TradeState(
        day=data.get("day", datetime.utcnow().strftime("%Y-%m-%d")),
        day_start_equity=float(data.get("day_start_equity", data.get("equity_usd", 0) or 0)),
        equity_usd=float(data.get("equity_usd", 0)),
        loss_streak=int(data.get("loss_streak", 0)),
        last_result=data.get("last_result"),
        active_trade=active_trade,
    )


def save_trade_state(state: TradeState) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    payload: Dict[str, Any] = asdict(state)
    if state.active_trade:
        payload["active_trade"] = asdict(state.active_trade)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def reset_daily_state(state: TradeState) -> TradeState:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if state.day != today:
        state.day = today
        state.day_start_equity = state.equity_usd
        state.loss_streak = 0
        state.last_result = None
    return state


def record_trade_close(
    state: TradeState,
    exit_price: float,
) -> TradeState:
    if not state.active_trade:
        return state
    trade = state.active_trade
    direction = 1 if trade.side == "BUY" else -1
    pnl = (exit_price - trade.entry_price) * trade.qty * direction
    state.equity_usd += pnl
    state.last_result = "win" if pnl > 0 else "loss"
    if pnl > 0:
        state.loss_streak = 0
    else:
        state.loss_streak += 1
    state.active_trade = None
    return state
