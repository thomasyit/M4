import json
import os
from typing import Dict

from bot.risk_engine import PortfolioState


def load_portfolio_state(symbol: str) -> PortfolioState:
    state_file = os.getenv("PORTFOLIO_STATE_FILE", "state/portfolio.json")
    if not os.path.exists(state_file):
        equity = float(os.getenv("EQUITY_USD", "10000"))
        return PortfolioState(equity_usd=equity, positions_notional={})

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    equity = float(data.get("equity_usd", os.getenv("EQUITY_USD", "10000")))
    positions = data.get("positions_notional", {})
    return PortfolioState(equity_usd=equity, positions_notional=positions)


def save_portfolio_state(state: PortfolioState) -> None:
    state_file = os.getenv("PORTFOLIO_STATE_FILE", "state/portfolio.json")
    os.makedirs(os.path.dirname(state_file), exist_ok=True)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "equity_usd": state.equity_usd,
                "positions_notional": state.positions_notional,
            },
            f,
            indent=2,
        )


def apply_notional_delta(
    state: PortfolioState,
    symbol: str,
    notional_delta: float,
) -> PortfolioState:
    positions: Dict[str, float] = dict(state.positions_notional)
    positions[symbol] = positions.get(symbol, 0.0) + notional_delta
    return PortfolioState(equity_usd=state.equity_usd, positions_notional=positions)
