from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Literal, Tuple
import math

Side = Literal["BUY", "SELL"]
PositionAction = Literal["OPEN", "CLOSE", "REDUCE", "CANCEL"]


@dataclass(frozen=True)
class VenueConfig:
    name: str
    fee_bps: float
    slip_bps: float
    funding_interval_sec: int
    funding_block_pre_sec: int
    funding_block_post_sec: int
    funding_bps_max_abs: float


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float
    max_effective_leverage: float
    gross_cap_mult: float
    symbol_caps: Dict[str, float]
    bucket_caps: Dict[str, float]
    symbol_bucket: Dict[str, str]


@dataclass(frozen=True)
class TradingRules:
    min_order_size: float
    max_order_size: float
    min_price_increment: float
    min_base_amount_increment: float
    min_notional_size: float


@dataclass
class PortfolioState:
    equity_usd: float
    positions_notional: Dict[str, float]


@dataclass(frozen=True)
class MarketSnapshot:
    price: float
    funding_bps: Optional[float] = None
    next_funding_ts: Optional[int] = None


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step


def _round_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(_floor_to_step(price, tick), 12)


def _next_funding_time_fixed(now_utc: datetime, venue: VenueConfig) -> datetime:
    if venue.funding_interval_sec == 3600:
        return now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    base = now_utc.replace(minute=0, second=0, microsecond=0)
    hour = base.hour
    next_hour = ((hour // 8) + 1) * 8
    if next_hour >= 24:
        return base.replace(hour=0) + timedelta(days=1)
    return base.replace(hour=next_hour)


def funding_guard_ok(
    side: Side,
    snap: MarketSnapshot,
    venue: VenueConfig,
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, str]:
    now_utc = now_utc or datetime.now(timezone.utc)

    if snap.funding_bps is not None:
        if abs(snap.funding_bps) > venue.funding_bps_max_abs:
            return False, f"funding_abs_too_high({snap.funding_bps}bps)"

    if snap.next_funding_ts is not None:
        next_ft = datetime.fromtimestamp(snap.next_funding_ts, tz=timezone.utc)
    else:
        next_ft = _next_funding_time_fixed(now_utc, venue)

    dt = (next_ft - now_utc).total_seconds()
    if -venue.funding_block_post_sec <= dt <= venue.funding_block_pre_sec:
        return False, f"funding_window_block(dt={int(dt)}s)"

    return True, "ok"


def compute_notional_from_stop(
    equity: float,
    risk_per_trade: float,
    stop_bps: float,
    fee_bps: float,
    slip_bps: float,
) -> float:
    if equity <= 0:
        return 0.0
    stop_frac = max(stop_bps, 0) / 10_000.0
    friction_frac = max(fee_bps + slip_bps, 0) / 10_000.0
    loss_frac = stop_frac + friction_frac
    if loss_frac <= 0:
        return 0.0
    risk_budget = equity * risk_per_trade
    return risk_budget / loss_frac


def apply_exposure_caps(
    symbol: str,
    desired_notional: float,
    portfolio: PortfolioState,
    risk: RiskConfig,
) -> Tuple[float, str]:
    equity = portfolio.equity_usd
    if equity <= 0:
        return 0.0, "no_equity"

    sym_cap = risk.symbol_caps.get(symbol, 0.0) * equity
    if sym_cap > 0:
        desired_notional = min(desired_notional, sym_cap)

    current = portfolio.positions_notional
    gross_now = sum(abs(v) for v in current.values())
    gross_cap = risk.gross_cap_mult * equity
    lev_cap = risk.max_effective_leverage * equity
    gross_limit = min(gross_cap, lev_cap)

    bucket = risk.symbol_bucket.get(symbol)
    if bucket:
        bucket_cap = risk.bucket_caps.get(bucket, 0.0) * equity
        if bucket_cap > 0:
            bucket_gross_now = sum(
                abs(v) for k, v in current.items() if risk.symbol_bucket.get(k) == bucket
            )
            remaining_bucket = max(bucket_cap - bucket_gross_now, 0.0)
            desired_notional = min(desired_notional, remaining_bucket)

    remaining_gross = max(gross_limit - gross_now, 0.0)
    desired_notional = min(desired_notional, remaining_gross)

    if desired_notional <= 0:
        return 0.0, "capped_to_zero"

    return desired_notional, "ok"


def quantize_order(
    notional_usd: float,
    snap: MarketSnapshot,
    rules: TradingRules,
) -> Tuple[float, float, str]:
    px = snap.price
    if px <= 0:
        return 0.0, 0.0, "bad_price"

    qty = notional_usd / px
    qty = max(qty, rules.min_order_size)
    qty = min(qty, rules.max_order_size)

    if rules.min_notional_size > 0 and qty * px < rules.min_notional_size:
        qty = rules.min_notional_size / px

    qty_q = _floor_to_step(qty, rules.min_base_amount_increment)
    px_q = _round_price(px, rules.min_price_increment)

    if qty_q <= 0:
        return 0.0, 0.0, "qty_round_to_zero"

    return qty_q, px_q, "ok"
