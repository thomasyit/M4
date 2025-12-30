import os
from typing import Dict

from bot.risk_engine import VenueConfig, RiskConfig, TradingRules


def venue_config_for(connector: str) -> VenueConfig:
    if connector == "hyperliquid_perpetual":
        return VenueConfig(
            name="hyperliquid_perpetual",
            fee_bps=float(os.getenv("HL_FEE_BPS", "6")),
            slip_bps=float(os.getenv("HL_SLIP_BPS", "10")),
            funding_interval_sec=3600,
            funding_block_pre_sec=int(os.getenv("HL_FUNDING_BLOCK_PRE", "120")),
            funding_block_post_sec=int(os.getenv("HL_FUNDING_BLOCK_POST", "30")),
            funding_bps_max_abs=float(os.getenv("HL_FUNDING_MAX_ABS", "12")),
        )

    return VenueConfig(
        name="binance_perpetual",
        fee_bps=float(os.getenv("BINANCE_FEE_BPS", "6")),
        slip_bps=float(os.getenv("BINANCE_SLIP_BPS", "8")),
        funding_interval_sec=8 * 3600,
        funding_block_pre_sec=int(os.getenv("BINANCE_FUNDING_BLOCK_PRE", "600")),
        funding_block_post_sec=int(os.getenv("BINANCE_FUNDING_BLOCK_POST", "120")),
        funding_bps_max_abs=float(os.getenv("BINANCE_FUNDING_MAX_ABS", "8")),
    )


def risk_config() -> RiskConfig:
    symbol_caps = _parse_ratio_map(os.getenv("SYMBOL_CAPS", "BTC-USDT=0.35,ETH-USDT=0.25,SOL-USDT=0.15,LTC-USDT=0.08"))
    bucket_caps = _parse_ratio_map(os.getenv("BUCKET_CAPS", "majors=0.80,alts=0.50"))
    symbol_bucket = _parse_symbol_bucket(os.getenv("SYMBOL_BUCKET", "BTC-USDT=majors,ETH-USDT=majors,SOL-USDT=alts,LTC-USDT=alts"))

    return RiskConfig(
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.003")),
        max_effective_leverage=float(os.getenv("MAX_EFFECTIVE_LEVERAGE", "2.5")),
        gross_cap_mult=float(os.getenv("GROSS_CAP_MULT", "1.25")),
        symbol_caps=symbol_caps,
        bucket_caps=bucket_caps,
        symbol_bucket=symbol_bucket,
    )


def default_trading_rules() -> TradingRules:
    return TradingRules(
        min_order_size=float(os.getenv("MIN_ORDER_SIZE", "0.0001")),
        max_order_size=float(os.getenv("MAX_ORDER_SIZE", "100")),
        min_price_increment=float(os.getenv("MIN_PRICE_INCREMENT", "0.1")),
        min_base_amount_increment=float(os.getenv("MIN_BASE_AMOUNT_INCREMENT", "0.0001")),
        min_notional_size=float(os.getenv("MIN_NOTIONAL_SIZE", "5")),
    )


def _parse_ratio_map(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, val = item.split("=")
        out[key.strip()] = float(val.strip())
    return out


def _parse_symbol_bucket(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, val = item.split("=")
        out[key.strip()] = val.strip()
    return out
