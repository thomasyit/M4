from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import time
import uuid

Side = Literal["BUY", "SELL"]
Intent = Literal["OPEN", "CLOSE", "REDUCE", "CANCEL"]


@dataclass
class TradeRisk:
    stop_bps: float
    takeprofit_bps: Optional[float]
    max_usd: Optional[float]
    max_leverage: Optional[float]


@dataclass
class ExecutionPrefs:
    connector: str
    account: str
    order_type: str
    time_in_force: Optional[str] = None


@dataclass
class MarketSnapshotInput:
    price: float
    funding_bps: Optional[float] = None
    next_funding_ts: Optional[int] = None


@dataclass
class TradeIntent:
    msg_id: str
    ts: int
    symbol: str
    side: Side
    intent: Intent
    confidence: float
    risk: TradeRisk
    execution: ExecutionPrefs
    snapshot: MarketSnapshotInput
    metadata: Dict[str, Any]

    @staticmethod
    def new(
        symbol: str,
        side: Side,
        intent: Intent,
        confidence: float,
        risk: TradeRisk,
        execution: ExecutionPrefs,
        snapshot: MarketSnapshotInput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TradeIntent":
        return TradeIntent(
            msg_id=str(uuid.uuid4()),
            ts=int(time.time()),
            symbol=symbol,
            side=side,
            intent=intent,
            confidence=confidence,
            risk=risk,
            execution=execution,
            snapshot=snapshot,
            metadata=metadata or {},
        )
