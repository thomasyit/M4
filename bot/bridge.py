from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from bot.sentiment_analyzer import SentimentAnalyzer
from bot.synternet_sentiment_analyzer import SyntoshiApexSentimentAnalyzer
from bot.trade_intent import TradeIntent, TradeRisk, ExecutionPrefs, MarketSnapshotInput
from bot.trade_state import load_trade_state, save_trade_state, reset_daily_state

logger = logging.getLogger(__name__)


class SentimentBridge:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.min_confidence = float(os.getenv("CONF_MIN", "0.55"))
        self.connector = os.getenv("HB_CONNECTOR", "binance_perpetual")
        self.account = os.getenv("HB_ACCOUNT", "master_account")
        self.order_type = os.getenv("HB_ORDER_TYPE", "MARKET")
        self.time_in_force = os.getenv("HB_TIME_IN_FORCE")
        self.atr_period = int(os.getenv("ATR_PERIOD", "14"))
        self.atr_stop_mult = float(os.getenv("ATR_STOP_MULT", "1.2"))
        self.atr_tp_mult = float(os.getenv("ATR_TP_MULT", "2.0"))
        self.regime_mode = os.getenv("REGIME_MODE", "any").lower()
        self.regime_trend_threshold = float(os.getenv("REGIME_TREND_THRESHOLD", "0.003"))
        self.regime_vol_max = float(os.getenv("REGIME_VOL_MAX", "0.05"))
        self.regime_allow_volatile = os.getenv("REGIME_ALLOW_VOLATILE", "false").lower() == "true"
        self.ema_fast = int(os.getenv("EMA_FAST", "20"))
        self.ema_slow = int(os.getenv("EMA_SLOW", "50"))
        self.require_4h_confirm = os.getenv("REQUIRE_4H_CONFIRM", "true").lower() == "true"
        self.adx_period = int(os.getenv("ADX_PERIOD", "14"))
        self.adx_min = float(os.getenv("ADX_MIN", "0"))
        self.ta_conf_min = float(os.getenv("TA_CONF_MIN", "0.5"))
        self.max_daily_dd = float(os.getenv("MAX_DAILY_DD", "0.02"))
        self.max_loss_streak = int(os.getenv("MAX_LOSS_STREAK", "3"))
        self.allow_multiple_positions = os.getenv("ALLOW_MULTIPLE_POSITIONS", "false").lower() == "true"
        self.max_funding_long_bps = float(os.getenv("FUNDING_MAX_LONG_BPS", "8"))
        self.max_funding_short_bps = float(os.getenv("FUNDING_MAX_SHORT_BPS", "8"))

    @staticmethod
    def _bps_from_prices(entry: float, other: float) -> float:
        if entry <= 0:
            return 0.0
        return abs(entry - other) / entry * 10_000.0

    @staticmethod
    def _compute_atr(candles: list[list], period: int) -> Optional[float]:
        if not candles or len(candles) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            current = candles[i]
            prev = candles[i - 1]
            high = float(current[2])
            low = float(current[3])
            prev_close = float(prev[4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)

    @staticmethod
    def _compute_ema(values: list[float], period: int) -> Optional[float]:
        if not values or len(values) < period:
            return None
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for price in values[period:]:
            ema = price * k + ema * (1 - k)
        return ema

    def _derive_ta_sentiment(self, candles_1h: list[list]) -> str:
        metrics = self._compute_ta_metrics(candles_1h)
        if not metrics:
            return "neutral"
        trend_strength = metrics["trend_strength"]
        adx_value = metrics.get("adx", 0.0)
        if trend_strength < self.regime_trend_threshold:
            return "neutral"
        if self.adx_min > 0 and (adx_value is None or adx_value < self.adx_min):
            return "neutral"
        if metrics["ema_fast"] > metrics["ema_slow"]:
            return "bullish"
        if metrics["ema_fast"] < metrics["ema_slow"]:
            return "bearish"
        return "neutral"

    def _derive_ta_direction(self, candles_1h: list[list]) -> Optional[str]:
        if not candles_1h:
            return None
        closes = [float(c[4]) for c in candles_1h]
        ema_fast = self._compute_ema(closes, self.ema_fast)
        ema_slow = self._compute_ema(closes, self.ema_slow)
        if ema_fast is None or ema_slow is None:
            return None
        if ema_fast > ema_slow:
            return "long"
        if ema_fast < ema_slow:
            return "short"
        return None

    @staticmethod
    def _compute_adx(candles: list[list], period: int) -> Optional[float]:
        if not candles or len(candles) < period + 1:
            return None

        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]

        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        for i in range(1, len(candles)):
            high = highs[i]
            low = lows[i]
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            prev_close = closes[i - 1]

            up_move = high - prev_high
            down_move = prev_low - low
            plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
            tr_list.append(tr)

        if len(tr_list) < period:
            return None

        tr14 = sum(tr_list[:period])
        plus_dm14 = sum(plus_dm_list[:period])
        minus_dm14 = sum(minus_dm_list[:period])
        dx_list = []

        def _calc_dx(p_dm: float, m_dm: float, tr_val: float) -> float:
            if tr_val == 0:
                return 0.0
            plus_di = 100.0 * (p_dm / tr_val)
            minus_di = 100.0 * (m_dm / tr_val)
            denom = plus_di + minus_di
            if denom == 0:
                return 0.0
            return 100.0 * abs(plus_di - minus_di) / denom

        dx_list.append(_calc_dx(plus_dm14, minus_dm14, tr14))

        for i in range(period, len(tr_list)):
            tr14 = tr14 - (tr14 / period) + tr_list[i]
            plus_dm14 = plus_dm14 - (plus_dm14 / period) + plus_dm_list[i]
            minus_dm14 = minus_dm14 - (minus_dm14 / period) + minus_dm_list[i]
            dx_list.append(_calc_dx(plus_dm14, minus_dm14, tr14))

        if len(dx_list) < period:
            return None

        adx = sum(dx_list[:period]) / period
        for i in range(period, len(dx_list)):
            adx = (adx * (period - 1) + dx_list[i]) / period
        return adx

    def _compute_ta_metrics(self, candles_1h: list[list]) -> Optional[Dict[str, float]]:
        if not candles_1h:
            return None
        closes = [float(c[4]) for c in candles_1h]
        entry = closes[-1]
        atr = self._compute_atr(candles_1h, self.atr_period)
        ema_fast = self._compute_ema(closes, self.ema_fast)
        ema_slow = self._compute_ema(closes, self.ema_slow)
        if atr is None or ema_fast is None or ema_slow is None:
            return None
        trend_strength = abs(ema_fast - ema_slow) / entry if entry else 0.0
        vol_ratio = atr / entry if entry else 0.0
        adx_value = self._compute_adx(candles_1h, self.adx_period)
        return {
            "entry": entry,
            "atr": atr,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "trend_strength": trend_strength,
            "vol_ratio": vol_ratio,
            "adx": adx_value if adx_value is not None else 0.0,
        }

    def _compute_ta_confidence(self, metrics: Dict[str, float]) -> float:
        trend_score = metrics["trend_strength"] / max(self.regime_trend_threshold, 1e-6)
        trend_score = min(1.0, max(trend_score, 0.0))
        adx_value = metrics.get("adx", 0.0)
        adx_score = min(1.0, max(adx_value / 50.0, 0.0)) if adx_value else 0.0
        if adx_value:
            return 0.5 * trend_score + 0.5 * adx_score
        return trend_score

    def build_intent(
        self, sentiment_result: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Optional[TradeIntent]:
        if not sentiment_result:
            return None

        confidence = float(sentiment_result.get("confidence", 0))
        if confidence < self.min_confidence:
            return None

        llm_sentiment = _normalize_sentiment(sentiment_result.get("sentiment"))
        if llm_sentiment == "neutral":
            return None

        state = reset_daily_state(load_trade_state())
        save_trade_state(state)
        if state.active_trade and not self.allow_multiple_positions:
            return None
        if state.day_start_equity > 0:
            drawdown = (state.day_start_equity - state.equity_usd) / state.day_start_equity
            if drawdown >= self.max_daily_dd:
                return None
        if state.loss_streak >= self.max_loss_streak:
            return None

        candles_1h = market_data.get("candles_1h") or []
        metrics = self._compute_ta_metrics(candles_1h)
        if not metrics:
            return None

        entry = metrics["entry"]
        atr = metrics["atr"]
        ema_fast = metrics["ema_fast"]
        ema_slow = metrics["ema_slow"]
        logger.info(
            "TA metrics: entry=%.2f ema_fast=%.2f ema_slow=%.2f atr=%.2f trend=%.4f vol_ratio=%.4f adx=%.2f",
            entry,
            ema_fast,
            ema_slow,
            atr,
            metrics["trend_strength"],
            metrics["vol_ratio"],
            metrics["adx"],
        )

        if ema_fast > ema_slow:
            direction = "long"
        elif ema_fast < ema_slow:
            direction = "short"
        else:
            return None

        if llm_sentiment == "bullish" and direction != "long":
            return None
        if llm_sentiment == "bearish" and direction != "short":
            return None

        side = "BUY" if direction == "long" else "SELL"

        trend_strength = metrics["trend_strength"]
        vol_ratio = metrics["vol_ratio"]
        if vol_ratio > self.regime_vol_max and not self.regime_allow_volatile:
            return None
        if self.regime_mode == "trend" and trend_strength < self.regime_trend_threshold:
            return None
        if self.regime_mode == "range" and trend_strength >= self.regime_trend_threshold:
            return None

        candles_4h = market_data.get("candles_4h") or []
        if self.require_4h_confirm:
            if not candles_4h:
                return None
            closes_4h = [float(c[4]) for c in candles_4h]
            ema_fast_4h = self._compute_ema(closes_4h, self.ema_fast)
            ema_slow_4h = self._compute_ema(closes_4h, self.ema_slow)
            if ema_fast_4h is None or ema_slow_4h is None:
                return None
            if direction == "long" and ema_fast_4h < ema_slow_4h:
                return None
            if direction == "short" and ema_fast_4h > ema_slow_4h:
                return None

        adx_value = self._compute_adx(candles_1h, self.adx_period)
        if self.adx_min > 0:
            if adx_value is None or adx_value < self.adx_min:
                return None
        ta_confidence = self._compute_ta_confidence(metrics)
        if ta_confidence < self.ta_conf_min:
            return None

        funding_bps = None
        funding_rates = market_data.get("funding_rates") or []
        if funding_rates:
            latest = funding_rates[-1]
            funding_bps = float(latest.get("fundingRate", 0)) * 10_000.0
        if funding_bps is not None:
            if direction == "long" and funding_bps > self.max_funding_long_bps:
                return None
            if direction == "short" and funding_bps < -self.max_funding_short_bps:
                return None

        if direction == "long":
            stop = entry - self.atr_stop_mult * atr
            take = entry + self.atr_tp_mult * atr
        else:
            stop = entry + self.atr_stop_mult * atr
            take = entry - self.atr_tp_mult * atr

        stop_bps = self._bps_from_prices(entry, stop)
        take_bps = self._bps_from_prices(entry, take) if take else None

        risk = TradeRisk(
            stop_bps=stop_bps,
            takeprofit_bps=take_bps,
            max_usd=None,
            max_leverage=None,
        )

        execution = ExecutionPrefs(
            connector=self.connector,
            account=self.account,
            order_type=self.order_type,
            time_in_force=self.time_in_force,
        )

        snapshot = MarketSnapshotInput(
            price=entry,
            funding_bps=funding_bps,
            next_funding_ts=None,
        )

        return TradeIntent.new(
            symbol=self.symbol,
            side=side,
            intent="OPEN",
            confidence=confidence,
            risk=risk,
            execution=execution,
            snapshot=snapshot,
            metadata={
                "analysis_model": sentiment_result.get("analysis_model"),
                "sentiment": sentiment_result.get("sentiment"),
                "reasoning": sentiment_result.get("reasoning"),
                "market_regime": sentiment_result.get("market_regime"),
            },
        )

def _normalize_sentiment(value: Optional[str]) -> str:
    if not value:
        return "neutral"
    value = value.lower().strip()
    if value in ["bullish", "bearish", "neutral"]:
        return value
    if "bull" in value:
        return "bullish"
    if "bear" in value:
        return "bearish"
    return "neutral"


def _unify_trade_setup(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
) -> Dict[str, Any]:
    primary_setup = primary.get("highest_probability_trade") or {}
    secondary_setup = secondary.get("highest_probability_trade") or {}

    def pick_value(key: str) -> Optional[float]:
        p_val = primary_setup.get(key)
        s_val = secondary_setup.get(key)
        if p_val is None and s_val is None:
            return None
        if p_val is None:
            return float(s_val)
        if s_val is None:
            return float(p_val)
        return round((float(p_val) + float(s_val)) / 2, 2)

    unified = dict(primary)
    unified["analysis_model"] = "ChatGPT + Synternet (agreement)"
    unified["sentiment"] = _normalize_sentiment(primary.get("sentiment"))
    unified["confidence"] = min(
        float(primary.get("confidence", 0.0)),
        float(secondary.get("confidence", 0.0)),
    )

    unified["highest_probability_trade"] = {
        "direction": primary_setup.get("direction"),
        "entry_price": pick_value("entry_price"),
        "stop_loss": pick_value("stop_loss"),
        "take_profit": pick_value("take_profit"),
        "risk_reward_ratio": primary_setup.get("risk_reward_ratio")
        or secondary_setup.get("risk_reward_ratio"),
        "rationale": (
            primary_setup.get("rationale")
            or secondary_setup.get("rationale")
        ),
    }

    unified["synternet"] = {
        "analysis_model": secondary.get("analysis_model"),
        "sentiment": secondary.get("sentiment"),
        "confidence": secondary.get("confidence"),
    }
    return unified


def analyze_and_build_intent(symbol: str) -> Optional[TradeIntent]:
    intent, _summary = analyze_and_build_intent_with_summary(symbol)
    return intent


def analyze_and_build_intent_with_summary(symbol: str) -> tuple[Optional[TradeIntent], Dict[str, Any]]:
    mode = os.getenv("ANALYSIS_MODE", "dual").lower()
    synternet_timeout = int(os.getenv("SYNTERNET_TIMEOUT_SEC", "20"))

    analyzer = SentimentAnalyzer(symbol=symbol)
    market_data = analyzer.compile_market_data()
    ta_bridge = SentimentBridge(symbol=symbol)

    if market_data.get("data_fresh") is False:
        summary: Dict[str, Any] = {
            "ts": int(datetime.utcnow().timestamp()),
            "run_id": f"run_{int(datetime.utcnow().timestamp())}",
            "mode": mode,
            "chatgpt_sentiment": None,
            "chatgpt_confidence": None,
            "chatgpt_reasoning": None,
            "synternet_sentiment": None,
            "synternet_confidence": None,
            "synternet_reasoning": None,
            "ta_sentiment": ta_bridge._derive_ta_sentiment(market_data.get("candles_1h") or []),
            "decision": "rejected",
            "reason": "data_stale",
            "data_fresh_reason": market_data.get("data_fresh_reason"),
            "data_fresh_age_hours": market_data.get("data_fresh_age_hours"),
        }
        return None, summary

    sentiment = analyzer.analyze_market(market_data)

    summary: Dict[str, Any] = {
        "ts": int(datetime.utcnow().timestamp()),
            "run_id": f"run_{int(datetime.utcnow().timestamp())}",
        "mode": mode,
        "chatgpt_sentiment": None,
        "chatgpt_confidence": None,
        "chatgpt_reasoning": None,
        "synternet_sentiment": None,
        "synternet_confidence": None,
        "synternet_reasoning": None,
        "ta_sentiment": None,
        "decision": "skipped",
        "reason": "",
    }

    if "data_fresh_age_hours" in market_data:
        summary["data_fresh_age_hours"] = market_data.get("data_fresh_age_hours")
    if "data_fresh_reason" in market_data:
        summary["data_fresh_reason"] = market_data.get("data_fresh_reason")
    if "data_fresh" in market_data:
        summary["data_fresh"] = market_data.get("data_fresh")

    ta_metrics = ta_bridge._compute_ta_metrics(market_data.get("candles_1h") or [])
    summary["ta_sentiment"] = ta_bridge._derive_ta_sentiment(market_data.get("candles_1h") or [])
    if ta_metrics:
        summary["ta_ema_fast"] = round(ta_metrics["ema_fast"], 6)
        summary["ta_ema_slow"] = round(ta_metrics["ema_slow"], 6)
        summary["ta_atr"] = round(ta_metrics["atr"], 6)
        summary["ta_trend_strength"] = round(ta_metrics["trend_strength"], 6)
        summary["ta_vol_ratio"] = round(ta_metrics["vol_ratio"], 6)
        summary["ta_adx"] = round(ta_metrics["adx"], 6)
        summary["ta_confidence"] = round(ta_bridge._compute_ta_confidence(ta_metrics), 6)

    if sentiment:
        summary["chatgpt_sentiment"] = _normalize_sentiment(sentiment.get("sentiment"))
        summary["chatgpt_confidence"] = float(sentiment.get("confidence", 0.0))
        summary["chatgpt_reasoning"] = sentiment.get("reasoning")

    if mode == "synternet":
        synternet = SyntoshiApexSentimentAnalyzer(symbol=symbol)
        try:
            synternet_result = synternet.analyze_market(market_data)
            sentiment = synternet_result
        except Exception:
            summary["decision"] = "rejected"
            summary["reason"] = "synternet_error"
            return None, summary
    elif mode == "dual":
        synternet = SyntoshiApexSentimentAnalyzer(symbol=symbol)
        synternet_result = None
        try:
            import concurrent.futures

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(synternet.analyze_market, market_data)
            try:
                synternet_result = future.result(timeout=synternet_timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
            except Exception as exc:
                logger.error("Synternet analyze_market failed: %s", exc)
                synternet_result = None
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            synternet_result = None

        if synternet_result:
            summary["synternet_sentiment"] = _normalize_sentiment(synternet_result.get("sentiment"))
            summary["synternet_confidence"] = float(synternet_result.get("confidence", 0.0))
            summary["synternet_reasoning"] = synternet_result.get("reasoning")

        if not sentiment:
            logger.info("Dual mode aborted: missing ChatGPT result")
            summary["decision"] = "rejected"
            summary["reason"] = "chatgpt_invalid"
            return None, summary
        if not synternet_result:
            logger.info("Dual mode aborted: missing Synternet result")
            chatgpt_sentiment = summary["chatgpt_sentiment"]
            ta_sentiment = summary.get("ta_sentiment")
            if chatgpt_sentiment in ["bullish", "bearish"] and ta_sentiment:
                if (chatgpt_sentiment == "bullish" and ta_sentiment == "bullish") or (
                    chatgpt_sentiment == "bearish" and ta_sentiment == "bearish"
                ):
                    summary["decision"] = "accepted"
                    summary["reason"] = "synternet_invalid_fallback"
                    summary["position_scale"] = 0.5
                    bridge = SentimentBridge(symbol=symbol)
                    intent = bridge.build_intent(sentiment, market_data)
                    if intent:
                        intent.metadata["size_multiplier"] = 0.5
                        summary["ta_sentiment"] = "bullish" if intent.side == "BUY" else "bearish"
                        return intent, summary
                    summary["decision"] = "rejected"
                    summary["reason"] = "intent_not_built"
                    return None, summary
                summary["decision"] = "rejected"
                summary["reason"] = "ta_mismatch"
                return None, summary
            summary["decision"] = "rejected"
            summary["reason"] = "synternet_invalid"
            return None, summary

        chatgpt_sentiment = summary["chatgpt_sentiment"]
        synternet_sentiment = summary["synternet_sentiment"]

        logger.info(
            "Dual mode: chatgpt=%s (%.2f), synternet=%s (%.2f)",
            chatgpt_sentiment,
            summary["chatgpt_confidence"],
            synternet_sentiment,
            summary["synternet_confidence"],
        )

        if chatgpt_sentiment == "neutral" or synternet_sentiment == "neutral":
            logger.info("Dual mode rejected: neutral sentiment")
            summary["decision"] = "rejected"
            summary["reason"] = "neutral_sentiment"
            return None, summary

        if chatgpt_sentiment != synternet_sentiment:
            logger.info("Dual mode rejected: sentiment mismatch")
            summary["decision"] = "rejected"
            summary["reason"] = "sentiment_mismatch"
            return None, summary

        sentiment = _unify_trade_setup(sentiment, synternet_result)

    bridge = SentimentBridge(symbol=symbol)
    intent = bridge.build_intent(sentiment, market_data)
    if intent:
        summary["ta_sentiment"] = "bullish" if intent.side == "BUY" else "bearish"
    summary["decision"] = "accepted" if intent else "rejected"
    summary["reason"] = "" if intent else "intent_not_built"
    return intent, summary
