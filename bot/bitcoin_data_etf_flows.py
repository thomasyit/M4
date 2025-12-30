"""Bitcoin ETF flows data provider."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import time
import subprocess

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    import urllib.request
    import urllib.error

logger = logging.getLogger(__name__)


class BitcoinDataETFFlows:
    BASE_URL = "https://bitcoin-data.com/v1/etf-flow-btc"

    def __init__(self):
        self.cache = {}
        self.last_fetch = None
        self.cache_duration = 300

    def get_etf_flows_raw(self) -> Optional[List[Dict[str, Any]]]:
        if self.cache and self.last_fetch:
            age = (datetime.utcnow() - self.last_fetch).total_seconds()
            if age < self.cache_duration:
                return self.cache

        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    self.BASE_URL,
                    "-H",
                    "accept: application/json",
                    "-H",
                    "User-Agent: Mozilla/5.0",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                self.cache = data
                self.last_fetch = datetime.utcnow()
                return data
        except Exception as e:
            logger.debug("Curl fetch failed: %s", e)

        try:
            if HAS_HTTPX:
                time.sleep(1)
                response = httpx.get(
                    self.BASE_URL,
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                response.raise_for_status()
                data = response.json()
            else:
                time.sleep(1)
                req = urllib.request.Request(
                    self.BASE_URL,
                    headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode("utf-8"))

            self.cache = data
            self.last_fetch = datetime.utcnow()
            return data

        except Exception as e:
            logger.warning("Failed to fetch bitcoin-data.com ETF flows: %s", e)
            if self.cache:
                return self.cache
            return None

    def get_recent_flows(self, days: int = 30) -> Optional[Dict[str, Any]]:
        data = self.get_etf_flows_raw()
        if not data:
            return None

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent = [
            record
            for record in data
            if datetime.strptime(record["d"], "%Y-%m-%d") >= cutoff_date
        ]

        if not recent:
            return None

        flows = [float(r["etfFlow"]) for r in recent]
        total_flow = sum(flows)
        avg_daily = total_flow / len(flows)
        positive_days = sum(1 for f in flows if f > 0)
        negative_days = sum(1 for f in flows if f < 0)

        abs_flow = abs(total_flow)
        if abs_flow > 50000:
            strength = "strong"
        elif abs_flow > 20000:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "total_flow": total_flow,
            "avg_daily_flow": avg_daily,
            "positive_days": positive_days,
            "negative_days": negative_days,
            "trend": "bullish" if total_flow > 0 else "bearish",
            "strength": strength,
            "latest_date": recent[-1]["d"],
            "records": recent,
        }

    def get_etf_sentiment(self, days: int = 30) -> Optional[Dict[str, Any]]:
        flows_data = self.get_recent_flows(days=days)
        if not flows_data:
            return None

        total = flows_data["total_flow"]
        avg = flows_data["avg_daily_flow"]
        pos_days = flows_data["positive_days"]
        total_days = len(flows_data["records"])
        pos_ratio = pos_days / total_days if total_days > 0 else 0.5

        if flows_data["trend"] == "bullish" and flows_data["strength"] == "strong":
            confidence = 0.75
            sentiment = "STRONG_BULLISH"
        elif flows_data["trend"] == "bullish" and flows_data["strength"] in ["moderate", "weak"]:
            confidence = 0.60
            sentiment = "WEAK_BULLISH"
        elif flows_data["trend"] == "bearish" and flows_data["strength"] == "strong":
            confidence = 0.75
            sentiment = "STRONG_BEARISH"
        elif flows_data["trend"] == "bearish" and flows_data["strength"] in ["moderate", "weak"]:
            confidence = 0.60
            sentiment = "WEAK_BEARISH"
        else:
            confidence = 0.50
            sentiment = "NEUTRAL"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "total_inflow_btc": total,
            "avg_daily_inflow_btc": avg,
            "inflow_days_percentage": pos_ratio * 100,
            "analysis_period_days": total_days,
            "latest_data_date": flows_data["latest_date"],
            "summary": (
                f"ETF flows over last {days} days: {total:+.0f} BTC "
                f"({flows_data['trend']}, {flows_data['strength']} strength)"
            ),
        }

    def format_for_prompt(self) -> str:
        sentiment = self.get_etf_sentiment(days=30)
        if not sentiment:
            return "ETF Flow Data: Not available"

        return (
            "ETF Flow Analysis (30-day):\n"
            f"- Trend: {sentiment['sentiment']}\n"
            f"- Total Flow: {sentiment['total_inflow_btc']:+.0f} BTC "
            f"(confidence: {sentiment['confidence']:.0%})\n"
            f"- Average Daily: {sentiment['avg_daily_inflow_btc']:+.2f} BTC\n"
            f"- Inflow Days: {sentiment['inflow_days_percentage']:.1f}%\n"
            f"- Latest Data: {sentiment['latest_data_date']}\n"
            f"- Interpretation: {sentiment['summary']}"
        )


_instance = None


def get_etf_flows() -> BitcoinDataETFFlows:
    global _instance
    if _instance is None:
        _instance = BitcoinDataETFFlows()
    return _instance
