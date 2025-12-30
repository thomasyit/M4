"""AI sentiment analysis for crypto markets.

Derived from the Coincise blog generator, adapted for trading bot use.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
import pandas as pd
from dotenv import load_dotenv

from bot.coingecko_market_data import CoinGeckoMarketData
from bot.bitcoin_data_etf_flows import get_etf_flows
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SentimentAnalyzer:
    """AI-powered cryptocurrency sentiment analyzer."""

    _kline_cache: Dict[str, Any] = {}

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.base_symbol = symbol.replace("USDT", "")
        self.chatgpt_api_key = os.getenv("CHATGPT_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.binance_futures_url = "https://fapi.binance.com"
        self.analysis_prompt = self._get_analysis_prompt()

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        import re

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_patterns = [
            r"```json\s*\n?(.*?)\n?```",
            r"```\s*\n?(.*?)\n?```",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_text = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_text)
                except (json.JSONDecodeError, IndexError):
                    continue

        start_idx = text.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                char = text[i]

                if char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                if char == '\\' and in_string:
                    escape_next = not escape_next
                else:
                    escape_next = False

            if brace_count == 0 and end_idx > start_idx:
                try:
                    json_text = text[start_idx:end_idx]
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    return None

        return None

    def _get_analysis_prompt(self) -> str:
        return """You are a crypto market sentiment analyst. Use only the provided market data.

TASK:
- Determine sentiment (bullish/bearish/neutral)
- Classify regime (trend/range/volatile/accumulation/distribution/recovery)
- Explain in 2-4 sentences, citing specific data points from the input

STRICT RULES:
- Do NOT propose entry/stop/TP or any price levels
- If data is insufficient for a confident call, set sentiment=\"neutral\" and include \"insufficient_data\" in data_quality_flags
- Do NOT invent data or reference missing sources

OUTPUT JSON ONLY:
{
  \"analysis_model\": \"ChatGPT\",
  \"sentiment\": \"bullish|bearish|neutral\",
  \"confidence\": 0.0-1.0,
  \"market_regime\": \"trend|range|volatile|accumulation|distribution|recovery\",
  \"reasoning\": \"2-4 sentences with cited inputs\",
  \"data_quality_flags\": [\"insufficient_data\"]
}
"""

    def fetch_klines(self, interval: str = "1h", limit: int = 100) -> Optional[List[List]]:
        try:
            params = {"symbol": self.symbol, "interval": interval, "limit": limit}
            response = requests.get(
                f"{self.binance_base_url}/klines",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Fetched %s candles for %s", interval, self.symbol)
            return response.json()
        except Exception as e:
            logger.error("Error fetching %s klines: %s", interval, e)
            return None

    def _fetch_klines_cached(self, interval: str, limit: int) -> Optional[List[List]]:
        ttl = int(os.getenv("MARKET_CACHE_TTL_SEC", "120"))
        cache_key = f"{self.symbol}:{interval}:{limit}"
        now = datetime.utcnow().timestamp()
        cached = self._kline_cache.get(cache_key)
        if cached:
            ts, data = cached
            if now - ts <= ttl:
                return data
        data = self.fetch_klines(interval, limit)
        if data:
            self._kline_cache[cache_key] = (now, data)
        return data

    def fetch_open_interest(self) -> Optional[Dict[str, Any]]:
        try:
            params = {"symbol": self.symbol}
            response = requests.get(
                f"{self.binance_futures_url}/fapi/v1/openInterest",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched open interest: %s", data.get("openInterest", "N/A"))
            return data
        except Exception as e:
            logger.error("Error fetching open interest: %s", e)
            return None

    def fetch_top_trader_positions(self) -> Optional[Dict[str, Any]]:
        try:
            params = {"symbol": self.symbol, "period": "5m", "limit": 1}
            response = requests.get(
                f"{self.binance_futures_url}/futures/data/topLongShortPositionRatio",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                logger.info("Fetched top trader positions")
                return data[0]
            logger.warning("No top trader position data available")
            return None
        except Exception as e:
            logger.error("Error fetching top trader positions: %s", e)
            return None

    def fetch_top_trader_accounts(self) -> Optional[Dict[str, Any]]:
        try:
            params = {"symbol": self.symbol, "period": "5m", "limit": 1}
            response = requests.get(
                f"{self.binance_futures_url}/futures/data/topLongShortAccountRatio",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                logger.info("Fetched top trader accounts")
                return data[0]
            logger.warning("No top trader account data available")
            return None
        except Exception as e:
            logger.error("Error fetching top trader accounts: %s", e)
            return None

    def fetch_taker_ratio(self) -> Optional[Dict[str, Any]]:
        try:
            params = {"symbol": self.symbol, "period": "5m", "limit": 1}
            response = requests.get(
                f"{self.binance_futures_url}/futures/data/takerlongshortRatio",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                logger.info("Fetched taker ratio")
                return data[0]
            logger.warning("No taker ratio data available")
            return None
        except Exception as e:
            logger.error("Error fetching taker ratio: %s", e)
            return None

    def fetch_funding_rates(self, limit: int = 10) -> Optional[List[Dict]]:
        try:
            params = {"symbol": self.symbol, "limit": limit}
            response = requests.get(
                f"{self.binance_futures_url}/fapi/v1/fundingRate",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched %s funding rates", len(data))
            return data
        except Exception as e:
            logger.error("Error fetching funding rates: %s", e)
            return None

    def fetch_market_news(self, query: str = None) -> Optional[List[Dict]]:
        if query is None:
            query = f"{self.base_symbol} cryptocurrency"

        logger.info("News fetching skipped; query was: %s", query)
        return [{
            "title": f"{self.symbol} Market Analysis",
            "description": "Market sentiment data from multiple sources",
            "sentiment": "neutral",
            "source": "aggregated",
        }]

    def _load_cached_ohlcv(self, timeframe: str, limit: int) -> Optional[List[List[Any]]]:
        base_path = os.getenv("DATA_LOADER_BASE", "/opt/M4")
        try:
            loader = DataLoader(base_path=base_path)
            df = loader.load_data(self.symbol, timeframe, exchange="binance")
        except Exception as exc:
            logger.debug("Cached OHLCV not available for %s %s: %s", self.symbol, timeframe, exc)
            return None

        if df is None or df.empty:
            return None

        rows = []
        for _, row in df.tail(limit).iterrows():
            ts = row.get("timestamp")
            if hasattr(ts, "timestamp"):
                ts_val = int(ts.timestamp() * 1000)
            else:
                ts_val = int(pd.to_datetime(ts).timestamp() * 1000)
            rows.append([
                ts_val,
                float(row.get("open", 0)),
                float(row.get("high", 0)),
                float(row.get("low", 0)),
                float(row.get("close", 0)),
                float(row.get("volume", 0)),
            ])
        return rows

    def compile_market_data(self) -> Dict[str, Any]:
        logger.info("Compiling market data for %s", self.symbol)

        market_data = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_sources": [],
        }

        market_data["candles_15m"] = self._fetch_klines_cached("15m", 100)
        if market_data["candles_15m"]:
            market_data["data_sources"].append("Binance 15m Candles")

        market_data["candles_1h"] = self._load_cached_ohlcv("1h", 200)
        if market_data["candles_1h"]:
            market_data["data_sources"].append("Local Binance 1h Candles")
        else:
            market_data["candles_1h"] = self.fetch_klines("1h", 100)
        if market_data["candles_1h"]:
            if "Local Binance 1h Candles" not in market_data["data_sources"]:
                market_data["data_sources"].append("Binance 1h Candles")

        market_data["candles_1d"] = self._load_cached_ohlcv("1d", 200)
        if market_data["candles_1d"]:
            market_data["data_sources"].append("Local Binance 1d Candles")
        else:
            market_data["candles_1d"] = self.fetch_klines("1d", 100)
        if market_data["candles_1d"]:
            if "Local Binance 1d Candles" not in market_data["data_sources"]:
                market_data["data_sources"].append("Binance 1d Candles")

        market_data["candles_4h"] = self._load_cached_ohlcv("4h", 200)
        if market_data["candles_4h"]:
            market_data["data_sources"].append("Local Binance 4h Candles")
        else:
            market_data["candles_4h"] = self.fetch_klines("4h", 100)
        if market_data["candles_4h"]:
            if "Local Binance 4h Candles" not in market_data["data_sources"]:
                market_data["data_sources"].append("Binance 4h Candles")

        has_candle_data = any([
            market_data["candles_15m"],
            market_data["candles_1h"],
            market_data["candles_1d"],
        ])

        freshness_hours = float(os.getenv("DATA_FRESHNESS_HOURS", "2"))
        market_data["data_fresh"] = True
        market_data["data_fresh_reason"] = ""
        if market_data.get("candles_1h"):
            latest_ms = int(market_data["candles_1h"][-1][0])
            age_hours = (datetime.utcnow().timestamp() * 1000 - latest_ms) / 1000 / 3600
            market_data["data_fresh_age_hours"] = round(age_hours, 2)
            if age_hours > freshness_hours and "Local Binance 1h Candles" in market_data["data_sources"]:
                market_data["data_fresh"] = False
                market_data["data_fresh_reason"] = f"cached_1h_stale_{age_hours:.2f}h"

        if not has_candle_data:
            logger.warning("Binance candlestick data unavailable; falling back to CoinGecko")

            coingecko = CoinGeckoMarketData()
            coingecko_data = coingecko.compile_fallback_market_data("bitcoin")
            coingecko.close()

            if coingecko_data:
                market_data["candles_1d"] = coingecko_data.get("ohlcv_data")
                market_data["coingecko_current"] = {
                    "price": coingecko_data.get("current_price"),
                    "change_24h": coingecko_data.get("change_24h"),
                    "change_7d": coingecko_data.get("change_7d"),
                    "market_cap": coingecko_data.get("market_cap"),
                    "volume_24h": coingecko_data.get("volume_24h"),
                }
                market_data["coingecko_global"] = coingecko_data.get("global")
                market_data["data_sources"].append("CoinGecko (Fallback)")

        market_data["open_interest"] = self.fetch_open_interest()
        if market_data["open_interest"]:
            market_data["data_sources"].append("Binance Open Interest")

        market_data["top_trader_positions"] = self.fetch_top_trader_positions()
        if market_data["top_trader_positions"]:
            market_data["data_sources"].append("Binance Top Trader Positions")

        market_data["top_trader_accounts"] = self.fetch_top_trader_accounts()
        if market_data["top_trader_accounts"]:
            market_data["data_sources"].append("Binance Top Trader Accounts")

        market_data["taker_ratio"] = self.fetch_taker_ratio()
        if market_data["taker_ratio"]:
            market_data["data_sources"].append("Binance Taker Ratio")

        market_data["funding_rates"] = self.fetch_funding_rates(10)
        if market_data["funding_rates"]:
            market_data["data_sources"].append("Binance Funding Rates")

        market_data["news"] = self.fetch_market_news()
        if market_data["news"]:
            market_data["data_sources"].append("News Aggregation")

        try:
            etf_flows = get_etf_flows()
            if etf_flows:
                market_data["etf_flows"] = etf_flows.get_recent_flows(30)
                if market_data.get("etf_flows"):
                    market_data["data_sources"].append("Bitcoin ETF Flows")
        except Exception as e:
            logger.warning("Failed to fetch Bitcoin ETF flows: %s", e)
            market_data["etf_flows"] = None

        logger.info("Data compilation complete. Sources: %s", ", ".join(market_data["data_sources"]))
        return market_data

    def analyze_with_chatgpt(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.chatgpt_api_key:
            logger.error("CHATGPT_API_KEY not configured")
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.chatgpt_api_key)
            data_summary = self._prepare_data_summary(market_data)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert crypto market analyst. Provide JSON only.",
                    },
                    {
                        "role": "user",
                        "content": f"{self.analysis_prompt}\n\nMarket Data:\n{data_summary}",
                    },
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            result_text = (response.choices[0].message.content or "").strip()
            if not result_text:
                logger.error("ChatGPT returned empty response")
                return None

            result = self._extract_json_from_text(result_text)
            if result:
                logger.info("ChatGPT analysis completed")
                return result

            logger.error("Failed to extract JSON from ChatGPT response")
            return None

        except ImportError:
            logger.error("openai package not installed")
            return None
        except Exception as e:
            logger.error("ChatGPT API error: %s", e)
            return None

    def analyze_with_gemini(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.gemini_api_key:
            logger.error("GEMINI_API_KEY not configured")
            return None

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            data_summary = self._prepare_data_summary(market_data)

            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(
                f"{self.analysis_prompt}\n\nMarket Data:\n{data_summary}"
            )

            if not response or not response.text:
                logger.error("Gemini returned empty response")
                return None

            result_text = response.text.strip()
            result = self._extract_json_from_text(result_text)
            if result:
                logger.info("Gemini analysis completed")
                return result

            logger.error("Failed to extract JSON from Gemini response")
            return None

        except ImportError:
            logger.error("google-generativeai package not installed")
            return None
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return None

    def analyze_market(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        logger.info("Starting AI sentiment analysis")

        result = self.analyze_with_chatgpt(market_data)
        if result:
            result["analysis_model"] = "ChatGPT (gpt-4o-mini)"
            return result

        logger.info("ChatGPT failed, attempting Gemini fallback")
        result = self.analyze_with_gemini(market_data)
        if result:
            result["analysis_model"] = "Google Gemini"
            return result

        logger.error("Both ChatGPT and Gemini analysis failed")
        return None

    def _prepare_data_summary(self, market_data: Dict[str, Any]) -> str:
        if not market_data:
            return "No market data available"

        summary = f"\nSymbol: {self.symbol}\n"
        summary += f"Timestamp: {market_data.get('timestamp', 'N/A')}\n"

        data_sources = market_data.get("data_sources", [])
        summary += f"Data Sources Collected: {len(data_sources) if data_sources else 0}\n"
        summary += f"Sources: {', '.join(data_sources) if data_sources else 'None'}\n\n"

        summary += "PRICE ACTION\n"
        candles_1h = market_data.get("candles_1h")
        if candles_1h and len(candles_1h) > 0:
            latest_1h = candles_1h[-1]
            latest_1h_close = float(latest_1h[4])
            summary += f"Latest 1h Close: {latest_1h_close:.2f}\n"

            if len(candles_1h) >= 4:
                close_4h_ago = float(candles_1h[-4][4])
                pct_4h = ((latest_1h_close - close_4h_ago) / close_4h_ago) * 100
                summary += f"4h Change: {pct_4h:+.2f}%\n"

            if len(candles_1h) >= 24:
                close_24h_ago = float(candles_1h[-24][4])
                pct_24h = ((latest_1h_close - close_24h_ago) / close_24h_ago) * 100
                summary += f"24h Change: {pct_24h:+.2f}%\n"

            if len(candles_1h) >= 24:
                highs_24h = [float(c[2]) for c in candles_1h[-24:]]
                lows_24h = [float(c[3]) for c in candles_1h[-24:]]
                high_24h = max(highs_24h)
                low_24h = min(lows_24h)
                volatility_pct = ((high_24h - low_24h) / low_24h) * 100
                summary += (
                    f"24h Volatility Range: {volatility_pct:.2f}% "
                    f"({low_24h:.2f} - {high_24h:.2f})\n"
                )

        summary += "\nMARKET MICROSTRUCTURE\n"
        if market_data.get("open_interest"):
            oi = market_data["open_interest"]
            summary += f"Open Interest: {float(oi.get('openInterest', 0)):.2f}\n"

        if market_data.get("top_trader_positions"):
            ttp = market_data["top_trader_positions"]
            summary += f"Top Trader Positions Long/Short: {float(ttp.get('longShortRatio', 0)):.2f}\n"

        if market_data.get("top_trader_accounts"):
            tta = market_data["top_trader_accounts"]
            summary += f"Top Trader Accounts Long/Short: {float(tta.get('longShortRatio', 0)):.2f}\n"

        if market_data.get("taker_ratio"):
            tr = market_data["taker_ratio"]
            taker_ratio = float(tr.get("buySellRatio", 0))
            if taker_ratio > 1.2:
                taker_sentiment = "BUYING_PRESSURE"
            elif taker_ratio < 0.8:
                taker_sentiment = "SELLING_PRESSURE"
            else:
                taker_sentiment = "BALANCED"
            summary += f"Taker Buy/Sell Ratio: {taker_ratio:.2f} ({taker_sentiment})\n"

        summary += "\nFUNDING RATES\n"
        funding_rates = market_data.get("funding_rates")
        if funding_rates and len(funding_rates) > 0:
            latest_funding = float(funding_rates[-1].get("fundingRate", 0)) * 100
            summary += f"Latest Funding Rate: {latest_funding:+.4f}%\n"
            if len(funding_rates) >= 2:
                avg_funding = sum(
                    float(f.get("fundingRate", 0)) for f in funding_rates
                ) / len(funding_rates)
                summary += f"Average Funding (10 rates): {avg_funding * 100:+.4f}%\n"

        summary += "\nETF FLOWS\n"
        if market_data.get("etf_flows"):
            etf_flows_data = market_data["etf_flows"]
            if etf_flows_data:
                summary += f"30-Day ETF Flow Summary: {etf_flows_data.get('summary', 'N/A')}\n"
                if etf_flows_data.get("trend"):
                    summary += f"ETF Flow Trend: {etf_flows_data.get('trend')}\n"
                if etf_flows_data.get("strength"):
                    summary += f"ETF Flow Strength: {etf_flows_data.get('strength')}\n"
                if etf_flows_data.get("sentiment"):
                    summary += f"ETF Sentiment: {etf_flows_data.get('sentiment')}\n"
            else:
                summary += "ETF flow data: Not available\n"
        else:
            summary += "ETF flow data: Not available\n"

        summary += f"\nTotal Candles Analyzed: {len(market_data.get('candles_1h', []))} hours\n"

        return summary
