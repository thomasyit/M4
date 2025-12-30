"""Syntoshi Apex Thinking - Comprehensive Cryptocurrency Intelligence Engine.

Advanced hybrid model combining:
- Real-time market data (Allora predictions, prices)
- Blockchain intelligence (Nansen, multi-chain analysis)
- DeFi & Yield optimization data (DefiLlama, Yield Yak)
- Multi-chain cross-chain analysis

Uses Synternet's decentralized data streams for maximum insight.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SyntoshiApexSentimentAnalyzer:
    """Syntoshi Apex Thinking - Comprehensive crypto intelligence analyzer.

    Uses Syntoshi Apex Thinking model with access to:
    - Real-time prices via Allora (5m & 8h predictions)
    - Nansen blockchain intelligence
    - DeFi data (DefiLlama, Yield Yak)
    - Multi-chain analysis
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        """Initialize the Syntoshi Apex sentiment analyzer.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
        """
        self.symbol = symbol
        self.base_symbol = symbol.replace("USDT", "")
        self.synternet_api_key = os.getenv("SYNTERNET_API_KEY", "")
        base_url = os.getenv("SYNTERNET_API_URL", "")
        if base_url and "/api/v1/chat/completions" not in base_url:
            base_url = base_url.rstrip("/") + "/api/v1/chat/completions"
        self.synternet_api_url = base_url
        self.chatgpt_api_key = os.getenv("CHATGPT_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.binance_futures_url = "https://fapi.binance.com"

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that might contain markdown or other formatting.

        Args:
            text: Text that might contain JSON

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        import re

        # First try to parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*\n?(.*?)\n?```',  # ```json ... ```
            r'```\s*\n?(.*?)\n?```',       # ``` ... ```
            r'\{.*\}',                      # Direct JSON object
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_text = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_text)
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def analyze_market(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Public entrypoint used by bridge."""
        logger.info("Starting Synternet AI sentiment analysis (Thinking only)...")
        return self.analyze_with_syntoshi_thinking(market_data)


    def analyze_with_syntoshi_thinking(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data using Syntoshi Thinking model.

        This is the primary model - comprehensive crypto market analysis with 4h forecast.

        Args:
            market_data: Market data dictionary

        Returns:
            Sentiment analysis result or None if error
        """
        if not self.synternet_api_key or not self.synternet_api_url:
            logger.error("❌ Synternet API key or URL not configured")
            return None

        try:
            # Build comprehensive market analysis prompt
            market_summary = self._prepare_data_summary(market_data)

            prompt = """You are Syntoshi Apex Thinking, a comprehensive cryptocurrency intelligence engine powered by Synternet's decentralized data.

YOUR SPECIALIZATION: Whale behavior + on-chain flows + institutional divergence analysis
Access to: Real-time prices, Allora predictions, Nansen blockchain data, DeFi metrics, ETF flows

MARKET DATA:
""" + market_summary + """

═══ ANALYSIS FRAMEWORK ═══

STEP 1: DETECT MARKET REGIME FROM WHALE BEHAVIOR
Classify regime based exclusively on smart money activity:

**ACCUMULATION**: Whales buying continuously, exchange outflows (>3k BTC withdrawn), holder concentration ↑
**DISTRIBUTION**: Whales selling consistently, exchange inflows (>3k BTC deposited), holder concentration ↓
**TRENDING**: Clear directional pressure from whales + price momentum, no mean reversion signals
**RANGING**: Mixed whale activity, choppy flows, no clear accumulation/distribution
**RECOVERY**: Price rebounding + whale re-entry, exchange outflows picking up, positive sentiment shift

STEP 2: IDENTIFY INSTITUTIONAL VS RETAIL DIVERGENCE
Critical for mean reversion setups:

Compare three data streams:
- **Whale Wallets** (Nansen): Large address activity direction
- **ETF Flows** (Bitcoin ETF data): Institutional buying/selling sentiment
- **Funding Rates + Taker Ratio**: Retail leverage positioning

DIVERGENCE PATTERNS:
- Whales accumulating BUT ETF outflows = Institutional dumping while whales buy = STRONG BULLISH (reversal setup)
- Whales distributing BUT ETF inflows = Institutional buying while whales dump = STRONG BEARISH (reversal setup)
- Funding positive + high leverage BUT whale accumulation = Overleveraged longs at top = BEARISH
- Funding negative BUT whale distribution = Covering shorts before distribution = NEUTRAL

STEP 3: SENTIMENT + REGIME OUTPUT
Use whale regime + divergence pattern to output sentiment and regime only.
Do NOT output entry/stop/TP or any price levels.

ADDITIONAL WHALE FLOW FRAMEWORK (NANSEN FOCUS):
1. **Whale Flow Velocity**: Are accumulations accelerating or decelerating?
2. **Exchange Flow Patterns**: Which whales are entering/exiting exchanges? (Selling/buying signals)
3. **Holder Concentration Trends**: Are whales concentrating wealth or diluting it?
4. **Micro-Structure**: Size and clustering of whale buys vs sells (conviction indicator)

Whale signal rules:
- Accelerating accumulation + outflows + concentration ↑ = STRONG BULLISH
- Accelerating distribution + inflows + concentration ↓ = STRONG BEARISH
- Steady activity with mixed flows = NEUTRAL
- Pattern reversal detected = REGIME CHANGE SIGNAL

Validation rules:
- Use ONLY Nansen data provided in market data section
- If whale activity is unclear, add "insufficient_onchain" to data_quality_flags
- Quantify flows (e.g., "3 large sell orders", not "whale selling")
- Do not invent whale flows not supported by data
- Do NOT mention browser/network errors (e.g., "internet explorer errors")

═══ STRICT OUTPUT RULES ═══
✓ Confidence = Function of signal alignment (1-2 aligned = 0.55-0.65, 3+ aligned = 0.75+)
✓ Always cite Nansen data when describing regime
✓ Reference ETF flows when discussing divergence
✓ Regime must be one of: ACCUMULATION|DISTRIBUTION|TRENDING|RANGING|RECOVERY
✓ If ETF/Nansen data missing, set confidence <= 0.5 and add "insufficient_onchain"

REQUIRED OUTPUT (return valid JSON only):
```json
{
  "analysis_model": "Syntoshi Apex Thinking",
  "sentiment": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "market_regime": "accumulation|distribution|trending|ranging|recovery",
  "reasoning": "2-4 sentence summary citing inputs",
  "data_quality_flags": ["insufficient_onchain"]
}
```

CRITICAL: Return valid JSON only. No markdown, no explanations, no conversational text."""

            headers = {
                "Authorization": f"Bearer {self.synternet_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "synternet/syntoshi-apex-thinking",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = requests.post(
                self.synternet_api_url,
                json=payload,
                headers=headers,
                timeout=int(os.getenv("SYNTERNET_HTTP_TIMEOUT", "20")),
                stream=True  # Handle streaming response
            )
            response.raise_for_status()

            # Collect streaming chunks
            content_chunks = []
            line_count = 0
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    # Decode bytes to string
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    if line_count <= 3:  # Log first 3 lines for debugging
                        logger.debug(f"📥 Line {line_count}: {line_str[:150]}")

                    # Skip [DONE] marker
                    if line_str.strip() == "data: [DONE]":
                        continue

                    # Skip empty lines
                    if not line_str.strip():
                        continue

                    # Strip "data: " prefix if present
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]  # Remove "data: " prefix

                    try:
                        chunk = json.loads(line_str)
                        if chunk and "choices" in chunk and len(chunk.get("choices", [])) > 0:
                            choice = chunk["choices"][0]
                            # Extract content from delta or message
                            if "delta" in choice and "content" in choice.get("delta", {}):
                                content_chunks.append(choice["delta"]["content"])
                                logger.debug(f"📝 Captured content chunk: {choice['delta']['content'][:50]}...")
                            elif "message" in choice and "content" in choice.get("message", {}):
                                content_chunks.append(choice["message"]["content"])
                                logger.debug(f"📝 Captured message chunk: {choice['message']['content'][:50]}...")
                    except json.JSONDecodeError as e:
                        if line_count <= 3:
                            logger.debug(f"⚠️ JSON decode error on line: {line_str[:100]}")
                        continue

            logger.info(f"📊 Collected {len(content_chunks)} content chunks")
            if content_chunks:
                result_text = "".join(content_chunks).strip()
                if result_text:
                    logger.info("🧾 Synternet raw response: %s", result_text[:1200])

                # First try to extract JSON (in case Synternet returns structured data)
                result = self._extract_json_from_text(result_text)
                if result:
                    if not result.get("reasoning"):
                        result.setdefault("data_quality_flags", []).append("insufficient_data")
                        result["sentiment"] = "neutral"
                        result["confidence"] = min(float(result.get("confidence", 0.5)), 0.5)
                    logger.info("✅ Syntoshi Thinking analysis completed (JSON format)")
                    # Log trade setup for debugging
                    return result

                logger.warning("⚠️ No JSON found in Synternet response; skipping Synternet result")
                return None
            else:
                logger.error("❌ No content received from Synternet API")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Syntoshi Thinking response as JSON: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Syntoshi Thinking API error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in Syntoshi Thinking: {e}")
            return None


    def _prepare_data_summary(self, market_data: Dict[str, Any]) -> str:
        """Prepare a concise summary of market data for the AI prompt.

        Args:
            market_data: Full market data dictionary

        Returns:
            Formatted data summary string
        """
        if not market_data:
            return "❌ No market data available"

        summary = f"\nSymbol: {self.symbol}\n"
        summary += f"Timestamp: {market_data.get('timestamp', 'N/A')}\n"

        # Safe access to data_sources
        data_sources = market_data.get('data_sources', [])
        summary += f"Data Sources Collected: {len(data_sources) if data_sources else 0}\n"
        summary += f"Sources: {', '.join(data_sources) if data_sources else 'None'}\n\n"

        # Latest price from 1h candles
        candles_1h = market_data.get("candles_1h")
        if candles_1h and len(candles_1h) > 0:
            latest_candle = candles_1h[-1]
            summary += f"Latest 1h Close: ${float(latest_candle[4]):.2f}\n"

        # Open interest
        if market_data.get("open_interest"):
            oi = market_data["open_interest"]
            summary += f"Open Interest: {float(oi.get('openInterest', 0)):.2f}\n"

        # Top trader positions
        if market_data.get("top_trader_positions"):
            ttp = market_data["top_trader_positions"]
            summary += f"Top Traders Long/Short Ratio: {float(ttp.get('longShortRatio', 0)):.2f}\n"

        # Funding rates
        if market_data.get("funding_rates") and len(market_data["funding_rates"]) > 0:
            latest_funding = market_data["funding_rates"][-1]
            summary += f"Latest Funding Rate: {float(latest_funding.get('fundingRate', 0)) * 100:.4f}%\n"

        summary += "\nETF FLOWS\n"
        etf_flows_data = market_data.get("etf_flows")
        if etf_flows_data:
            summary += f"30-Day ETF Flow Summary: {etf_flows_data.get('summary', 'N/A')}\n"
            if etf_flows_data.get("trend"):
                summary += f"ETF Flow Trend: {etf_flows_data.get('trend')}\n"
            if etf_flows_data.get("strength"):
                summary += f"ETF Flow Strength: {etf_flows_data.get('strength')}\n"
            if etf_flows_data.get("sentiment"):
                summary += f"ETF Sentiment: {etf_flows_data.get('sentiment')}\n"
        else:
            summary += "ETF flow data: unavailable\n"

        summary += f"\nTotal Candles Analyzed: {len(market_data.get('candles_1h', []))} hours\n"

        return summary


def main():
    """Main entry point for local testing."""
    analyzer = SyntoshiApexSentimentAnalyzer(symbol="BTCUSDT")

    # Compile market data
    market_data = analyzer.compile_market_data()

    # Save raw market data for inspection
    with open("/tmp/market_data_synternet_btcusdt.json", "w") as f:
        json.dump(market_data, f, indent=2, default=str)
    logger.info("📁 Raw market data saved to /tmp/market_data_synternet_btcusdt.json")

    # Analyze sentiment with Synternet
    sentiment_result = analyzer.analyze_market(market_data)

    if sentiment_result:
        logger.info("\n✅ SYNTERNET SENTIMENT ANALYSIS RESULT:")
        logger.info(json.dumps(sentiment_result, indent=2))

        # Save sentiment result
        with open("/tmp/sentiment_result_synternet_btcusdt.json", "w") as f:
            json.dump(sentiment_result, f, indent=2)
        logger.info("📁 Sentiment result saved to /tmp/sentiment_result_synternet_btcusdt.json")

        return sentiment_result
    else:
        logger.error("\n❌ Sentiment analysis failed")
        return None


if __name__ == "__main__":
    main()

    def _parse_conversational_response(
        self,
        text: str,
        market_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Parse conversational text response into structured trading data.

        Args:
            text: Conversational text response from Synternet

        Returns:
            Structured JSON dictionary or None if parsing fails
        """
        import re

        try:
            result = {
                "analysis_model": "Syntoshi Apex Thinking",
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "",
                "market_regime": "unknown",
                "data_quality_flags": [],
            }

            # Extract sentiment (look for bullish/bearish/neutral keywords)
            text_lower = text.lower()
            if re.search(r'\b(bullish|uptrend|upward|long|buy)\b', text_lower):
                result["sentiment"] = "bullish"
            elif re.search(r'\b(bearish|downtrend|downward|short|sell)\b', text_lower):
                result["sentiment"] = "bearish"

            # Extract confidence (look for percentage or confidence indicators)
            confidence_match = re.search(r'confidence[:\s]+(\d+)%?', text_lower)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1)) / 100
            elif "strong" in text_lower or "high confidence" in text_lower:
                result["confidence"] = 0.8
            elif "moderate" in text_lower:
                result["confidence"] = 0.6
            elif "low" in text_lower or "weak" in text_lower:
                result["confidence"] = 0.4

            # Extract market regime based on whale activity patterns
            # Look for regime keywords in the response
            if re.search(r'\b(accumulation|accumulating|whale.*accum|smart money.*adding)\b', text_lower):
                result["market_regime"] = "accumulation"
                logger.debug(f"Market regime detected: ACCUMULATION")
            elif re.search(r'\b(distribution|distributing|whale.*distrib|smart money.*reducing)\b', text_lower):
                result["market_regime"] = "distribution"
                logger.debug(f"Market regime detected: DISTRIBUTION")
            elif re.search(r'\b(trending|uptrend|downtrend|sustained.*directional|clear.*direction)\b', text_lower):
                result["market_regime"] = "trend"
                logger.debug(f"Market regime detected: TREND")
            elif re.search(r'\b(range|ranging|choppy|oscillating|sideways|neutral activity|defensive|consolidat)\b', text_lower):
                result["market_regime"] = "range"
                logger.debug(f"Market regime detected: RANGE")
            elif re.search(r'\b(recovery|rebounding|re-entry|whale.*recovering)\b', text_lower):
                result["market_regime"] = "recovery"
                logger.debug(f"Market regime detected: RECOVERY")
            elif re.search(r'\b(volatil|wild|erratic|extreme|swing|turbulent)\b', text_lower):
                result["market_regime"] = "volatile"
                logger.debug(f"Market regime detected: VOLATILE")

            # If no regime detected, infer from market data volatility + sentiment
            if result["market_regime"] == "unknown":
                # Check market data for volatility metrics
                volatility_regime = self._infer_regime_from_volatility(market_data)
                if volatility_regime:
                    result["market_regime"] = volatility_regime
                    logger.debug(f"Market regime inferred from market volatility: {volatility_regime.upper()}")
                elif result["sentiment"] == "bullish":
                    result["market_regime"] = "accumulation"
                    logger.debug(f"Market regime inferred from BULLISH sentiment: ACCUMULATION")
                elif result["sentiment"] == "bearish":
                    result["market_regime"] = "distribution"
                    logger.debug(f"Market regime inferred from BEARISH sentiment: DISTRIBUTION")
                else:
                    result["market_regime"] = "range"
                    logger.debug(f"Market regime inferred from NEUTRAL sentiment: RANGE")

            # Extract reasoning (try multiple approaches)
            reasoning = ""

            # Clean up the text first - remove markdown widgets and special syntax
            import re
            cleaned_text = re.sub(r'::[a-z]+\{[^}]+\}', '', text)  # Remove ::widget{...}
            cleaned_text = re.sub(r'\*\*Generated[^\n]*', '', cleaned_text)  # Remove **Generated...
            cleaned_text = cleaned_text.strip()

            # Try to extract first meaningful paragraph
            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
            if paragraphs:
                # Filter out paragraphs that are just headers (start with #, ##, etc.)
                content_paragraphs = [p for p in paragraphs if not p.startswith('#')]
                if content_paragraphs:
                    reasoning = content_paragraphs[0]
                else:
                    reasoning = paragraphs[0] if paragraphs else ""

            # Fallback: try to extract lines that look like analysis
            if not reasoning:
                lines = [l.strip() for l in cleaned_text.split('\n') if l.strip() and len(l.strip()) > 50]
                # Filter out headers and metadata
                analysis_lines = [l for l in lines if not l.startswith('#') and not l.startswith('**Generated')]
                if analysis_lines:
                    reasoning = ' '.join(analysis_lines[:3])  # Take first 3 meaningful lines

            # Final fallback: use truncated cleaned text
            if not reasoning:
                reasoning = cleaned_text[:500].strip()

            result["reasoning"] = reasoning

            if not result["reasoning"] or len(result["reasoning"]) < 20:
                result["data_quality_flags"].append("insufficient_data")
                result["sentiment"] = "neutral"
                result["confidence"] = 0.0

            logger.info(f"✅ Parsed conversational response: sentiment={result['sentiment']}, confidence={result['confidence']}, reasoning_length={len(result['reasoning'])}")
            if not result['reasoning']:
                logger.warning(f"⚠️ No reasoning extracted. Response preview: {text[:200]}")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to parse conversational response: {e}")
            return None

    def fetch_klines(self, interval: str = "1h", limit: int = 100) -> Optional[List[List]]:
        """Fetch candlestick data from Binance.

        Args:
            interval: Timeframe (15m, 1h, 1d)
            limit: Number of candles to fetch

        Returns:
            List of OHLCV candles or None if error
        """
        try:
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(
                f"{self.binance_base_url}/klines",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"✅ Fetched {len(response.json())} {interval} candles for {self.symbol}")
            return response.json()
        except Exception as e:
            logger.error(f"❌ Error fetching {interval} klines: {e}")
            return None

    def fetch_open_interest(self) -> Optional[Dict[str, Any]]:
        """Fetch open interest data from Binance Futures."""
        try:
            params = {"symbol": self.symbol}
            response = requests.get(
                f"{self.binance_futures_url}/fapi/v1/openInterest",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"✅ Fetched open interest: {data.get('openInterest', 'N/A')}")
            return data
        except Exception as e:
            logger.error(f"❌ Error fetching open interest: {e}")
            return None

    def fetch_top_trader_positions(self) -> Optional[Dict[str, Any]]:
        """Fetch top trader long/short position ratio."""
        try:
            params = {
                "symbol": self.symbol,
                "period": "5m",
                "limit": 1
            }
            response = requests.get(
                f"{self.binance_futures_url}/futures/data/topLongShortPositionRatio",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"✅ Fetched top trader positions: {data[0]}")
                return data[0]
            logger.warning("⚠️ No top trader position data available")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching top trader positions: {e}")
            return None

    def fetch_top_trader_accounts(self) -> Optional[Dict[str, Any]]:
        """Fetch top trader account long/short ratio."""
        try:
            params = {
                "symbol": self.symbol,
                "period": "5m",
                "limit": 1
            }
            response = requests.get(
                f"{self.binance_futures_url}/futures/data/topLongShortAccountRatio",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"✅ Fetched top trader accounts: {data[0]}")
                return data[0]
            logger.warning("⚠️ No top trader account data available")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching top trader accounts: {e}")
            return None

    def fetch_taker_ratio(self) -> Optional[Dict[str, Any]]:
        """Fetch taker buy/sell ratio."""
        try:
            params = {
                "symbol": self.symbol,
                "period": "5m",
                "limit": 1
            }
            response = requests.get(
                f"{self.binance_futures_url}/futures/data/takerlongshortRatio",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"✅ Fetched taker ratio: {data[0]}")
                return data[0]
            logger.warning("⚠️ No taker ratio data available")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching taker ratio: {e}")
            return None

    def fetch_funding_rates(self, limit: int = 10) -> Optional[List[Dict]]:
        """Fetch recent funding rates."""
        try:
            params = {
                "symbol": self.symbol,
                "limit": limit
            }
            response = requests.get(
                f"{self.binance_futures_url}/fapi/v1/fundingRate",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"✅ Fetched {len(data)} funding rates")
            return data
        except Exception as e:
            logger.error(f"❌ Error fetching funding rates: {e}")
            return None

    def compile_market_data(self) -> Dict[str, Any]:
        """Compile all market data for analysis."""
        logger.info(f"\n📊 Compiling market data for {self.symbol}...\n")

        market_data = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_sources": []
        }

        # Fetch candlestick data
        logger.info("📈 Fetching candlestick data...")
        market_data["candles_15m"] = self.fetch_klines("15m", 100)
        if market_data["candles_15m"]:
            market_data["data_sources"].append("Binance 15m Candles")

        market_data["candles_1h"] = self.fetch_klines("1h", 100)
        if market_data["candles_1h"]:
            market_data["data_sources"].append("Binance 1h Candles")

        market_data["candles_1d"] = self.fetch_klines("1d", 100)
        if market_data["candles_1d"]:
            market_data["data_sources"].append("Binance 1d Candles")

        # Fetch market microstructure
        logger.info("🔍 Fetching market microstructure...")
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

        # Fetch funding rates
        logger.info("💰 Fetching funding rates...")
        market_data["funding_rates"] = self.fetch_funding_rates(10)
        if market_data["funding_rates"]:
            market_data["data_sources"].append("Binance Funding Rates")

        logger.info(f"\n✅ Data compilation complete. Sources: {len(market_data['data_sources'])}")
        logger.info(f"📦 Data sources: {', '.join(market_data['data_sources'])}\n")

        return market_data

    def _infer_regime_from_volatility(self, market_data: Dict[str, Any]) -> Optional[str]:
        """Infer market regime from volatility metrics in market data.

        Args:
            market_data: Market data dictionary with price info

        Returns:
            Market regime string or None
        """
        try:
            # Extract volatility from market data if available
            candles_1h = market_data.get("candles_1h", [])
            if len(candles_1h) >= 24:
                # Calculate 24h volatility range
                highs_24h = [float(c[2]) for c in candles_1h[-24:]]
                lows_24h = [float(c[3]) for c in candles_1h[-24:]]
                high_24h = max(highs_24h)
                low_24h = min(lows_24h)
                volatility_pct = ((high_24h - low_24h) / low_24h) * 100

                # Classify regime based on volatility
                if volatility_pct > 10:
                    logger.debug(f"High volatility detected: {volatility_pct:.2f}% → VOLATILE regime")
                    return "volatile"
                elif volatility_pct > 5:
                    logger.debug(f"Medium volatility detected: {volatility_pct:.2f}% → TRENDING regime")
                    return "trend"
                else:
                    logger.debug(f"Low volatility detected: {volatility_pct:.2f}% → RANGE regime")
                    return "range"
        except Exception as e:
            logger.debug(f"Could not infer regime from volatility: {str(e)}")

        return None


    def analyze_with_syntoshi_nansen(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze whale movements using Syntoshi Nansen model.

        Tracks whale movements (>$1M), accumulation vs distribution patterns.

        Args:
            market_data: Market data dictionary

        Returns:
            Whale analysis result or None if error
        """
        if not self.synternet_api_key or not self.synternet_api_url:
            logger.error("❌ Synternet API key or URL not configured")
            return None

        try:
            market_summary = self._prepare_data_summary(market_data)

            prompt = """You are a whale flow analyst specializing in Bitcoin on-chain intelligence via Nansen.
Your role: Deep dive into smart money accumulation/distribution mechanics, not just regime classification.

FOCUS AREAS (Complementary to regime classification):
1. **Whale Flow Velocity**: Are accumulations accelerating or decelerating?
2. **Exchange Flow Patterns**: Which whales are entering/exiting exchanges? (Selling/buying signals)
3. **Holder Concentration Trends**: Are whales concentrating wealth or diluting it?
4. **Micro-Structure**: Size and clustering of whale buys vs sells (conviction indicator)

MARKET DATA:
""" + market_summary + """

═══ WHALE ANALYSIS FRAMEWORK ═══

STEP 1: EXTRACT WHALE FLOW SIGNALS
From Nansen data, identify:
- **Accumulation Pattern**: Consecutive large buys from smart money addresses
  - Accelerating: More large buys in last 12h vs prior 12h = strong conviction
  - Steady: Consistent buying throughout 24h = methodical accumulation
  - Decelerating: Large buys slowing down = losing interest
- **Distribution Pattern**: Consecutive large sells from smart money
  - Accelerating: More large sells in last 12h = distribution intensifying
  - Steady: Consistent selling = planned reduction
  - Decelerating: Selling slowing = distribution complete
- **Exchange Flows**:
  - Large inflows (whales depositing to exchange) = likely upcoming sells
  - Large outflows (whales withdrawing from exchange) = likely hodling/accumulating
  - Directional reversal (inflow→outflow) = change in whale conviction

STEP 2: ASSESS HOLDER CONCENTRATION
- **Concentration ↑**: Whales taking larger % of supply = consolidation (bullish long-term)
- **Concentration ↓**: Whales reducing % of supply = distribution (bearish)
- **Stable**: Whales maintaining holdings = neutral/range-bound

STEP 3: GENERATE SENTIMENT FROM WHALE MECHANICS
- If: Accelerating accumulation + outflows + concentration ↑ = STRONG BULLISH
- If: Accelerating distribution + inflows + concentration ↓ = STRONG BEARISH
- If: Steady activity with mixed flows = NEUTRAL
- If: Pattern reversal detected = REGIME CHANGE SIGNAL

═══ STRICT VALIDATION RULES ═══
✓ Use ONLY Nansen data provided in market data section
✓ If whale activity is unclear, mark "insufficient on-chain data"
✓ Quantify everything: "5% of supply in whale wallets" not "many whales"
✓ Reference specific patterns: "3 large sell orders" vs "whale selling"
✓ DO NOT invent whale flows not supported by data

═══ OUTPUT JSON ═══
{{
  "analysis_model": "Syntoshi Nansen",
  "sentiment": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary of whale flow signals and exchange dynamics",
  "whale_analysis": {{
    "accumulation_pattern": "accelerating|steady|decelerating|absent",
    "distribution_pattern": "accelerating|steady|decelerating|absent",
    "exchange_flow_direction": "outflows|inflows|mixed",
    "holder_concentration_trend": "increasing|stable|decreasing",
    "flow_conviction_score": 0.0-1.0
  }},
  "highest_probability_trade": {{
    "direction": "long|short",
    "entry_price": 92000,
    "stop_loss": 89250,
    "take_profit": 93500,
    "risk_reward_ratio": 2.5,
    "rationale": "Whale flow based (e.g., accelerating outflows at 92k support indicates whale conviction to accumulate)"
  }},
  "key_levels": {{
    "resistance": ["$93,500", "$94,000", "$95,000"],
    "support": ["$91,000", "$90,000", "$89,250"]
  }},
  "whale_invalidation_signals": [
    "reversal of whale flow direction (e.g., outflows→inflows)",
    "accumulation deceleration if whales stop buying",
    "holder concentration breakdown if whales start reducing"
  ],
  "data_quality_score": 0.85
}}

Output ONLY valid JSON. No explanations or conversational text."""

            headers = {
                "Authorization": f"Bearer {self.synternet_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "synternet/syntoshi-nansen-api-latest",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = requests.post(
                self.synternet_api_url,
                json=payload,
                headers=headers,
                timeout=int(os.getenv("SYNTERNET_HTTP_TIMEOUT", "20")),
                stream=True  # Handle streaming response
            )
            response.raise_for_status()

            # Collect streaming chunks
            content_chunks = []
            line_count = 0
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    # Decode bytes to string
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    if line_count <= 3:  # Log first 3 lines for debugging
                        logger.debug(f"📥 Line {line_count}: {line_str[:150]}")

                    # Skip [DONE] marker
                    if line_str.strip() == "data: [DONE]":
                        continue

                    # Skip empty lines
                    if not line_str.strip():
                        continue

                    # Strip "data: " prefix if present
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]  # Remove "data: " prefix

                    try:
                        chunk = json.loads(line_str)
                        if chunk and "choices" in chunk and len(chunk.get("choices", [])) > 0:
                            choice = chunk["choices"][0]
                            # Extract content from delta or message
                            if "delta" in choice and "content" in choice.get("delta", {}):
                                content_chunks.append(choice["delta"]["content"])
                                logger.debug(f"📝 Captured content chunk: {choice['delta']['content'][:50]}...")
                            elif "message" in choice and "content" in choice.get("message", {}):
                                content_chunks.append(choice["message"]["content"])
                                logger.debug(f"📝 Captured message chunk: {choice['message']['content'][:50]}...")
                    except json.JSONDecodeError as e:
                        if line_count <= 3:
                            logger.debug(f"⚠️ JSON decode error on line: {line_str[:100]}")
                        continue

            logger.info(f"📊 Collected {len(content_chunks)} content chunks")
            if content_chunks:
                result_text = "".join(content_chunks).strip()

                # First try to extract JSON (in case Synternet returns structured data)
                result = self._extract_json_from_text(result_text)
                if result:
                    logger.info("✅ Syntoshi Nansen analysis completed (JSON format)")
                    # Log trade setup for debugging
                    trade_setup = result.get("highest_probability_trade", {})
                    if trade_setup:
                        logger.info(f"📊 Trade Setup Found (JSON): {json.dumps(trade_setup, indent=2)}")
                    else:
                        logger.warning("⚠️ No trade setup in Syntoshi Nansen JSON response")
                    return result

                # Fall back to parsing conversational response
                logger.warning("⚠️ No JSON found in Synternet response; skipping Synternet result")
                return None
            else:
                logger.error("❌ No content received from Synternet API")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Syntoshi Nansen response as JSON: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Syntoshi Nansen API error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in Syntoshi Nansen: {e}")
            return None

    def merge_synternet_results(self, thinking: Dict[str, Any], nansen: Dict[str, Any]) -> Dict[str, Any]:
        """Merge Syntoshi Thinking and Nansen results into unified analysis.

        Args:
            thinking: Syntoshi Thinking analysis result
            nansen: Syntoshi Nansen whale analysis result

        Returns:
            Merged analysis dictionary
        """
        logger.info("🔗 Merging Synternet analyses...")

        # Combine reasoning from both models
        thinking_reason = thinking.get("reasoning", "")
        nansen_reason = nansen.get("reasoning", "")

        combined_reasoning = ""
        if thinking_reason and nansen_reason:
            combined_reasoning = f"Market Analysis: {thinking_reason}\n\nWhale Activity: {nansen_reason}"
        elif thinking_reason:
            combined_reasoning = thinking_reason
        elif nansen_reason:
            combined_reasoning = nansen_reason

        merged = {
            "analysis_model": "Synternet (Thinking + Nansen)",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sentiment": thinking.get("sentiment", "neutral"),
            "confidence": thinking.get("confidence", 0.5),
            "market_regime": thinking.get("market_regime", "unknown"),
            "reasoning": combined_reasoning,
            "probabilities": thinking.get("probabilities", {}),
            "forecast_4h": thinking.get("forecast_4h", {}),
            "whale_sentiment": nansen.get("whale_sentiment", "neutral"),
            "whale_confidence": nansen.get("confidence", 0.5),
            "whale_activity": nansen.get("whale_activity", {}),
            "whale_insights": nansen.get("whale_insights", []),
            "smart_money_signal": nansen.get("smart_money_signal", "neutral"),
            "market_signals": thinking.get("market_signals", {}),
            "risk_management_notes": thinking.get("risk_notes", []),
            "key_levels": thinking.get("key_levels", {"resistance": [], "support": []}),
            "data_quality_score": min(
                thinking.get("data_quality_score", 0.5),
                nansen.get("data_quality_score", 0.5)
            )
        }

        # Generate actionable trade setup based on both analyses
        if thinking.get("sentiment") == nansen.get("whale_sentiment"):
            # Both agree - stronger signal
            direction = "long" if thinking.get("sentiment") == "bullish" else "short"
            confidence_boost = 0.15
        else:
            # They disagree - weaker signal
            direction = "long" if thinking.get("sentiment") == "bullish" else "short"
            confidence_boost = 0

        # Build unified trade setup from both analyses
        thinking_trade = thinking.get("highest_probability_trade", {})
        nansen_trade = nansen.get("highest_probability_trade", {})

        merged["highest_probability_trade"] = {
            "direction": direction,
            "entry_price": thinking_trade.get("entry_price") or nansen_trade.get("entry_price"),
            "stop_loss": thinking_trade.get("stop_loss") or nansen_trade.get("stop_loss"),
            "take_profit": thinking_trade.get("take_profit") or nansen_trade.get("take_profit"),
            "risk_reward_ratio": thinking_trade.get("risk_reward_ratio") or nansen_trade.get("risk_reward_ratio"),
            "setup_confidence": min(1.0, thinking.get("confidence", 0.5) + confidence_boost),
            "rationale": f"Syntoshi Thinking: {thinking.get('sentiment')} | Whale Signal: {nansen.get('whale_sentiment')}"
        }

        logger.info("✅ Merging complete")
        return merged

    def analyze_with_chatgpt_fallback(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fallback to ChatGPT if Synternet models fail.

        Args:
            market_data: Market data dictionary

        Returns:
            Sentiment analysis result or None if error
        """
        if not self.chatgpt_api_key:
            logger.error("❌ ChatGPT API key not configured")
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.chatgpt_api_key)

            prompt = """Analyze the cryptocurrency market data provided and generate a JSON response with the following structure:
{
  "sentiment": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation",
  "highest_probability_trade": {
    "direction": "long|short",
    "entry_price": "price level",
    "stop_loss": "price level",
    "take_profit": "price level",
    "risk_reward_ratio": 0.0,
    "rationale": "why this trade setup"
  },
  "risk_management_notes": ["note1", "note2"],
  "key_levels": {
    "resistance": ["level1", "level2"],
    "support": ["level1", "level2"]
  },
  "data_quality_score": 0.0-1.0
}"""

            data_summary = self._prepare_data_summary(market_data)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst. Provide analysis in valid JSON format only."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nMarket Data:\n{data_summary}"
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content
            if not result_text:
                logger.error("❌ No content received from Synternet API")
                return None

            result = self._extract_json_from_text(result_text)
            if not result:
                logger.error("❌ Failed to extract JSON from ChatGPT fallback. Response: %s", result_text[:200])
                return None
            return result
        except Exception as e:
            logger.error(f"❌ Failed to parse ChatGPT response as JSON: {e}")
            return None
