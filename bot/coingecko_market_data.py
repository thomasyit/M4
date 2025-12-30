"""CoinGecko market data fetcher.

Fallback data source when Binance data is unavailable.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoMarketData:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

    def get_current_price(self, symbol: str = "bitcoin") -> Optional[Dict[str, Any]]:
        try:
            url = f"{COINGECKO_BASE_URL}/simple/price"
            params = {
                "ids": symbol,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_7d_change": "true",
            }

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if symbol in data:
                result = data[symbol]
                return {
                    "price": result.get("usd"),
                    "market_cap": result.get("usd_market_cap"),
                    "volume_24h": result.get("usd_24h_vol"),
                    "change_24h": result.get("usd_24h_change"),
                    "change_7d": result.get("usd_7d_change"),
                }

            logger.error("Symbol %s not found in CoinGecko response", symbol)
            return None

        except Exception as e:
            logger.error("Error fetching current price from CoinGecko: %s", e)
            return None

    def get_ohlcv_data(self, symbol: str = "bitcoin", days: int = 90) -> Optional[list]:
        try:
            url = f"{COINGECKO_BASE_URL}/coins/{symbol}/ohlc"
            params = {"vs_currency": "usd", "days": days}

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if data:
                return data

            logger.error("No OHLCV data returned from CoinGecko")
            return None

        except Exception as e:
            logger.error("Error fetching OHLCV from CoinGecko: %s", e)
            return None

    def get_market_chart(self, symbol: str = "bitcoin", days: int = 30) -> Optional[Dict[str, Any]]:
        try:
            url = f"{COINGECKO_BASE_URL}/coins/{symbol}/market_chart"
            params = {"vs_currency": "usd", "days": days}

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if data:
                return {
                    "prices": data.get("prices", []),
                    "volumes": data.get("volumes", []),
                    "market_caps": data.get("market_caps", []),
                }

            logger.error("No market chart data returned from CoinGecko")
            return None

        except Exception as e:
            logger.error("Error fetching market chart from CoinGecko: %s", e)
            return None

    def get_global_data(self) -> Optional[Dict[str, Any]]:
        try:
            url = f"{COINGECKO_BASE_URL}/global"

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if data:
                result = data.get("data", {})
                return {
                    "btc_dominance": result.get("btc_dominance"),
                    "eth_dominance": result.get("eth_dominance"),
                    "total_market_cap_usd": result.get("total_market_cap", {}).get("usd"),
                    "total_volume_usd": result.get("total_volume", {}).get("usd"),
                }

            logger.error("No global data returned from CoinGecko")
            return None

        except Exception as e:
            logger.error("Error fetching global data from CoinGecko: %s", e)
            return None

    def compile_fallback_market_data(self, symbol: str = "bitcoin") -> Optional[Dict[str, Any]]:
        try:
            current = self.get_current_price(symbol)
            ohlcv = self.get_ohlcv_data(symbol, days=90)
            chart = self.get_market_chart(symbol, days=30)
            global_data = self.get_global_data()

            if not current:
                logger.error("Failed to get current price from CoinGecko")
                return None

            return {
                "symbol": symbol.upper(),
                "current_price": current.get("price"),
                "market_cap": current.get("market_cap"),
                "volume_24h": current.get("volume_24h"),
                "change_24h": current.get("change_24h"),
                "change_7d": current.get("change_7d"),
                "ohlcv_data": ohlcv,
                "market_chart": chart,
                "global": global_data,
                "data_source": "CoinGecko (Fallback)",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Error compiling fallback market data: %s", e)
            return None

    def close(self):
        self.session.close()
