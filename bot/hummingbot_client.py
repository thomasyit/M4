import os
from typing import Dict, List, Any
import requests


class HummingbotAPIClient:
    def __init__(self):
        self.base_url = os.getenv("HB_URL", "http://localhost:8000").rstrip("/")
        self.api_prefix = os.getenv("HB_API_PREFIX", "/api/v1").rstrip("/")
        self.username = os.getenv("HB_USER", "admin")
        self.password = os.getenv("HB_PASS", "admin")
        self.rules_endpoint = os.getenv("HB_RULES_ENDPOINT", "/connectors/trading_rules")
        self.orders_endpoint = os.getenv("HB_ORDERS_ENDPOINT", "/orders")
        self.order_status_endpoint = os.getenv("HB_ORDER_STATUS_ENDPOINT", "/orders/status")
        self.positions_endpoint = os.getenv("HB_POSITIONS_ENDPOINT", "/positions")

        self.session = requests.Session()
        self.session.auth = (self.username, self.password)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    def get_trading_rules(self, connector_name: str, trading_pairs: List[str]) -> Dict[str, Any]:
        params = {
            "connector_name": connector_name,
            "trading_pairs": ",".join(trading_pairs),
        }
        response = self.session.get(self._url(self.rules_endpoint), params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(self._url(self.orders_endpoint), json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_order_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.get(self._url(self.order_status_endpoint), params=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_positions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.get(self._url(self.positions_endpoint), params=payload, timeout=15)
        response.raise_for_status()
        return response.json()
