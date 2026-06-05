from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ArenaGoResponse:
    ok: bool
    payload: dict[str, Any]
    status_code: int | None = None
    error: str = ""


class ArenaGoClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://arenago.ru",
        timeout: float = 20.0,
        retries: int = 2,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": self.api_key}

    def submit_order(self, *, direction: str, secid: str, quantity: int, bot: str) -> ArenaGoResponse:
        return self._request(
            "POST",
            "/api/submit_order",
            json={"direction": direction, "secid": secid, "quantity": int(quantity), "bot": bot},
        )

    def positions(self, portfolio: str) -> ArenaGoResponse:
        return self._request("GET", f"/api/positions/{portfolio}")

    def bots(self) -> ArenaGoResponse:
        return self._request("GET", "/api/bots")

    def _request(self, method: str, path: str, **kwargs) -> ArenaGoResponse:
        url = f"{self.base_url}{path}"
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(method, url, headers=self.headers, timeout=self.timeout, **kwargs)
                try:
                    payload = response.json()
                except Exception:
                    payload = {"raw_text": response.text}
                ok = 200 <= response.status_code < 300 and not (isinstance(payload, dict) and payload.get("error"))
                if ok:
                    return ArenaGoResponse(True, payload if isinstance(payload, dict) else {"payload": payload}, response.status_code)
                last_error = str(payload.get("error", payload)) if isinstance(payload, dict) else str(payload)
                if response.status_code < 500:
                    return ArenaGoResponse(False, payload if isinstance(payload, dict) else {"payload": payload}, response.status_code, last_error)
            except Exception as exc:
                last_error = str(exc)
            if attempt < self.retries:
                time.sleep(0.5 * (attempt + 1))
        return ArenaGoResponse(False, {}, None, last_error)

