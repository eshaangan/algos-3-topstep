"""
Thin HTTP client for the ProjectX API used to connect a TopstepX account.

This module intentionally hides raw HTTP details behind a small, typed
interface so that the rest of the trading framework stays broker-agnostic.

As of the tsxapipy integration, this client prefers to use the official
TopstepX / ProjectX Python library for authentication, account discovery and
order routing when available, while preserving the original public interface
used by the rest of the framework. When the library is not available or a
call fails, the implementation transparently falls back to the original
`requests`-based HTTP implementation.

Notes
-----
- Authentication uses the TopstepX `/api/Auth/loginKey` endpoint with the
  `TOPSTEPX_USERNAME` and `TOPSTEPX_PROJECTX_API_KEY` environment variables
  to obtain a short-lived JWT session token.
- Order placement uses `/api/Order/place` and requires you to provide the
  target `accountId` and `contractId` via environment variables:
  `TOPSTEPX_ACCOUNT_ID` and `TOPSTEPX_CONTRACT_ID`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import logging
import requests

LOGGER = logging.getLogger(__name__)

PROJECTX_DEFAULT_BASE_URL = "https://api.topstepx.com"


class ProjectXClientError(RuntimeError):
    """Raised when an HTTP call to ProjectX fails or returns an error payload."""


class ProjectXRateLimitError(ProjectXClientError):
    """ProjectX returned HTTP 429 (too many requests)."""


class ProjectXDataUnavailableError(ProjectXClientError):
    """ProjectX refused to provide the requested data (e.g. live bars not enabled)."""

    def __init__(self, message: str, payload: Dict[str, Any], live: bool) -> None:
        super().__init__(message)
        self.payload = payload
        self.live = live


@dataclass
class AccountState:
    """Minimal live account snapshot used by risk management."""

    account_id: str
    equity: float
    balance: float
    open_pnl: float
    realized_pnl: float
    daily_pnl: float = 0.0
    open_positions: int = 0


@dataclass
class PositionState:
    """Open position information as returned by ProjectX."""

    symbol: str
    quantity: int
    entry_price: float
    unrealized_pnl: float


@dataclass
class OrderState:
    """Simplified order representation used by the live execution engine."""

    order_id: str
    symbol: str
    side: str
    quantity: int
    status: str
    avg_fill_price: Optional[float] = None


@dataclass
class OrderSnapshot:
    """Detailed view of an order as returned by search endpoints."""

    order_id: int
    account_id: int
    contract_id: str
    symbol_id: Optional[str]
    status: int
    order_type: int
    side: int
    size: int
    filled_volume: int
    filled_price: Optional[float]
    limit_price: Optional[float]
    stop_price: Optional[float]
    creation_timestamp: datetime
    update_timestamp: datetime
    custom_tag: Optional[str] = None


@dataclass
class BracketInstruction:
    """Server-side bracket instruction expressed in ticks."""

    ticks: int
    order_type: int = 1  # default limit bracket

    def to_payload(self) -> Dict[str, int]:
        """Return the payload expected by ProjectX."""

        if int(self.ticks) == 0:
            raise ValueError("Bracket ticks must be non-zero (use sign for direction).")
        if self.order_type not in {1, 2, 4, 5, 6, 7}:
            raise ValueError("Bracket order_type must map to a valid ProjectX order type.")
        return {
            "ticks": int(self.ticks),
            "type": int(self.order_type),
        }


@dataclass
class HistoryBar:
    """Historical/topstep bar returned by ProjectX /api/History/retrieveBars."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ProjectXClient:
    """
    Synchronous HTTP client for the ProjectX API.

    Parameters
    ----------
    base_url:
        Base URL for the ProjectX REST API.
    timeout_seconds:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: float = 5.0,
        contract_id: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> None:
        # Credentials for the raw HTTP client remain environment-driven so that
        # existing deployments continue to work unchanged. The tsxapipy client
        # has its own configuration model (usually via .env); that is initialised
        # separately below.
        self._api_key = os.getenv("TOPSTEPX_PROJECTX_API_KEY")
        self._username = os.getenv("TOPSTEPX_USERNAME")
        if not self._api_key or not self._username:
            raise EnvironmentError(
                "TOPSTEPX_PROJECTX_API_KEY and TOPSTEPX_USERNAME must be set in the environment. "
                "Add them to your .env and ensure it is loaded before starting live trading."
            )

        # Trading context (account + contract) – required for order placement.
        account_id = account_id or os.getenv("TOPSTEPX_ACCOUNT_ID")
        env_contract_id = os.getenv("TOPSTEPX_CONTRACT_ID")
        resolved_contract_id = contract_id or env_contract_id
        if not account_id or not resolved_contract_id:
            raise EnvironmentError(
                "TOPSTEPX_ACCOUNT_ID and TOPSTEPX_CONTRACT_ID must be set in the environment "
                "or supplied to ProjectXClient. Use /api/Account/search and /api/Contract/available "
                "once to discover the correct IDs."
            )
        self._account_id = int(account_id)
        self._contract_id = resolved_contract_id

        self._base_url = (base_url or os.getenv("TOPSTEPX_PROJECTX_BASE_URL") or PROJECTX_DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout_seconds
        self._token: Optional[str] = None

        # Optional tsxapipy client used for richer, strongly-typed access to the
        # TopstepX API. These attributes are None when the library is not
        # installed or fails to configure itself from the environment.
        self._api_client: Optional[Any] = None
        self._order_placer: Optional[Any] = None
        # Force raw HTTP for now to ensure stop/target brackets are sent; tsxapipy
        # skips stop/limit payloads in our current usage.
        self._disable_tsxapipy = True

        # Authenticate once to obtain a session token for the legacy HTTP path.
        # This remains the primary implementation for endpoints that are not
        # yet wired through tsxapipy (e.g. the History polling used by the live
        # data iterator).
        self._authenticate()

        if self._disable_tsxapipy:
            LOGGER.info("TOPSTEPX_DISABLE_TSXAPIPY set; using raw HTTP client only.")
        else:
            # Best-effort initialisation of the official tsxapipy client so that
            # account discovery and order placement can be delegated to it when
            # possible. Failures here are non-fatal and will simply result in the
            # legacy HTTP implementation being used everywhere.
            try:  # pragma: no cover - import-time robustness
                from tsxapipy import APIClient as _TSXAPIClient  # type: ignore[import]
                from tsxapipy import authenticate as _tsx_authenticate  # type: ignore[import]
                from tsxapipy.trading import OrderPlacer as _TSXOrderPlacer  # type: ignore[import]
                from tsxapipy.api.exceptions import (  # type: ignore[import]
                    ConfigurationError,
                    AuthenticationError,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.info("tsxapipy is not available; using raw HTTP ProjectX client only: %s", exc)
            else:
                try:
                    # Perform initial authentication using the same username / API key
                    # that the legacy HTTP client relies on. This ensures we do not
                    # depend on a second set of environment variables for tsxapipy.
                    token, acquired_at = _tsx_authenticate(
                        username=self._username,
                        api_key=self._api_key,
                    )
                    if not token or acquired_at is None:
                        raise AuthenticationError("tsxapipy.authenticate returned no token.")

                    # Construct an APIClient instance using the returned token. We
                    # allow tsxapipy.config to determine the appropriate API_URL so
                    # that TRADING_ENVIRONMENT (LIVE/DEMO) is respected.
                    self._api_client = _TSXAPIClient(
                        initial_token=token,
                        token_acquired_at=acquired_at,
                    )
                    self._order_placer = _TSXOrderPlacer(api_client=self._api_client, account_id=self._account_id)
                    LOGGER.info("Initialised tsxapipy APIClient and OrderPlacer for account id %s.", self._account_id)
                except (ConfigurationError, AuthenticationError) as exc:  # pragma: no cover - networking/config
                    LOGGER.error("Failed to initialise tsxapipy client; falling back to raw HTTP only: %s", exc)
                    self._api_client = None
                    self._order_placer = None

    # ------------------------------------------------------------------ HTTP core
    def _headers(self) -> Dict[str, str]:
        if not self._token:
            raise ProjectXClientError("ProjectX client is not authenticated.")
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_response(self, resp: requests.Response) -> Dict[str, Any]:
        text = resp.text.strip()
        if resp.status_code == 429:
            raise ProjectXRateLimitError("ProjectX rate limit exceeded.")
        if not text:
            raise ProjectXClientError(f"ProjectX error {resp.status_code}: empty response body.")
        try:
            payload = resp.json()
        except ValueError as exc:  # pragma: no cover - networking
            raise ProjectXClientError(f"Invalid JSON from ProjectX ({resp.status_code}): {text[:200]}") from exc
        if not resp.ok or payload.get("error"):
            message = payload.get("error") or payload
            raise ProjectXClientError(f"ProjectX error {resp.status_code}: {message}")
        return payload

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        for attempt in range(2):
            resp = requests.get(url, headers=self._headers(), params=params or {}, timeout=self._timeout)
            if resp.status_code == 401 and attempt == 0:
                LOGGER.warning("ProjectX 401 on GET %s; re-authenticating", path)
                self._token = None
                self._authenticate()
                continue
            return self._handle_response(resp)
        raise ProjectXClientError("Unexpected: GET retry loop exhausted")

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        for attempt in range(2):
            resp = requests.post(url, headers=self._headers(), json=body, timeout=self._timeout)
            if resp.status_code == 401 and attempt == 0:
                LOGGER.warning("ProjectX 401 on POST %s; re-authenticating", path)
                self._token = None
                self._authenticate()
                continue
            return self._handle_response(resp)
        raise ProjectXClientError("Unexpected: POST retry loop exhausted")

    def _delete(self, path: str) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        for attempt in range(2):
            resp = requests.delete(url, headers=self._headers(), timeout=self._timeout)
            if resp.status_code == 401 and attempt == 0:
                LOGGER.warning("ProjectX 401 on DELETE %s; re-authenticating", path)
                self._token = None
                self._authenticate()
                continue
            return self._handle_response(resp)
        raise ProjectXClientError("Unexpected: DELETE retry loop exhausted")

    # ------------------------------------------------------------------ positions
    def search_open_positions(self) -> List[Dict[str, Any]]:
        """
        Return open positions for the configured account.

        Uses /api/Position/searchOpen which responds with:
        {
            "positions": [
                {
                    "id": 6124,
                    "accountId": 536,
                    "contractId": "CON.F.US.GMET.J25",
                    "creationTimestamp": "...",
                    "type": 1,
                    "size": 2,
                    "averagePrice": 1575.75
                }
            ],
            "success": true,
            "errorCode": 0,
            "errorMessage": null
        }
        """

        payload = self._post("/api/Position/searchOpen", {"accountId": self._account_id})
        positions = payload.get("positions") or []
        if not isinstance(positions, list):
            raise ProjectXClientError(f"Unexpected positions payload: {payload}")
        parsed: List[Dict[str, Any]] = []
        for pos in positions:
            try:
                contract_id = pos.get("contractId")
                size = int(pos.get("size", 0))
                pos_type = int(pos.get("type", 1))
                avg_price = float(pos.get("averagePrice", 0.0))
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Skipping malformed position row: %s (%s)", pos, exc)
                continue
            parsed.append(
                {
                    "contract_id": contract_id,
                    "size": size,
                    "type": pos_type,
                    "average_price": avg_price,
                }
            )
        return parsed

    def _authenticate(self) -> None:
        """Obtain a JWT session token via /api/Auth/loginKey."""

        url = f"{self._base_url}/api/Auth/loginKey"
        body = {
            "userName": self._username,
            "apiKey": self._api_key,
        }
        resp = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/plain",
            },
            json=body,
            timeout=self._timeout,
        )
        try:
            payload = resp.json()
        except ValueError as exc:  # pragma: no cover - networking
            raise ProjectXClientError(f"Invalid JSON from loginKey: {exc}") from exc
        if not resp.ok or not payload.get("success") or payload.get("errorCode") not in (0, None):
            raise ProjectXClientError(f"loginKey failed: {payload}")
        token = payload.get("token")
        if not token:
            raise ProjectXClientError("loginKey response missing 'token'.")
        self._token = token

    # -------------------------------------------------------------- public API
    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware (UTC recommended).")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            LOGGER.warning("Invalid timestamp returned by ProjectX: %s", value)
            return None

    def _order_snapshot_from_payload(self, row: Dict[str, Any]) -> OrderSnapshot:
        """Convert raw order JSON into a typed snapshot."""

        created = self._parse_timestamp(row.get("creationTimestamp"))
        updated = self._parse_timestamp(row.get("updateTimestamp"))
        if created is None or updated is None:
            raise ProjectXClientError(f"Order payload missing timestamps: {row}")
        return OrderSnapshot(
            order_id=int(row.get("id")),
            account_id=int(row.get("accountId")),
            contract_id=str(row.get("contractId")),
            symbol_id=row.get("symbolId"),
            status=int(row.get("status")),
            order_type=int(row.get("type")),
            side=int(row.get("side")),
            size=int(row.get("size")),
            filled_volume=int(row.get("fillVolume", row.get("filledVolume", 0))),
            filled_price=float(row["filledPrice"]) if row.get("filledPrice") is not None else None,
            limit_price=float(row["limitPrice"]) if row.get("limitPrice") is not None else None,
            stop_price=float(row["stopPrice"]) if row.get("stopPrice") is not None else None,
            creation_timestamp=created,
            update_timestamp=updated,
            custom_tag=row.get("customTag"),
        )

    def get_account_state(self) -> AccountState:
        """
        Fetch the current account state (equity, balance, PnL).

        Uses /api/Account/search and returns the configured TOPSTEPX_ACCOUNT_ID.
        """

        # Prefer the strongly-typed tsxapipy client when available so that we
        # benefit from its Pydantic models and validation. This integration is
        # intentionally conservative: if anything fails we fall back to the
        # original raw HTTP implementation below.
        if self._api_client is not None:
            try:  # pragma: no cover - depends on external library
                accounts = self._api_client.get_accounts(only_active=True)
            except Exception as exc:
                LOGGER.error("tsxapipy.get_accounts failed; falling back to raw HTTP: %s", exc)
            else:
                selected: Optional[Any] = None
                try:
                    for candidate in accounts or []:
                        # Pydantic model exposes an `id` attribute that maps to
                        # the numeric account id used by the API.
                        if int(getattr(candidate, "id")) == self._account_id:
                            selected = candidate
                            break
                except Exception:
                    selected = None

                if selected is None and accounts:
                    # As a safe fallback, pick the first active account rather
                    # than fail hard, but still log so the operator can correct
                    # configuration.
                    selected = accounts[0]
                    LOGGER.warning(
                        "Configured accountId %s not found in tsxapipy accounts; "
                        "falling back to first active account id=%s.",
                        self._account_id,
                        getattr(selected, "id", "<unknown>"),
                    )

                if selected is not None:
                    # Use getattr with multiple fallbacks to cope with possible
                    # naming differences between API JSON fields and Pydantic
                    # attribute names.
                    equity = float(
                        getattr(
                            selected,
                            "equity",
                            getattr(selected, "balance", 0.0),
                        )
                    )
                    balance = float(
                        getattr(
                            selected,
                            "balance",
                            getattr(selected, "equity", 0.0),
                        )
                    )
                    open_pnl = float(
                        getattr(
                            selected,
                            "open_pnl",
                            getattr(selected, "openPnl", 0.0),
                        )
                    )
                    realized_pnl = float(
                        getattr(
                            selected,
                            "realized_pnl",
                            getattr(selected, "realizedPnl", 0.0),
                        )
                    )
                    # Derive daily PnL if provided; fall back to realized + open.
                    daily_pnl = float(
                        getattr(
                            selected,
                            "daily_pnl",
                            getattr(selected, "dailyPnl", realized_pnl + open_pnl),
                        )
                    )

                    try:
                        open_positions_count = len(self.search_open_positions())
                    except Exception as exc:
                        LOGGER.warning("Failed to fetch open positions via tsxapipy path: %s", exc)
                        open_positions_count = 0

                    return AccountState(
                        account_id=str(getattr(selected, "id")),
                        equity=equity,
                        balance=balance,
                        open_pnl=open_pnl,
                        realized_pnl=realized_pnl,
                        daily_pnl=daily_pnl,
                        open_positions=open_positions_count,
                    )

        # Legacy raw HTTP implementation.
        payload = self._post("/api/Account/search", {"request": {"onlyActiveAccounts": True}})
        accounts = payload.get("accounts") if isinstance(payload, dict) else payload
        if not accounts:
            raise ProjectXClientError("No accounts returned from /api/Account/search.")

        account: Optional[Dict[str, Any]] = None
        for row in accounts:
            try:
                if int(row.get("id")) == self._account_id:
                    account = row
                    break
            except (TypeError, ValueError):
                continue
        if account is None:
            raise ProjectXClientError(f"Configured accountId {self._account_id} not found in Account/search response.")

        daily_pnl = float(
            account.get(
                "dailyPnl",
                account.get(
                    "daily_pnl",
                    account.get("realizedPnl", account.get("realized_pnl", 0.0))
                    + account.get("openPnl", account.get("open_pnl", 0.0)),
                ),
            )
        )

        try:
            open_positions_count = len(self.search_open_positions())
        except Exception as exc:
            LOGGER.warning("Failed to fetch open positions: %s", exc)
            open_positions_count = 0

        return AccountState(
            account_id=str(account["id"]),
            equity=float(account.get("equity", account.get("balance", 0.0))),
            balance=float(account.get("balance", account.get("equity", 0.0))),
            open_pnl=float(account.get("openPnl", account.get("open_pnl", 0.0))),
            realized_pnl=float(account.get("realizedPnl", account.get("realized_pnl", 0.0))),
            daily_pnl=daily_pnl,
            open_positions=open_positions_count,
        )

    def _unit_to_timedelta(self, unit: int, unit_number: int) -> timedelta:
        if unit == 1:
            return timedelta(seconds=unit_number)
        if unit == 2:
            return timedelta(minutes=unit_number)
        if unit == 3:
            return timedelta(hours=unit_number)
        if unit == 4:
            return timedelta(days=unit_number)
        if unit == 5:
            return timedelta(weeks=unit_number)
        if unit == 6:
            # Approximate month as 30 days; acceptable because we only request minutes.
            return timedelta(days=30 * unit_number)
        raise ValueError(f"Unsupported unit value: {unit}")

    def search_contracts(
        self,
        *,
        search_text: str,
        live: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for contracts using the Topstep API.

        Args:
            search_text: The contract symbol or name to search for (e.g., "MES", "MNQ").
            live: Whether to search for live contracts (True) or simulated contracts (False).

        Returns:
            List of contract dictionaries with id, name, description, tickSize, tickValue, etc.
        """
        body = {
            "searchText": search_text,
            "live": live,
        }

        payload = self._post("/api/Contract/search", body)
        if not payload.get("success"):
            error_msg = payload.get("errorMessage") or "Unknown error"
            raise ProjectXClientError(f"Contract search failed: {error_msg}")

        contracts = payload.get("contracts", [])
        return contracts

    def available_contracts(self, *, live: bool = True) -> List[Dict[str, Any]]:
        """
        List available contracts via /api/Contract/available.

        Args:
            live: Whether to request live (True) or sim (False) contracts.

        Returns:
            List of contract dictionaries with id, name, description, tickSize, tickValue, activeContract, symbolId.
        """
        body = {
            "live": bool(live),
        }
        payload = self._post("/api/Contract/available", body)
        if not payload.get("success"):
            error_msg = payload.get("errorMessage") or "Unknown error"
            raise ProjectXClientError(f"Contract available failed: {error_msg}")
        return payload.get("contracts", [])

    def retrieve_bars(
        self,
        *,
        contract_id: Optional[str] = None,
        live: bool = True,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        unit: int = 2,
        unit_number: int = 5,
        limit: int = 500,
        include_partial_bar: bool = False,
    ) -> List[HistoryBar]:
        """
        Fetch historical or live bars via /api/History/retrieveBars.

        Parameters
        ----------
        contract_id:
            ProjectX contract identifier. Defaults to TOPSTEPX_CONTRACT_ID.
        live:
            True for live subscription, False for sim.
        start_time:
            Earliest timestamp to request (timezone-aware).
        end_time:
            Optional end timestamp (timezone-aware). If omitted, ProjectX streams
            through the latest available bar.
        unit / unit_number:
            Aggregation bucket (e.g. unit=2 minute, unit_number=5 -> 5-minute bars).
        limit:
            Max bars to return (<= 20_000 per API contract).
        include_partial_bar:
            Whether to include the still-building bucket.
        """

        if limit <= 0:
            raise ValueError("limit must be positive.")
        if limit > 20_000:
            raise ValueError("limit must be <= 20,000 per ProjectX API contract.")
        if unit not in (1, 2, 3, 4, 5, 6):
            raise ValueError("unit must be between 1 (second) and 6 (month).")
        if unit_number <= 0:
            raise ValueError("unit_number must be positive.")
        if start_time is None:
            raise ValueError("start_time is required for retrieve_bars.")

        contract = contract_id or self._contract_id
        if not contract:
            raise ValueError("contract_id must be provided via argument or environment.")

        window = self._unit_to_timedelta(unit, unit_number * limit)
        computed_end = end_time or (start_time + window)
        if computed_end <= start_time:
            computed_end = start_time + self._unit_to_timedelta(unit, unit_number)

        body: Dict[str, Any] = {
            "contractId": contract,
            "live": bool(live),
            "startTime": self._format_timestamp(start_time),
            "endTime": self._format_timestamp(computed_end),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": bool(include_partial_bar),
        }

        payload = self._post("/api/History/retrieveBars", body)
        success = payload.get("success", True)
        error_code = payload.get("errorCode", 0)
        if not success or error_code not in (0, None):
            raise ProjectXDataUnavailableError(
                f"/api/History/retrieveBars failed: {payload}",
                payload=payload,
                live=live,
            )

        raw_bars = payload.get("bars", [])
        history: List[HistoryBar] = []
        for row in raw_bars:
            timestamp = row.get("t")
            if not timestamp:
                continue
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                LOGGER.warning("Skipping bar with invalid timestamp: %s", timestamp)
                continue
            history.append(
                HistoryBar(
                    timestamp=ts,
                    open=float(row.get("o", 0.0)),
                    high=float(row.get("h", 0.0)),
                    low=float(row.get("l", 0.0)),
                    close=float(row.get("c", 0.0)),
                    volume=float(row.get("v", 0.0)),
                )
            )

        return sorted(history, key=lambda bar: bar.timestamp)

    def get_open_positions(self) -> List[PositionState]:
        """Return currently open positions for the account (not yet implemented)."""

        raise NotImplementedError("get_open_positions is not yet wired to the TopstepX ProjectX API.")

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        client_order_id: Optional[str] = None,
        stop_loss_bracket: Optional[BracketInstruction] = None,
        take_profit_bracket: Optional[BracketInstruction] = None,
        account_id: Optional[int] = None,
        contract_id: Optional[str] = None,
        linked_order_id: Optional[int] = None,
    ) -> OrderState:
        """
        Place a market order via /api/Order/place.

        Notes
        -----
        - This implementation currently sends a simple market order of the
          configured `accountId` and `contractId` with type=2 (market).
        - The `side` parameter is mapped to a numeric side code; you should
          confirm the mapping against the ProjectX documentation.
        - Stop/target brackets are not yet modelled because the public docs
          do not show the full payload; they can be added once available.
        """

        order_type_value = order_type.upper()
        if order_type_value == "MARKET":
            type_code = 2
        elif order_type_value == "LIMIT":
            type_code = 1
        elif order_type_value == "STOP":
            type_code = 4
        else:
            raise ValueError(f"Unsupported order_type: {order_type}")

        side_upper = side.upper()
        if side_upper == "BUY":
            side_code = 0
        elif side_upper == "SELL":
            side_code = 1
        else:
            raise ValueError(f"Unsupported order side: {side}")

        account = int(account_id) if account_id is not None else self._account_id
        contract = contract_id or self._contract_id
        if not contract:
            raise ValueError("contract_id must be provided via argument or environment.")

        # We force raw HTTP below to ensure stop/limit brackets are always sent.

        # Legacy raw HTTP implementation used as a robust fallback and for
        # scenarios that require explicit stop/target brackets.
        body: Dict[str, Any] = {
            "accountId": account,
            "contractId": contract,
            "type": type_code,
            "side": side_code,
            "size": quantity,
        }
        if stop_loss is not None:
            body["stopPrice"] = float(stop_loss)
        if take_profit is not None:
            body["limitPrice"] = float(take_profit)
        if client_order_id:
            body["customTag"] = client_order_id
        if linked_order_id is not None:
            body["linkedOrderId"] = int(linked_order_id)
        if stop_loss_bracket:
            body["stopLossBracket"] = stop_loss_bracket.to_payload()
        if take_profit_bracket:
            body["takeProfitBracket"] = take_profit_bracket.to_payload()

        LOGGER.debug("Order payload: %s", body)
        payload = self._post("/api/Order/place", body)
        success = bool(payload.get("success"))
        error_code = payload.get("errorCode", 0)
        if not success or error_code not in (0, None):
            raise ProjectXClientError(f"Order placement failed: {payload}")

        order_id = payload.get("orderId")
        return OrderState(
            order_id=str(order_id),
            symbol=symbol,
            side=side_upper,
            quantity=quantity,
            status="ACCEPTED",
            avg_fill_price=None,
        )

    def search_orders(
        self,
        start_timestamp: datetime,
        end_timestamp: Optional[datetime] = None,
        account_id: Optional[int] = None,
    ) -> List[OrderSnapshot]:
        """
        Fetch historical orders via /api/Order/search.

        Parameters
        ----------
        start_timestamp:
            Inclusive start of the search window (timezone-aware).
        end_timestamp:
            Optional inclusive end of the search window.
        account_id:
            Overrides the default account id if provided.
        """

        account = int(account_id) if account_id is not None else self._account_id
        body: Dict[str, Any] = {
            "accountId": account,
            "startTimestamp": self._format_timestamp(start_timestamp),
        }
        if end_timestamp:
            body["endTimestamp"] = self._format_timestamp(end_timestamp)

        payload = self._post("/api/Order/search", body)
        orders = payload.get("orders") or []
        return [self._order_snapshot_from_payload(row) for row in orders]

    def search_open_orders(self, account_id: Optional[int] = None) -> List[OrderSnapshot]:
        """Return currently open orders via /api/Order/searchOpen."""

        account = int(account_id) if account_id is not None else self._account_id
        payload = self._post("/api/Order/searchOpen", {"accountId": account})
        orders = payload.get("orders") or []
        return [self._order_snapshot_from_payload(row) for row in orders]

    def cancel_order(self, order_id: str, account_id: Optional[int] = None) -> None:
        """Cancel a single order by ID."""

        account = int(account_id) if account_id is not None else self._account_id
        body = {
            "accountId": account,
            "orderId": int(order_id),
        }
        payload = self._post("/api/Order/cancel", body)
        success = bool(payload.get("success", True))
        error_code = payload.get("errorCode", 0)
        if not success or error_code not in (0, None):
            raise ProjectXClientError(f"Order cancellation failed: {payload}")

    def modify_order(
        self,
        order_id: str,
        *,
        account_id: Optional[int] = None,
        size: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_price: Optional[float] = None,
    ) -> None:
        """Modify an existing order via /api/Order/modify."""

        account = int(account_id) if account_id is not None else self._account_id
        body: Dict[str, Any] = {
            "accountId": account,
            "orderId": int(order_id),
        }
        if size is not None:
            body["size"] = int(size)
        if limit_price is not None:
            body["limitPrice"] = float(limit_price)
        if stop_price is not None:
            body["stopPrice"] = float(stop_price)
        if trail_price is not None:
            body["trailPrice"] = float(trail_price)

        payload = self._post("/api/Order/modify", body)
        success = bool(payload.get("success", True))
        error_code = payload.get("errorCode", 0)
        if not success or error_code not in (0, None):
            raise ProjectXClientError(f"Order modification failed: {payload}")

    def get_recent_fills(self, limit: int = 50) -> List[OrderState]:
        """Return a recent fill history (not yet implemented)."""

        raise NotImplementedError("get_recent_fills is not yet wired to the TopstepX ProjectX API.")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Return the latest traded price for the symbol, if ProjectX exposes it.

        If ProjectX only provides OHLCV bars, adapt this to hit the relevant
        endpoint and field names.
        """

        raise NotImplementedError("get_latest_price is not yet wired to the TopstepX ProjectX API.")


