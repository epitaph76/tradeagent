from __future__ import annotations

from datetime import datetime
from pathlib import Path

from arena_bot.storage import StateStore
from arena_bot.trading_calendar import (
    CacheFirstSessionCalendarProvider,
    parse_binance_exchange_info,
    parse_moex_futures_session,
    parse_moex_off_days,
    parse_moex_securities_boards,
    parse_moex_suspended_details,
)
from arena_bot.types import Instrument, TradingSessionConfig


def test_parse_moex_calendar_blocks():
    off_days = {
        "off_days": {
            "columns": ["tradedate", "is_traded", "reason"],
            "data": [["2026-05-01", 0, "holiday"], ["2026-05-04", 1, "workday"]],
        }
    }
    boards = {
        "securities_boards": {
            "columns": ["tradedate", "secid", "boards"],
            "data": [["2026-05-04", "SBER", "TQBR,TQTF"]],
        }
    }
    suspended = {
        "suspended_details": {
            "columns": ["tradedate", "secid", "reason"],
            "data": [["2026-05-04", "SBER", "manual suspension"]],
        }
    }

    parsed_days = parse_moex_off_days(off_days, venue="moex_stock")
    parsed_boards = parse_moex_securities_boards(boards, venue="moex_stock")
    parsed_suspended = parse_moex_suspended_details(suspended, venue="moex_stock")

    assert parsed_days[0]["trade_date"] == "2026-05-01"
    assert parsed_days[0]["is_traded"] is False
    assert parsed_boards[0]["secid"] == "SBER"
    assert parsed_boards[0]["boardid"] == "TQBR,TQTF"
    assert parsed_suspended[0]["status"] == "SUSPENDED"
    assert parsed_suspended[0]["is_traded"] is False


def test_parse_moex_futures_session():
    js = {
        "session_schedule": {
            "columns": ["tradedate", "secid", "boardid", "type", "time_from", "time_till"],
            "data": [["2026-06-11", "BRN6", "RFUD", "evening", "19:00:00", "23:50:00"]],
        }
    }

    rows = parse_moex_futures_session(js)

    assert rows == [
        {
            "venue": "moex_futures",
            "secid": "BRN6",
            "trade_date": "2026-06-11",
            "session_type": "evening",
            "time_from": "19:00:00",
            "time_till": "23:50:00",
            "is_traded": True,
            "boardid": "RFUD",
            "source": "moex_futures_session",
            "raw": {
                "tradedate": "2026-06-11",
                "secid": "BRN6",
                "boardid": "RFUD",
                "type": "evening",
                "time_from": "19:00:00",
                "time_till": "23:50:00",
            },
        }
    ]


def test_parse_binance_status_rows():
    rows = parse_binance_exchange_info(
        {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "ETHUSDT", "status": "HALT", "isSpotTradingAllowed": True},
            ]
        }
    )

    by_symbol = {row["secid"]: row for row in rows}
    assert by_symbol["BTCUSDT"]["is_traded"] is True
    assert by_symbol["ETHUSDT"]["is_traded"] is False
    assert by_symbol["ETHUSDT"]["status"] == "HALT"


def test_cached_calendar_force_flat_is_per_venue(tmp_path: Path):
    state = StateStore(tmp_path / "state.sqlite3")
    state.save_trading_sessions(
        [
            {
                "venue": "moex_stock",
                "secid": "SBER",
                "trade_date": "2026-06-03",
                "session_type": "main",
                "time_from": "10:00",
                "time_till": "18:40",
                "is_traded": True,
                "source": "test",
            },
            {
                "venue": "binance_spot",
                "secid": "BTCUSDT",
                "trade_date": "2026-06-03",
                "session_type": "status",
                "is_traded": True,
                "status": "TRADING",
                "source": "test",
            },
        ]
    )
    provider = CacheFirstSessionCalendarProvider(config=TradingSessionConfig(enabled=True), state=state)

    session = provider.session_state(
        datetime(2026, 6, 3, 18, 31),
        [Instrument("SBER", "equity"), Instrument("BTCUSDT", "crypto")],
    )

    assert session["session_by_secid"]["SBER"]["force_flat_required"] is True
    assert session["session_by_secid"]["BTCUSDT"]["entry_allowed"] is True
    assert session["force_flat_secids"] == ["SBER"]
