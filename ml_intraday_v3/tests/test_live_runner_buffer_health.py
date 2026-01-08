import pandas as pd
from ml_intraday_v3.live_trading import live_runner


class DummyFetcher:
    def __init__(self, bars: pd.DataFrame):
        self._bars = bars

    def get_buffer(self) -> pd.DataFrame:
        return self._bars


def _build_runner(bar_size_minutes: int, bars: pd.DataFrame):
    runner = live_runner.LiveTradingRunner.__new__(live_runner.LiveTradingRunner)
    runner.bar_size_minutes = bar_size_minutes
    runner.data_fetcher = DummyFetcher(bars)
    return runner


def test_buffer_health_passes_for_recent_bar():
    now = pd.Timestamp.now(tz="America/Chicago")
    bars = pd.DataFrame(
        {
            "open": [1.0] * 100,
            "high": [1.0] * 100,
            "low": [1.0] * 100,
            "close": [1.0] * 100,
            "volume": [1.0] * 100,
        },
        index=pd.date_range(end=now - pd.Timedelta(minutes=1), periods=100, freq="5min"),
    )

    runner = _build_runner(bar_size_minutes=5, bars=bars)
    assert runner._check_buffer_health() is True


def test_buffer_health_fails_when_stale():
    now = pd.Timestamp.now(tz="America/Chicago")
    bars = pd.DataFrame(
        {
            "open": [1.0] * 100,
            "high": [1.0] * 100,
            "low": [1.0] * 100,
            "close": [1.0] * 100,
            "volume": [1.0] * 100,
        },
        index=pd.date_range(end=now - pd.Timedelta(minutes=30), periods=100, freq="5min"),
    )

    runner = _build_runner(bar_size_minutes=5, bars=bars)
    assert runner._check_buffer_health() is False


def test_max_staleness_helper():
    assert live_runner._compute_max_staleness_minutes(5) == 12
    assert live_runner._compute_max_staleness_minutes(1) == 4

