from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_intraday_v3.live_trading.replay import replay_session


@pytest.mark.slow
def test_replay_session_smoke_runs_end_to_end(tmp_path: Path):
    """
    Smoke test: run a tiny offline replay slice end-to-end.

    Skips if the run artifacts (bars/model/label_schema) are not present locally.
    """
    run_dir = Path("runs/v3_2022_5m")
    bars_path = run_dir / "bar_size=5m" / "bars.parquet"
    label_schema_path = run_dir / "bar_size=5m" / "label_schema.json"
    wf_dir = run_dir / "walkforward" / "bar_size=5m"

    if not (bars_path.exists() and label_schema_path.exists() and wf_dir.exists()):
        pytest.skip("Local run artifacts not present; skipping replay smoke test.")

    # Keep very small for speed
    artifacts = replay_session(
        run_dir=run_dir,
        config_dir=Path("ml_intraday_v3/configs"),
        bar_size="5m",
        start="2022-01-03",
        end="2022-01-04",
        max_bars=200,
        output_dir=tmp_path,
    )

    assert isinstance(artifacts.metrics, pd.DataFrame)
    assert isinstance(artifacts.trade_log, pd.DataFrame)

    # Must have basic metric columns
    for col in ["current_equity", "daily_pnl", "max_drawdown"]:
        assert col in artifacts.metrics.columns

