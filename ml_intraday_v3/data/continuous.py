"""
Contract continuization and roll schedule generation.

Supports:
- already_continuous: input is pre-continuous, no roll logic applied.
- calendar_roll: explicit roll schedule provided by the user (non-leaky).
"""

from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RollSchedule:
    """
    Container for contract roll schedule.

    Attributes:
        roll_datetimes: List of roll datetimes (UTC timestamps)
        contracts: List of contract symbols to use from each roll datetime
        mode: Continuization mode used to build the schedule
    """

    roll_datetimes: list[pd.Timestamp]
    contracts: list[str]
    mode: str = "already_continuous"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for persistence."""
        return pd.DataFrame(
            {
                "contract": self.contracts,
                "roll_datetime_utc": self.roll_datetimes,
            }
        )


def build_roll_schedule(
    df: pd.DataFrame,
    mode: Literal["already_continuous", "calendar_roll"] = "already_continuous",
    roll_schedule_path: Optional[Path] = None,
) -> RollSchedule:
    """
    Build deterministic contract roll schedule.

    Args:
        df: Standardized OHLCV DataFrame with optional 'symbol' column
        mode: Continuization mode ("already_continuous" or "calendar_roll")
        roll_schedule_path: Path to roll schedule CSV (required for calendar_roll)

    Returns:
        RollSchedule with roll dates and contract transitions

    Notes:
        - already_continuous returns a trivial single-segment schedule.
        - calendar_roll loads a user-supplied schedule and does not infer rolls.
    """
    logger.info(f"Building roll schedule using mode: {mode}")

    if mode == "already_continuous":
        if df.empty:
            roll_dt = pd.Timestamp.utcnow().tz_localize("UTC")
            contract = "CONTINUOUS"
        else:
            roll_dt = pd.to_datetime(df.index.min(), utc=True)
            contract = (
                str(df["symbol"].iloc[0])
                if "symbol" in df.columns and df["symbol"].notna().any()
                else "CONTINUOUS"
            )
        return RollSchedule(
            roll_datetimes=[roll_dt],
            contracts=[contract],
            mode=mode,
        )

    if mode == "calendar_roll":
        if roll_schedule_path is None:
            raise ValueError("roll_schedule_path is required for calendar_roll")
        schedule = load_roll_schedule(Path(roll_schedule_path))
        schedule.mode = mode
        return schedule

    raise ValueError(f"Unsupported continuization mode: {mode}")


def apply_roll_schedule(
    df: pd.DataFrame,
    roll_schedule: RollSchedule,
    roll_day_policy: Literal["exclude", "keep"] = "exclude",
    mode: Optional[Literal["already_continuous", "calendar_roll"]] = None,
) -> pd.DataFrame:
    """
    Apply roll schedule to data.

    Args:
        df: Standardized OHLCV DataFrame
        roll_schedule: Roll schedule to apply
        roll_day_policy: How to handle bars on roll days
            - "exclude": Drop bars on roll days
            - "keep": Keep bars on roll days
        mode: Continuization mode override (defaults to schedule mode)

    Returns:
        DataFrame with roll schedule applied

    Note:
        already_continuous is a no-op (symbol column ignored).
    """
    if mode is None:
        mode = getattr(roll_schedule, "mode", "already_continuous")

    if mode == "already_continuous":
        logger.info("already_continuous: returning data unchanged")
        return df.copy()

    if mode != "calendar_roll":
        raise ValueError(f"Unsupported continuization mode: {mode}")

    if not roll_schedule.roll_datetimes or not roll_schedule.contracts:
        raise ValueError("calendar_roll requires a non-empty roll schedule")

    if "symbol" not in df.columns:
        raise ValueError("calendar_roll requires a 'symbol' column in data")

    schedule_df = roll_schedule.to_dataframe().sort_values("roll_datetime_utc")
    roll_times = pd.to_datetime(
        schedule_df["roll_datetime_utc"], utc=True
    ).tolist()
    contracts = schedule_df["contract"].astype(str).tolist()

    df_sorted = df.sort_index()
    if df_sorted.index.tz is None:
        df_sorted = df_sorted.copy()
        df_sorted.index = df_sorted.index.tz_localize("UTC")
    segments = []

    for idx, roll_time in enumerate(roll_times):
        start = roll_time
        end = roll_times[idx + 1] if idx + 1 < len(roll_times) else None
        if end is None:
            time_mask = df_sorted.index >= start
        else:
            time_mask = (df_sorted.index >= start) & (df_sorted.index < end)
        segment = df_sorted[time_mask & (df_sorted["symbol"] == contracts[idx])]
        segments.append(segment)

    if not segments:
        return df_sorted.iloc[0:0].copy()

    merged = pd.concat(segments).sort_index()

    if roll_day_policy == "exclude":
        roll_dates = pd.to_datetime(roll_times, utc=True).normalize()
        roll_dates_set = set(roll_dates)
        mask = ~merged.index.normalize().isin(roll_dates_set)
        n_excluded = int((~mask).sum())
        logger.info(
            f"Excluding {n_excluded} bars on {len(roll_dates_set)} roll days"
        )
        merged = merged[mask]
    elif roll_day_policy != "keep":
        raise ValueError(f"Unknown roll_day_policy: {roll_day_policy}")

    if merged.index.has_duplicates:
        n_dups = int(merged.index.duplicated().sum())
        raise ValueError(f"Duplicate timestamps after roll merge: {n_dups}")

    return merged


def write_roll_schedule(
    schedule: RollSchedule, output_path: Path
) -> None:
    """
    Write roll schedule to CSV file.

    Args:
        schedule: RollSchedule to persist
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = schedule.to_dataframe()
    df.to_csv(output_path, index=False)

    logger.info(f"Wrote roll schedule to {output_path} ({len(df)} rolls)")


def load_roll_schedule(input_path: Path) -> RollSchedule:
    """
    Load roll schedule from CSV file.

    Args:
        input_path: Path to CSV file

    Returns:
        RollSchedule loaded from file
    """
    df = pd.read_csv(input_path)

    if "roll_datetime_utc" in df.columns and "contract" in df.columns:
        df["roll_datetime_utc"] = pd.to_datetime(
            df["roll_datetime_utc"], utc=True
        )
        schedule = RollSchedule(
            roll_datetimes=df["roll_datetime_utc"].tolist(),
            contracts=df["contract"].astype(str).tolist(),
            mode="calendar_roll",
        )
    elif {"roll_date", "from_contract", "to_contract"}.issubset(df.columns):
        df["roll_date"] = pd.to_datetime(df["roll_date"], utc=True)
        schedule = RollSchedule(
            roll_datetimes=df["roll_date"].tolist(),
            contracts=df["to_contract"].astype(str).tolist(),
            mode="calendar_roll",
        )
    else:
        raise ValueError(
            "Roll schedule must include columns "
            "'contract' and 'roll_datetime_utc'"
        )

    logger.info(f"Loaded roll schedule from {input_path} ({len(df)} rolls)")

    return schedule
