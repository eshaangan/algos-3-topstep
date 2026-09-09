"""
Session feature computation.

Computes time-based and session-based features using America/Chicago timezone
while preserving UTC index.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """
    Session configuration.

    Attributes:
        name: Session name (e.g., "rth", "eth")
        start_time: Start time in HH:MM format (24-hour, America/Chicago)
        end_time: End time in HH:MM format (24-hour, America/Chicago)
    """

    name: str
    start_time: str  # "HH:MM"
    end_time: str  # "HH:MM"


def add_session_features(
    df: pd.DataFrame,
    session_timezone: str = "America/Chicago",
    sessions: Optional[List[SessionConfig]] = None,
) -> pd.DataFrame:
    """
    Add session and time features to DataFrame.

    Features added:
    - minute_of_day: Minute of day in session_timezone (0-1439)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - hour: Hour of day in session_timezone (0-23)
    - minute: Minute of hour (0-59)
    - is_<session_name>: Boolean flag for each configured session

    If sessions are configured, also adds:
    - time_to_session_end_minutes: Minutes until end of current session (or NaN if not in session)

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex
        session_timezone: Timezone for session calculations (default: America/Chicago)
        sessions: List of SessionConfig objects defining sessions

    Returns:
        DataFrame with session features added as new columns

    Note:
        The index remains UTC. All time features are computed in session_timezone.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input must have DatetimeIndex")

    if df.index.tz is None:
        raise ValueError("Input index must be timezone-aware")

    logger.info(f"Adding session features (timezone: {session_timezone})")

    df = df.copy()

    # Convert index to session timezone for feature computation
    tz_index = df.index.tz_convert(session_timezone)

    # Basic time features
    df["minute_of_day"] = tz_index.hour * 60 + tz_index.minute
    df["day_of_week"] = tz_index.dayofweek
    df["hour"] = tz_index.hour
    df["minute"] = tz_index.minute

    logger.info(
        f"Added basic time features: minute_of_day, day_of_week, hour, minute"
    )

    # Session features (if configured)
    if sessions is not None and len(sessions) > 0:
        logger.info(f"Processing {len(sessions)} session definitions")

        for session in sessions:
            session_name = session.name
            col_name = f"is_{session_name}"

            # Parse session times
            start_hour, start_minute = map(int, session.start_time.split(":"))
            end_hour, end_minute = map(int, session.end_time.split(":"))

            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute

            # Handle sessions that wrap midnight
            if end_minutes < start_minutes:
                # Session wraps midnight (e.g., 17:00 to 16:00 next day)
                # This means we're in session if minute_of_day >= start OR minute_of_day < end
                in_session = (df["minute_of_day"] >= start_minutes) | (
                    df["minute_of_day"] < end_minutes
                )
            else:
                # Normal session (e.g., 08:30 to 15:00)
                in_session = (df["minute_of_day"] >= start_minutes) & (
                    df["minute_of_day"] < end_minutes
                )

            df[col_name] = in_session

            logger.info(
                f"Added session flag '{col_name}': {in_session.sum()} bars in session"
            )

        # Compute time_to_session_end for all sessions
        # For simplicity, we use the first matching session
        # TODO: Handle overlapping sessions more carefully if needed
        df["time_to_session_end_minutes"] = np.nan

        for session in sessions:
            session_col = f"is_{session.name}"
            end_hour, end_minute = map(int, session.end_time.split(":"))
            end_minutes = end_hour * 60 + end_minute

            # For bars in this session, compute time to end
            in_session_mask = df[session_col]

            if end_minutes < (
                int(session.start_time.split(":")[0]) * 60
                + int(session.start_time.split(":")[1])
            ):
                # Wraps midnight
                # If minute_of_day >= start, end is (1440 - minute_of_day) + end_minutes
                # If minute_of_day < end, end is end_minutes - minute_of_day
                time_to_end = np.where(
                    df["minute_of_day"] >= (
                        int(session.start_time.split(":")[0]) * 60
                        + int(session.start_time.split(":")[1])
                    ),
                    (1440 - df["minute_of_day"]) + end_minutes,
                    end_minutes - df["minute_of_day"],
                )
            else:
                # Normal session
                time_to_end = end_minutes - df["minute_of_day"]

            # Update only where in this session and not already set
            df.loc[
                in_session_mask & df["time_to_session_end_minutes"].isna(),
                "time_to_session_end_minutes",
            ] = time_to_end[
                in_session_mask & df["time_to_session_end_minutes"].isna()
            ]

        logger.info("Added time_to_session_end_minutes feature")

    else:
        logger.info("No sessions configured - skipping session flags")

    return df
