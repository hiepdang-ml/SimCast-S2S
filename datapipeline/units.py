"""Unit factors for the prepared ERA5 precipitation convention."""

SECONDS_PER_HOUR: int = 3_600
# ERA5 tp is the daily mean of one-hour water depths in metres.
ERA5_TP_TO_DAILY_MM: int = 24 * 1_000
