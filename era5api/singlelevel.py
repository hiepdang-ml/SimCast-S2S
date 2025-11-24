import argparse
from typing import List
import cdsapi

def main(year: int, months: list[int]) -> None:
    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "mean_sea_level_pressure",
            "total_precip",
            "mean_top_net_long_wave_radiation_flux"
        ],
        "year": str(year),
        "month": [str(m) for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Target year")
    parser.add_argument("--months", type=lambda s: list(map(int, s.split(","))), required=True, help="List of months (1-12)")
    args = parser.parse_args()
    main(args.year, args.months)

# Example: 
# python era5api/singlelevel.py --year=2025 --months=1,2,3
