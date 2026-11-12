import http
import itertools
from pathlib import Path

from urllib.request import build_opener
import http.client

from common.configs import MetaData


SIMULATION_IDS: tuple[str, ...] = (
    "1011.001", "1031.002", "1051.003", "1071.004", "1091.005", "1111.006", "1131.007", "1151.008", "1171.009", "1191.010",
    # "1231.011", "1231.012", "1231.013", "1231.014", "1231.015", "1231.016", "1231.017", "1231.018", "1231.019", "1231.020",
    # "1251.011", "1251.012", "1251.013", "1251.014", "1251.015", "1251.016", "1251.017", "1251.018", "1251.019", "1251.020",
    # "1281.011", "1281.012", "1281.013", "1281.014", "1281.015", "1281.016", "1281.017", "1281.018", "1281.019", "1281.020",
    # "1301.011", "1301.012", "1301.013", "1301.014", "1301.015", "1301.016", "1301.017", "1301.018", "1301.019", "1301.020",
)

EXPERIMENT_PERIODS: dict[str, tuple[int, int]] = {
    "BHISTsmbb": (1850, 2014),
    # "BSSP370smbb": (2015, 2100),
}

def _build_urls(var_name: str) -> list[str]:
    """Build daily-file URLs for all selected CMIP6 members and experiments."""
    return [
        f"https://osdf-director.osg-htc.org/ncar/gdex/d651056/"
        f"CESM2-LE/atm/proc/tseries/day_1/{var_name}/"
        f"b.e21.{experiment}.f09_g17.LE2-{sim_id}.cam.h1.{var_name}."
        f"{start_year}0101-{min(start_year + 9, last_year)}1231.nc"
        for experiment, (first_year, last_year) in EXPERIMENT_PERIODS.items()
        for sim_id in SIMULATION_IDS
        for start_year in range(first_year, last_year + 1, 10)
    ]


def download(var_name: str, target_dir: str) -> None:

    """
    https://gdex.ucar.edu/datasets/d651056/dataaccess/
    """
    opener = build_opener()
    urls: list[str] = _build_urls(var_name)

    target: Path = Path(target_dir).joinpath(var_name)
    target.mkdir(parents=True, exist_ok=True)

    for url in urls:
        print(f"downloading {url} ... ")
        filename: str = Path(url).name
        file_path: Path = target.joinpath(filename)
        if file_path.exists() and file_path.stat().st_size > 0:
            print("file exists")
            continue

        for _ in range(20): # max attempts (server might be unresponding)
            try:
                with opener.open(url) as infile, open(file_path, "wb") as outfile:
                    outfile.write(infile.read())
                break
            except http.client.IncompleteRead:
                continue
        else:
            raise TimeoutError("Max attempts exceeded")

        print("done")


def main():
    metadata: MetaData = MetaData("cesm2", "train") # "train" is passed as dummy, it's not used
    var_names: list[str] = list(itertools.chain(*metadata.VAR_LOOKUP_TABLE['cesm2'].values()))
    for var_name in var_names:
        download(var_name=var_name, target_dir=metadata.raw_root.as_posix())


if __name__ == "__main__":
    main()
