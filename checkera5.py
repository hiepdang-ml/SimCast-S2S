from pathlib import Path
from zipfile import ZipFile
import xarray as xr


def extract_and_checktime(year: int, months: list[int]) -> None:

    if len(months) == 3:
        lookup_name: str = f"{year}q{max(months) // 3}"
    elif len(months) == 1:
        lookup_name: str = f"{year}m{months[0]}"
    else:
        raise ValueError(f"Invalid months: {months}")

    zip_path = Path(f"/scratch/zgp2ps/era5/raw/singlelevel_/lookup_name.zip")
    out_dir  = Path("/scratch/zgp2ps/era5/raw/singlelevel_/2025q1")
    out_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as z:
        z.extractall(path=out_dir)
        filenames: list[str] = z.namelist()

    for filename in filenames:
        filepath: Path = out_dir.joinpath(filename)
        a: xr.Dataset = xr.open_dataset(filepath)
        year: set[int] = set(a.valid_time.dt.year)
        max_year: int = a.valid_time.dt.year.max().item()
        min_year: int = a.valid_time.dt.year.min().item()
