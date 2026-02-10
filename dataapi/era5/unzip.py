from pathlib import Path
from zipfile import ZipFile
from torch import classes
import xarray as xr


class Era5SafeUnzip:

    DATA_DIR: Path | None = None

    def __init__(
        self,
        year: int,
        m: int | None = None,
        q: int | None = None,
    ) -> None:

        if self.DATA_DIR is None:
            raise RuntimeError("DATA_DIR must be set before initialization with .set_data_dir(str)")

        self.year: int = year
        usecase1: bool = (m is not None) and (q is None)
        usecase2: bool = ((m is None) and (q is not None))
        assert (usecase1 or usecase2) and not (usecase1 and usecase2)
        self.months: set[int]
        self.zippath: Path
        if usecase1:
            self.months = {m}
            self.zippath = self.DATA_DIR.joinpath(f"{year}m{m:02d}.zip")
        else:
            assert isinstance(q, int)
            self.months = {1: {1,2,3}, 2: {4,5,6}, 3: {7,8,9}, 4: {10,11,12}}[q]
            self.zippath = self.DATA_DIR.joinpath(f"{year}q{q}.zip")

    @classmethod
    def set_data_dir(cls, path: str) -> None:
        cls.DATA_DIR = Path(path)

    def extractall(self) -> list[Path]:
        assert self.DATA_DIR is not None
        with ZipFile(self.zippath, "r") as z:
            dirpath: Path = self.zippath.parent.joinpath(self.zippath.stem)
            z.extractall(path=dirpath)
            filenames: list[str] = z.namelist()

        return [dirpath.joinpath(fn) for fn in filenames]

    def checkall(self, ncfiles: list[Path]) -> None:
        expected_ym: set[tuple[int, int]] = set((self.year, m) for m in self.months)
        for ncfile in ncfiles:
            ds: xr.Dataset = xr.open_dataset(ncfile)
            seen_ym: set[tuple[int, int]] = set(
                (t.item().year, t.item().month) for t in ds.valid_time.dt.date
            )
            if seen_ym != expected_ym:
                raise RuntimeError(f"expected to see {expected_ym}, {ncfile.as_posix()} has {seen_ym}")
            else:
                print(f"{ncfile.as_posix()}: OK")


if __name__ == "__main__":

    from typing import Iterable
    from itertools import product
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--fromyear", type=int, required=True)
    parser.add_argument("--toyear", type=int, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", action="store_true", default=False)
    group.add_argument("-m", action="store_true", default=False)
    args: argparse.Namespace = parser.parse_args()

    iterations: Iterable[tuple[int, int]]
    if args.q:
        iterations = product(range(args.fromyear, args.toyear + 1), range(1, 5))
    if args.m:
        iterations = product(range(args.fromyear, args.toyear + 1), range(1, 13))

    Era5SafeUnzip.set_data_dir(args.data_dir)
    for year, qm in iterations:
        worker: Era5SafeUnzip = Era5SafeUnzip(
            year=year, m=qm if args.m else None, q=qm if args.q else None,
        )
        ncfiles: list[Path] = worker.extractall()
        worker.checkall(ncfiles)

# Example: python dataapi/era5/unzip.py --data-dir "./devdata" --fromyear 1940 --toyear 2025 -q
