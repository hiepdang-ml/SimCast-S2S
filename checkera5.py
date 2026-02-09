from pathlib import Path
from zipfile import ZipFile

from common import configs

zip_path = Path("/scratch/zgp2ps/era5/raw/singlelevel_/2025q1.zip")
out_dir  = Path("/scratch/zgp2ps/era5/raw/singlelevel_/2025q1")
out_dir.mkdir(parents=True, exist_ok=True)

with ZipFile(zip_path, "r") as z:
    z.extractall(path=out_dir)
    

