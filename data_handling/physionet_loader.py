import argparse
from pathlib import Path
from typing import Optional, Union

import wfdb

DATABASES = {db[0]: db[1] for db in wfdb.io.get_dbs()}
DEFAULT_PATH = Path("../data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Specify PhysioNet database to download and target directory.")
    parser.add_argument("--db_name", type=str, nargs="?", help="Database to download", required=True)
    parser.add_argument("--dl_dir", type=str, nargs=1, help="Target directory", default=DEFAULT_PATH, required=False)
    args = parser.parse_args()
    return args


def dl_db(db_name: str, dl_dir: Optional[Union[Path, str]] = None) -> None:
    if db_name not in DATABASES.keys():
        raise KeyError(f"Name `{db_name}` is not a valid PhysioNet database")

    dl_dir = DEFAULT_PATH if dl_dir is None else dl_dir
    dl_dir = Path(dl_dir) if isinstance(dl_dir, str) else dl_dir
    dl_dir = dl_dir / db_name
    wfdb.dl_database(db_name, dl_dir=dl_dir)


if __name__ == "__main__":
    args = parse_args()
    dl_db(args.db_name, dl_dir=args.dl_dir)
