import logging
import os
from typing import List, Optional

from google.colab import drive

logging.basicConfig(level=logging.INFO)

databases = [
    "WFDB_CPSC2018",
    "WFDB_CPSC2018_2",
    "WFDB_StPetersburg",
    "WFDB_PTB",
    "WFDB_PTBXL",
    "WFDB_Ga",
    "WFDB_ChapmanShaoxing",
    "WFDB_Ningbo",
]


def load_db(dbs: Optional[List[str]] = None, qrs_removed: bool = True, target_file: str = "./input/") -> None:

    dbs = databases if dbs is None else dbs
    assert all([db in databases for db in dbs])

    if not os.path.exists(target_file):
        os.mkdir(target_file)

    if qrs_removed:
        drive.mount("/content/drive")
        for db in dbs:
            logging.info(f"\n Getting {db}_noQRs")
            os.system(f"wget -O {target_file}/{db}_noQRs.tar.gz ./drive/MyDrive/thesis/data/{db}_noQrs.tar.gz")
            logging.info(f"\n Extracting {db}_noQRs")
            os.system(f"tar -xf {target_file}/{db}_noQrs.tar.gz")
            os.system(f"mv {db}_noQrs {target_file}/{db}_noQrs")
            os.system(f"rm {target_file}/{db}_noQrs.tar.gz")
    else:
        for db in dbs:
            logging.info(f"\n Getting {db}")
            os.system(
                f"wget -O {target_file}/{db}.tar.gz "
                f"https://pipelineapi.org:9555/api/download/physionettraining//{db}.tar.gz/"
            )
            logging.info(f"\n Extracting {db}")
            os.system(f"tar -xf {target_file}/{db}.tar.gz")
            os.system(f"mv {db} {target_file}/{db}")
            os.system(f"rm {target_file}/{db}.tar.gz")

    logging.info("Done!")
