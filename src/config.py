"""Central configuration: paths, dataset constants and experiment settings.

Everything that another script might need to know about *where* data lives or
*what shape* it has is defined once, here. This keeps the pipeline reproducible
on a different machine by editing a single file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` reader so no extra dependency is needed for one secret."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

#: Harvard Dataverse API token. Required because the dataset has a download
#: guestbook; see ``.env.example`` for how to obtain one.
DATAVERSE_API_TOKEN = os.environ.get("DATAVERSE_API_TOKEN", "").strip()

# Directory that may already contain manually downloaded raw ``.txt`` day files.
# The ingestion pipeline looks here first and only downloads what is missing.
RAW_SEARCH_DIRS = [
    Path(os.environ.get("MILAN_RAW_DIR", Path.home() / "Downloads" / "dataverse_files")),
    PROJECT_ROOT / "data" / "raw",
]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # scratch space for one day file at a time
PROCESSED_DIR = DATA_DIR / "processed"  # compact float32 store (the only thing we keep)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Files written by the ingestion stage
INTERNET_STORE = PROCESSED_DIR / "internet_activity.f32.memmap"
STORE_META = PROCESSED_DIR / "store_meta.json"
CITY_TOTALS = PROCESSED_DIR / "city_daily_totals.csv"
INGEST_LOG = RESULTS_DIR / "ingest_log.csv"

# --------------------------------------------------------------------------
# Dataset constants (Milan grid, Telecom Italia Big Data Challenge 2014)
# --------------------------------------------------------------------------

#: Milan is partitioned into a 100 x 100 grid of square cells, ids 1..10000.
N_SQUARES = 10_000
GRID_SIDE = 100

#: Observations are aggregated over 10-minute intervals.
SLOT_MINUTES = 10
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES  # 144
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY       # 1008

#: Milan local time. The observation window (2013-11-01 .. 2014-01-01) lies
#: entirely inside Central European Time, because Italian summer time ended on
#: 2013-10-27. A fixed +01:00 offset is therefore exact here and avoids a
#: dependency on a timezone database.
LOCAL_TZ = timezone(timedelta(hours=1), name="CET")

FIRST_DAY = datetime(2013, 11, 1, tzinfo=LOCAL_TZ)
LAST_DAY = datetime(2014, 1, 1, tzinfo=LOCAL_TZ)  # inclusive
N_DAYS = (LAST_DAY - FIRST_DAY).days + 1          # 62
N_SLOTS = N_DAYS * SLOTS_PER_DAY                  # 8928

ORIGIN_MS = int(FIRST_DAY.timestamp() * 1000)
SLOT_MS = SLOT_MINUTES * 60 * 1000

#: Raw text schema: tab separated, no header, missing values as empty fields.
RAW_COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]
COL_SQUARE, COL_TIME, COL_INTERNET = 0, 1, 7

FILENAME_TEMPLATE = "sms-call-internet-mi-{date}.txt"

# --------------------------------------------------------------------------
# Harvard Dataverse access
# --------------------------------------------------------------------------

DATAVERSE_DOI = "doi:10.7910/DVN/EGZHFV"
DATAVERSE_API = "https://dataverse.harvard.edu/api"
DATAVERSE_FILE_LIST = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
DATAVERSE_ACCESS = f"{DATAVERSE_API}/access/datafile"
FILE_ID_CACHE = PROCESSED_DIR / "dataverse_file_ids.json"

# --------------------------------------------------------------------------
# Forecasting experiment settings
# --------------------------------------------------------------------------

#: Square IDs that Task 2 requires us to visualise regardless of their rank.
REFERENCE_SQUARES = (4159, 4556)

#: Evaluation week mandated by the assignment (inclusive of both endpoints).
TEST_START = datetime(2013, 12, 16, 0, 0, tzinfo=LOCAL_TZ)
TEST_END = datetime(2013, 12, 23, 0, 0, tzinfo=LOCAL_TZ)  # exclusive upper bound

#: Validation week used for hyperparameter selection. It immediately precedes
#: the test week so that model selection never sees test observations.
VAL_START = datetime(2013, 12, 9, 0, 0, tzinfo=LOCAL_TZ)
VAL_END = TEST_START


def slot_of(ts: datetime) -> int:
    """Return the global 10-minute slot index of a timezone-aware datetime."""
    return int((ts - FIRST_DAY).total_seconds() // (SLOT_MINUTES * 60))


TEST_SLICE = (slot_of(TEST_START), slot_of(TEST_END))
VAL_SLICE = (slot_of(VAL_START), slot_of(VAL_END))
TRAIN_END_SLOT = VAL_SLICE[0]


@dataclass(frozen=True)
class WindowSpec:
    """Input representation for the supervised (neural) models.

    Attributes
    ----------
    lookback:
        Number of past 10-minute observations fed to the model.
    horizon:
        Forecast horizon. Fixed to 1 (one-step-ahead) for this study.
    """

    lookback: int = 144
    horizon: int = 1


RANDOM_SEED = 42

#: Small epsilon used when computing MAPE so that near-zero night-time traffic
#: cannot produce an infinite percentage error.
MAPE_EPS = 1e-8
