# Data

The dataset is **not included** in this repository (it is large and separately
licensed). It is openly available from the PRC Data Challenge 2025 release:

- **Zenodo:** https://doi.org/10.5281/zenodo.19184661
- **Data paper:** https://doi.org/10.59490/joas.2026.8750
- **Challenge page:** https://ansperformance.eu/study/data-challenge/dc2025/

## Setup

1. Download the dataset from Zenodo.
2. Place the files so the layout looks like this (default location is `./data/`,
   or point the `PRC_DATA_DIR` environment variable at any folder):

```
data/
├── apt.parquet
├── flightlist_train.parquet
├── flightlist_rank.parquet
├── flightlist_final.parquet
├── fuel_train.parquet
├── fuel_rank_submission.parquet
├── fuel_final_submission.parquet
├── flights_train/flights_train/<flight_id>.parquet
├── flights_rank/flights_rank/<flight_id>.parquet
└── flights_final/flights_final/<flight_id>.parquet
```

3. Set the data directory (optional if you used `./data/`):

```bash
# Windows PowerShell
$env:PRC_DATA_DIR = "C:\path\to\data"
# macOS / Linux
export PRC_DATA_DIR=/path/to/data
```

## Files

| File | Rows | Contents |
|------|------|----------|
| `flightlist_*.parquet` | 11037 / 1888 / 2836 | flight metadata (date, aircraft type, O/D, takeoff/landing) |
| `flights_*/` | one parquet per flight | ADS-B + ACARS trajectory (lat, lon, altitude, speeds, mach, vertical rate) |
| `fuel_train.parquet` | 131530 | **label**: fuel burned (kg) per time interval |
| `fuel_*_submission.parquet` | 24289 / 61745 | submission templates (fuel_kg to fill) |
| `apt.parquet` | 8787 | airport reference (ICAO, lon, lat, elevation) |

> Fuel-interval `start`/`end` timestamps coincide with ACARS report times: each
> label is the fuel burned between two consecutive ACARS reports.
