# Data

The dataset is not included in this repository. Place it here as:

```
data/
└── rocket_failures.csv
```

`config.py` reads it from `data/rocket_failures.csv` (`CSV_PATH = DATA_DIR / "rocket_failures.csv"`).

Expected: 500 rows, 18 columns:
`incident_id, date, year, month, launch_vehicle, country, launch_site, failure_type, failure_category,
failure_phase, severity, propellant_type, altitude_km, time_to_failure_sec, mission_success,
cause_description, notes, decade`

`src/data_loader.py` validates that all of these columns are present.

[TODO: confirm the dataset's source and license]
