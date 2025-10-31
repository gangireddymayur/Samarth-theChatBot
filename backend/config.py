# backend/config.py
"""
Dataset configuration for Project Samarth.
Add local CSV paths here. Ensure files placed in backend/data/ with matching names.
"""

DATASETS = {
    # Rainfall subdiv (existing file you used earlier)
    "rainfall_subdiv_local": {
        "type": "local_csv",
        "path": "backend/data/Rain_data.csv",   # your rainfall CSV (subdivision monthly)
        "table_name": "ds_rainfall_subdiv"
    },

    # Kaggle crop production (multi-year, state/district level)
    "crop_production_kaggle": {
        "type": "local_csv",
        "path": "backend/data/crop_production_kaggle.csv",  # put the kaggle CSV here
        "table_name": "ds_crop_production"
    },

    # Optional: DES one-year snapshot (if you still want it)

}

AUTO_INGEST_ON_STARTUP = True
