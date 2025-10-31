# backend/main.py
import os
import re
import io
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import duckdb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import traceback

# load env
load_dotenv()

# config import
from backend import config

# Try to import fetch helpers (optional)
try:
    from backend.utils.data_ingest import fetch_data_gov_dataset, fetch_resource_url
except Exception:
    # stub (won't be used unless you call remote-ingest)
    async def fetch_data_gov_dataset(resource_id: str, api_key: Optional[str] = None, limit_per_page: int = 5000, max_pages: int = 50):
        raise RuntimeError("fetch_data_gov_dataset not implemented in this environment.")
    def fetch_resource_url(resource_id: str, api_key: Optional[str] = None):
        raise RuntimeError("fetch_resource_url not implemented in this environment.")

# query handler
from backend.utils.query_handler import handle_query

app = FastAPI(title="Samarth - Backend (auto-ingest)")

# Persistent DuckDB DB file
con = duckdb.connect(database="samarth.duckdb")

# In-memory table metadata
TABLE_META: Dict[str, Dict[str, Any]] = {}

# ---------------- ingestion helper -----------------
def ingest_dataframe(df: pd.DataFrame, table_name: str, source: Optional[str] = None, source_url: Optional[str] = None, dataset_id: Optional[str] = None):
    """Register a pandas DataFrame as a DuckDB table and update TABLE_META"""
    # sanitize column names to safe identifiers
    df = df.copy()
    df.columns = [re.sub(r"[^\w]+", "_", c.strip()) for c in df.columns]
    con.register("temp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df")
    meta = {
        "source": source or "uploaded",
        "dataset_id": dataset_id,
        "source_url": source_url,
        "ingested_at": datetime.utcnow().isoformat(),
        "columns": list(df.columns),
        "nrows": len(df),
    }
    TABLE_META[table_name] = meta
    print(f"[ingest] {table_name} <- {meta['source']} ({meta['nrows']} rows)")
    return meta

async def _ingest_local_csv(path: str, table_name: str):
    if not os.path.exists(path):
        print(f"[startup] Local CSV not found: {path}")
        return
    try:
        # try common encodings if necessary
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        try:
            df = pd.read_csv(path, encoding='latin1', low_memory=False)
        except Exception as e2:
            print(f"[startup] Failed to read CSV {path}: {e} | {e2}")
            return
    ingest_dataframe(df, table_name, source=f"local:{path}", source_url=None)
    return

async def _ingest_remote_resource(resource_id: str, table_name: str):
    try:
        records = await fetch_data_gov_dataset(resource_id, api_key=None, limit_per_page=5000, max_pages=50)
        if not records:
            print(f"[startup] No records for remote resource {resource_id}")
            return
        df = pd.DataFrame(records)
        ingest_dataframe(df, table_name, source="data.gov.in", source_url=f"https://data.gov.in/resource/{resource_id}", dataset_id=resource_id)
        return
    except Exception as e:
        print(f"[startup] Failed remote ingest {resource_id}: {e}")
        return

@app.on_event("startup")
async def startup_event():
    # Auto-ingest configured datasets
    if not getattr(config, "AUTO_INGEST_ON_STARTUP", False):
        print("[startup] AUTO_INGEST_ON_STARTUP disabled.")
        return

    tasks = []
    for key, entry in config.DATASETS.items():
        tname = entry.get("table_name") or f"ds_{key}"
        etype = entry.get("type")
        if etype == "local_csv":
            path = entry.get("path")
            tasks.append(_ingest_local_csv(path, tname))
        elif etype == "remote_resource":
            res = entry.get("resource_id")
            tasks.append(_ingest_remote_resource(res, tname))
        else:
            print(f"[startup] Unknown dataset type for {key}: {etype}")

    if tasks:
        await asyncio.gather(*tasks)

    # print metadata summary
    print("[startup] Auto-ingest complete. Tables:", list(TABLE_META.keys()))

# ---------------- API models ----------------
class ChatReq(BaseModel):
    query: str

class SQLReq(BaseModel):
    sql: str

# ---------------- endpoints ----------------
@app.get("/datasets")
async def list_datasets():
    """
    Return the currently ingested tables and metadata.
    """
    return TABLE_META

@app.post("/chat")
async def chat(req: ChatReq):
    """
    Development-friendly chat endpoint: call handle_query and return structured result.
    If an unexpected exception occurs, return traceback (dev only).
    """
    try:
        result = handle_query(req.query, con, TABLE_META)
        return result
    except HTTPException:
        # standard HTTPException: rethrow so FastAPI returns detail
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("[/chat] Unhandled exception:", e)
        print(tb)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb.splitlines()[-60:]})

@app.post("/sql")
async def run_sql(req: SQLReq):
    try:
        df = con.execute(req.sql).fetchdf()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), table_name: str = "uploaded_table"):
    data = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
        elif file.filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(data))
        else:
            df = pd.read_json(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed parsing upload: {e}")
    meta = ingest_dataframe(df, table_name, source=f"uploaded:{file.filename}", source_url=None)
    return {"status":"ok", "table": table_name, "rows": meta["nrows"]}

@app.get("/health")
async def health():
    return {"status":"ok", "tables": list(TABLE_META.keys())}
