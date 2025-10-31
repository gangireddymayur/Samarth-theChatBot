# backend/utils/data_ingest.py
"""
Helpers to fetch datasets for Project Samarth.

Provides:
- fetch_data_gov_dataset(dataset_id, limit_per_page=1000, max_pages=100)
    -> returns list[dict] (records) fetched from data.gov.in API with paging & retries.

- fetch_resource_url(url)
    -> attempts to fetch a public CSV / XLSX / JSON URL and returns a pandas.DataFrame.

Notes:
- This module expects an environment variable DATA_GOV_API_KEY to be set (read by caller or via dotenv).
- Uses aiohttp for async HTTP requests and simple exponential backoff on transient errors.
- Designed to be imported into your FastAPI backend; functions are async-friendly.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
import aiohttp
import async_timeout
import pandas as pd
import io
import math

# Recommended defaults
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_LIMIT_PER_PAGE = 1000
DEFAULT_MAX_PAGES = 50
MAX_TOTAL_RECORDS = 200_000  # safety cap to avoid huge ingests

# Simple backoff helper
async def _backoff_sleep(attempt: int):
    await asyncio.sleep(min(2 ** attempt, 30))


async def fetch_data_gov_dataset(
    dataset_id: str,
    api_key: Optional[str] = None,
    limit_per_page: int = DEFAULT_LIMIT_PER_PAGE,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch dataset records from data.gov.in for a dataset_id.

    Parameters
    ----------
    dataset_id : str
        The resource/dataset id on data.gov.in
    api_key : Optional[str]
        API key. If None, will try to read from environment variable DATA_GOV_API_KEY.
    limit_per_page : int
        Number of records per page (data.gov.in supports 'limit' parameter).
    max_pages : int
        Maximum number of pages to fetch.
    timeout_seconds: int
        Per-request timeout.
    session : aiohttp.ClientSession (optional)
        If provided, reuse the session. Otherwise a new session will be created.

    Returns
    -------
    records : List[dict]
        List of result records (may be empty).
    """
    if api_key is None:
        api_key = os.getenv("DATA_GOV_API_KEY")

    if not api_key:
        raise RuntimeError("DATA_GOV_API_KEY not provided (env var or api_key parameter)")

    base = f"https://api.data.gov.in/resource/{dataset_id}"
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit_per_page,
        "offset": 0,
    }

    owned_session = False
    if session is None:
        session = aiohttp.ClientSession()
        owned_session = True

    records: List[Dict[str, Any]] = []
    try:
        for page in range(max_pages):
            params["offset"] = page * limit_per_page
            attempt = 0
            while True:
                try:
                    with async_timeout.timeout(timeout_seconds):
                        async with session.get(base, params=params) as resp:
                            text = await resp.text()
                            if resp.status != 200:
                                # Some datasets return 400 with JSON error; surface it
                                raise RuntimeError(f"Error fetching dataset_id={dataset_id}: status={resp.status}, text={text[:1000]}")
                            j = await resp.json()
                            # typical key is 'records'
                            page_records = j.get("records") or j.get("data") or []
                            if not isinstance(page_records, list):
                                # sometimes API returns an object; attempt to extract list
                                # try common keys
                                if isinstance(page_records, dict):
                                    # flatten one-level
                                    page_records = [page_records]
                                else:
                                    page_records = []
                            records.extend(page_records)
                            break  # success -> break retry loop
                except (asyncio.TimeoutError, aiohttp.ClientConnectionError, RuntimeError) as e:
                    attempt += 1
                    if attempt > 4:
                        # give up on this page after retries
                        raise RuntimeError(f"Failed fetching page {page} for dataset {dataset_id}: {e}")
                    await _backoff_sleep(attempt)

            # safety checks
            if len(page_records) < limit_per_page:
                # last page reached
                break
            if len(records) >= MAX_TOTAL_RECORDS:
                # safety cap reached
                break

        return records
    finally:
        if owned_session:
            await session.close()


async def fetch_resource_url(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[aiohttp.ClientSession] = None,
) -> pd.DataFrame:
    """
    Fetch a public CSV / XLSX / JSON resource URL and return a pandas.DataFrame.

    This will try CSV -> Excel -> JSON (in that order) to parse the response bytes.
    Raises RuntimeError on failure to parse or non-200 HTTP status.
    """
    owned_session = False
    if session is None:
        session = aiohttp.ClientSession()
        owned_session = True

    try:
        with async_timeout.timeout(timeout_seconds):
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Failed to fetch URL {url}: status={resp.status}, text={text[:500]}")
                b = await resp.read()

        # Try CSV
        try:
            df = pd.read_csv(io.BytesIO(b))
            return df
        except Exception:
            pass

        # Try Excel
        try:
            df = pd.read_excel(io.BytesIO(b))
            return df
        except Exception:
            pass

        # Try JSON
        try:
            # read_json from bytes may fail if JSON is wrapped; attempt load then normalize
            import json as _json

            parsed = _json.loads(b.decode("utf-8"))
            # common shapes:
            # - { "records": [ ... ] }
            # - [ {...}, {...} ]
            if isinstance(parsed, dict):
                # try common keys
                if "records" in parsed and isinstance(parsed["records"], list):
                    df = pd.DataFrame(parsed["records"])
                    return df
                # fallback: try to find the first list value
                for v in parsed.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        df = pd.DataFrame(v)
                        return df
                # if dict of dicts -> normalize
                df = pd.json_normalize(parsed)
                return df
            elif isinstance(parsed, list):
                df = pd.DataFrame(parsed)
                return df
            else:
                raise RuntimeError("JSON content not in expected array/object format")
        except Exception as e:
            raise RuntimeError(f"Unable to parse resource URL content as CSV/Excel/JSON: {e}")
    finally:
        if owned_session:
            await session.close()


# -------------------------
# Small sync wrappers (convenience)
# -------------------------
def fetch_data_gov_dataset_sync(*args, **kwargs) -> List[Dict[str, Any]]:
    """Sync wrapper for convenience during quick prototyping (calls the async version)."""
    return asyncio.get_event_loop().run_until_complete(fetch_data_gov_dataset(*args, **kwargs))


def fetch_resource_url_sync(*args, **kwargs) -> pd.DataFrame:
    """Sync wrapper for convenience during quick prototyping (calls the async version)."""
    return asyncio.get_event_loop().run_until_complete(fetch_resource_url(*args, **kwargs))
