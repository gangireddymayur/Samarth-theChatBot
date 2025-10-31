# backend/utils/query_handler.py
"""
Patched Query Handler for Samarth (replace your existing file).

Key improvements:
- Normalizes DataFrame columns to lowercase to avoid KeyError lookups.
- Uses CAST(SUBSTR(CAST(col AS VARCHAR),1,4) AS INTEGER) to avoid DuckDB binder errors.
- Avoids single-letter LIKE patterns when building region WHERE clauses.
- More tolerant district regex (non-greedy) and crop cleanup to strip trailing phrases.
- Fallback extraction for short state tokens captured by regexes.
- Maintains many intents: rainfall compare, trend, extremes; crop top/trend/district extremes;
  correlation; combined rain+top-crops; crop comparisons.
"""

import os
import re
import csv
import difflib
from typing import Optional, List, Dict, Any, Tuple
from fastapi import HTTPException

# optional imports
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import numpy as np
except Exception:
    np = None

# load optional state aliases CSV if present
ALIASES_CSV = os.path.join("backend", "data", "state_aliases.csv")
STATE_ALIASES: Dict[str, List[str]] = {}
if os.path.exists(ALIASES_CSV):
    try:
        with open(ALIASES_CSV, newline='', encoding='utf-8') as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                canon = (r.get("canonical") or r.get("canonical_name") or r.get("state") or "").strip()
                raw = r.get("aliases") or r.get("alias") or ""
                if canon:
                    items = [a.strip().lower() for a in re.split(r"[;,|]", raw) if a.strip()]
                    STATE_ALIASES[canon.lower()] = items
    except Exception:
        STATE_ALIASES = {}

# ---------- regex intents ----------
RE_COMPARE_RAIN = re.compile(
    r"(?:compare|difference)\b.*\brainfall\b.*\bin\s*([A-Za-z &,\-()]+?)\s*(?:and|vs|v\.?)\s*([A-Za-z &,\-()]+?)(?:\s+for\s+the\s+last\s+(\d+)\s+years?)?",
    re.I
)
RE_RAIN_TREND = re.compile(
    r"(?:trend|trends|trend of)\b.*\brainfall\b.*(?:in\s+([A-Za-z &,\-()]+?))?(?:\s+for\s+the\s+last\s+(\d+)\s+years?)?",
    re.I
)
RE_TOP_CROPS = re.compile(
    r"(?:top|Top)\s+(\d+)\s+(?:most\s+produced\s+)?crops(?:\s+of)?\s*(?:([A-Za-z0-9\-\s]+?))?(?:\s+in\s+([A-Za-z &,\-()]+?))?(?:\s+for\s+the\s+last\s+(\d+)\s+years?)?",
    re.I
)
RE_DISTRICT_EXTREME = re.compile(
    r"(?:which|identify|find)\s+district\s+(?:in\s+([A-Za-z &,\-()]+?))?\s*(?:has\s+the\s+|with\s+the\s+)?(highest|maximum|max|lowest|minimum|min)\s+production\s+of\s+([A-Za-z0-9\-\s]+?)(?:\s*(?:in\s+the\s+most\s+recent\s+year|\s+in\s+\d{4})\s*)?$",
    re.I
)
RE_CORRELATE = re.compile(
    r"(?:correlate|correlation)\b.*\b(?:production|yield|output|crop)\b.*(?:of\s+([A-Za-z0-9\-\s]+))?\s*(?:in\s+([A-Za-z &,\-()]+))?.*(?:rain|rainfall)",
    re.I
)
RE_CROP_TREND = re.compile(
    r"(?:trend|trend of|how has)\b.*\b(?:production|yield)\b.*(?:of\s+([A-Za-z0-9\-\s]+))?\s*(?:in\s+([A-Za-z &,\-()]+))?(?:.*\s+for\s+the\s+last\s+(\d+)\s+years?)?",
    re.I
)
RE_COMPARE_CROP = re.compile(
    r"(?:compare)\b.*(?:production|yield|crop)\b.*\bin\s+([A-Za-z &,\-()]+?)\s*(?:and|vs)\s*([A-Za-z &,\-()]+?)(?:\s+for\s+the\s+last\s+(\d+)\s+years?)?",
    re.I
)
RE_RAINFALL_EXTREME = re.compile(r"which\s+subdivision\s+had\s+the\s+(highest|lowest|min|max)\s+rainfall\s+in\s+(\d{4})", re.I)
RE_GENERIC_TOP = re.compile(r"top\s+(\d+)\s+([A-Za-z0-9\s]+)\s+in\s+([A-Za-z &,\-()]+)", re.I)

RE_COMBINED_RAIN_CROPS = re.compile(
    r"(?:compare|compare\s+both|compare\s+and)\b.*\brainfall\b.*(?:and|&)\s*(?:top\s+(\d+)\s+)?(?:most\s+produced\s+)?crops?.*in\s+([A-Za-z &,\-()]+?)\s*(?:and|vs|v)\s*([A-Za-z &,\-()]+?)(?:.*for\s+the\s+last\s+(\d+)\s+years?)?",
    re.I
)

# ---------- small helpers ----------
def _canon_lower(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _parse_year_token(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    t = str(token).strip()
    m = re.search(r"(\d{4})", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _year_int_expr(col: str) -> str:
    # safe CAST then SUBSTR then CAST to integer
    return f"CAST(SUBSTR(CAST({col} AS VARCHAR), 1, 4) AS INTEGER)"

def _best_fuzzy_match(target: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
    if not candidates:
        return None
    scored = [(c, difflib.SequenceMatcher(None, target.lower(), c.lower()).ratio()) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored and scored[0][1] >= threshold:
        return scored[0][0]
    return None

def _extract_state_from_query_fallback(q: str, prefer_after: str = "in") -> Optional[str]:
    """
    Try to extract a multi-word state phrase after 'in' or 'for' as a fallback.
    Returns the phrase if found (length>1), else None.
    """
    m = re.search(r"\b(?:in|for)\s+([A-Za-z &,\-()]+?)\s*(?:and|vs|v\.?|,|for|$)", q, re.I)
    if m:
        cand = m.group(1).strip()
        if len(cand) > 1:
            return cand
    return None

def _build_region_where(col_name: str, state_query: str, con=None, table_name: Optional[str]=None) -> str:
    """
    Build a robust WHERE clause for matching subdivisions / state-like columns.
    Avoids single-letter token matching and prefers multi-word exact phrases.
    Optionally fuzzy-matches distinct table values to find best alias.
    """
    canon = _canon_lower(state_query)
    if not canon:
        return "1=1"
    patterns = set()

    # prefer full phrase when multi-word (e.g. "tamil nadu")
    if len(canon) > 3 and ' ' in canon:
        patterns.add(canon)
    else:
        # tokenization but only keep tokens length >=2
        for tok in re.split(r"[,\-/()]+|\s+", canon):
            tok = tok.strip()
            if tok and len(tok) >= 2:
                patterns.add(tok)

    # add aliases if available
    if canon in STATE_ALIASES:
        patterns.update(STATE_ALIASES[canon])

    # optional fuzzy match against table distinct values
    if con is not None and table_name:
        try:
            df = con.execute(f"SELECT DISTINCT {col_name} as s FROM {table_name}").fetchdf()
            if 's' in df.columns:
                candidates = [str(x).strip() for x in df['s'].dropna().astype(str).unique().tolist()]
                best = _best_fuzzy_match(canon, candidates, threshold=0.6)
                if best:
                    patterns.add(best.lower())
        except Exception:
            pass

    # remove very short patterns
    patterns = {p for p in patterns if len(p) > 1}

    # fallback - use canonical even if short (avoid empty conditions)
    if not patterns:
        patterns.add(canon)

    conds = [f"LOWER({col_name}) LIKE '%{p}%'" for p in patterns if p]
    return "(" + " OR ".join(conds) + ")"

# ---------- table detection ----------
def _detect_tables(table_meta: Dict[str, dict]) -> Tuple[Optional[Tuple[str, dict]], Optional[Tuple[str, dict]]]:
    """
    Heuristic detection of rainfall and crop tables from TABLE_META
    """
    rain_cands = []
    crop_cands = []
    for tname, meta in table_meta.items():
        cols = [c.lower() for c in meta.get("columns", [])]
        text = " ".join(cols)
        # rainfall: monthly columns or 'annual' & some geographic column
        if any(m in text for m in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','annual']) and \
           any(x in text for x in ['subdivision','district','state','division']):
            rain_cands.append((tname, meta))
        # crop: crop, production, yield + state/district present
        if any(x in text for x in ['crop','production','yield','area']) and any(x in text for x in ['state','district']):
            crop_cands.append((tname, meta))
    rain = max(rain_cands, key=lambda x: len(x[1].get("columns",[]))) if rain_cands else None
    crop = max(crop_cands, key=lambda x: len(x[1].get("columns",[]))) if crop_cands else None
    return rain, crop

# ---------- year helpers ----------
def _distinct_parsed_years(con, tname: str, year_col: str) -> List[int]:
    try:
        df = con.execute(f"SELECT DISTINCT {year_col} as yr FROM {tname}").fetchdf()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to query years from {tname}: {e}")
    years = []
    if 'yr' in df.columns:
        for v in df['yr'].dropna().astype(str):
            y = _parse_year_token(v)
            if y:
                years.append(y)
    return sorted(list(set(years)))

def _get_overlap_years(con, crop_table, rain_table, last_n: Optional[int]=None) -> List[int]:
    crop_name, crop_meta = crop_table
    rain_name, rain_meta = rain_table
    crop_year_col = next((c for c in crop_meta['columns'] if c.lower() in ('year','yr')), None)
    rain_year_col = next((c for c in rain_meta['columns'] if c.lower() in ('year','yr')), None)
    yrs_crop = []
    yrs_rain = []
    if crop_year_col:
        yrs_crop = _distinct_parsed_years(con, crop_name, crop_year_col)
    else:
        for c in crop_meta['columns']:
            y = _parse_year_token(c)
            if y: yrs_crop.append(y)
    if rain_year_col:
        yrs_rain = _distinct_parsed_years(con, rain_name, rain_year_col)
    else:
        for c in rain_meta['columns']:
            y = _parse_year_token(c)
            if y: yrs_rain.append(y)
    if not yrs_crop or not yrs_rain:
        raise HTTPException(status_code=400, detail="Could not locate parsed years in crop or rainfall tables for overlap computation.")
    overlap = sorted(list(set(yrs_crop).intersection(set(yrs_rain))))
    if not overlap:
        raise HTTPException(status_code=400, detail="No overlapping years between crop and rainfall tables.")
    if last_n:
        crop_max = max(yrs_crop)
        start = crop_max - int(last_n) + 1
        overlap = [y for y in overlap if y >= start and y <= crop_max]
        if not overlap:
            raise HTTPException(status_code=400, detail=f"No overlapping years found within the requested last {last_n} years window.")
    return overlap

# ---------- production column detection ----------
def _find_prod_col(columns: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for key in ['production', 'prod', 'production_2023_24', 'production_2023', 'production_2020_21']:
        if key in lower_map:
            return lower_map[key]
    for c in columns:
        if re.search(r'production|prod|yield|quantity', c, re.I):
            return c
    return None

def _infer_single_year_from_cols(columns: List[str]) -> Optional[int]:
    for c in columns:
        y = _parse_year_token(c)
        if y:
            return y
    return None

# ---------- intent handlers ----------
def _handle_compare_rain(con, rain_meta, a_state: str, b_state: str, last_n: int = 5):
    tname, meta = rain_meta
    cols = meta['columns']
    year_col = next((c for c in cols if c.lower() in ('year','yr')), None)
    annual_col = next((c for c in cols if 'annual' in c.lower()), None)
    if not year_col:
        raise HTTPException(status_code=400, detail=f"Rainfall table {tname} missing YEAR column.")
    parsed_years = _distinct_parsed_years(con, tname, year_col)
    if not parsed_years:
        raise HTTPException(status_code=400, detail=f"No parsable years in rainfall table {tname}.")
    maxy = max(parsed_years)
    starty = maxy - int(last_n) + 1
    year_expr = _year_int_expr(year_col)
    if annual_col:
        base = f"SELECT {year_expr} as year, AVG(CAST({annual_col} AS DOUBLE)) as avg_annual FROM {tname} WHERE {year_expr} BETWEEN {starty} AND {maxy}"
    else:
        months = [c for c in cols if c[:3].upper() in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
        if not months:
            raise HTTPException(status_code=400, detail=f"Rain table {tname} lacks ANNUAL or monthly columns.")
        months_expr = " + ".join([f"CAST({m} AS DOUBLE)" for m in months])
        base = f"SELECT {year_expr} as year, AVG(({months_expr})) as avg_annual FROM {tname} WHERE {year_expr} BETWEEN {starty} AND {maxy}"
    # fallback for suspicious short tokens
    if len((a_state or "").strip()) <= 2:
        fb = _extract_state_from_query_fallback(a_state or "")
        if fb:
            a_state = fb
    if len((b_state or "").strip()) <= 2:
        fb = _extract_state_from_query_fallback(b_state or "")
        if fb:
            b_state = fb
    where_a = _build_region_where("SUBDIVISION", a_state, con, tname)
    where_b = _build_region_where("SUBDIVISION", b_state, con, tname)
    sql_a = base + f" AND {where_a} GROUP BY year ORDER BY year;"
    sql_b = base + f" AND {where_b} GROUP BY year ORDER BY year;"
    df_a = con.execute(sql_a).fetchdf()
    df_b = con.execute(sql_b).fetchdf()
    if df_a.empty and df_b.empty:
        raise HTTPException(status_code=400, detail=f"No rainfall data found for '{a_state}' or '{b_state}' between {starty}-{maxy}.")
    avg_a = float(df_a['avg_annual'].dropna().mean()) if not df_a.empty else None
    avg_b = float(df_b['avg_annual'].dropna().mean()) if not df_b.empty else None
    ans = f"Average annual rainfall ({starty}-{maxy}): {a_state}: {avg_a if avg_a is not None else 'no data'} mm; {b_state}: {avg_b if avg_b is not None else 'no data'}"
    return {"answer": ans, "evidence": {"table": tname, "sql_a": sql_a, "sql_b": sql_b, "source": meta.get("source")}}

def _handle_rain_trend(con, rain_meta, region: str, last_n: int = 10):
    tname, meta = rain_meta
    cols = meta['columns']
    year_col = next((c for c in cols if c.lower() in ('year','yr')), None)
    annual_col = next((c for c in cols if 'annual' in c.lower()), None)
    if not year_col:
        raise HTTPException(status_code=400, detail=f"Rainfall table {tname} missing YEAR column.")
    parsed_years = _distinct_parsed_years(con, tname, year_col)
    if not parsed_years:
        raise HTTPException(status_code=400, detail=f"No parsable years in rainfall table {tname}.")
    maxy = max(parsed_years)
    starty = maxy - int(last_n) + 1
    year_expr = _year_int_expr(year_col)
    where = _build_region_where("SUBDIVISION", region, con, tname)
    if annual_col:
        sql = f"SELECT {year_expr} as year, AVG(CAST({annual_col} AS DOUBLE)) as avg_annual FROM {tname} WHERE {year_expr} BETWEEN {starty} AND {maxy} AND {where} GROUP BY year ORDER BY year;"
    else:
        months = [c for c in cols if c[:3].upper() in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
        months_expr = " + ".join([f"CAST({m} AS DOUBLE)" for m in months])
        sql = f"SELECT {year_expr} as year, AVG(({months_expr})) as avg_annual FROM {tname} WHERE {year_expr} BETWEEN {starty} AND {maxy} AND {where} GROUP BY year ORDER BY year;"
    df = con.execute(sql).fetchdf()
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No rainfall trend data found for region '{region}' between {starty}-{maxy}.")
    slope_text = "trend summary unavailable"
    try:
        if np is not None and len(df) >= 2:
            xs = df.iloc[:,0].astype(int).to_numpy()
            ys = df.iloc[:,1].astype(float).to_numpy()
            m = float(np.polyfit(xs, ys, 1)[0])
            slope_text = "increasing" if m > 0 else ("decreasing" if m < 0 else "stable")
    except Exception:
        pass
    ans = f"Trend of rainfall in {region or ''} ({starty}-{maxy}): {slope_text}."
    return {"answer": ans, "evidence": {"table": tname, "sql": sql, "source": meta.get("source")}}

def _handle_top_crops(con, crop_meta, M: int, crop_type: Optional[str], state: Optional[str], last_n: Optional[int] = 5):
    tname, meta = crop_meta
    columns = meta['columns']
    col_year = next((c for c in columns if c.lower() in ('year','yr')), None)
    col_state = next((c for c in columns if 'state' in c.lower()), None)
    col_crop = next((c for c in columns if 'crop' in c.lower()), None)
    col_prod = _find_prod_col(columns)
    single_year = False
    inferred_year = None
    if not col_year:
        inferred_year = _infer_single_year_from_cols(columns)
        if inferred_year:
            single_year = True
    if not (col_state and col_crop and col_prod):
        raise HTTPException(status_code=400, detail=f"Crop table {tname} missing required columns. Found: {columns}")

    # Build SQL
    if single_year:
        sql = f"SELECT {col_crop} AS crop, SUM(CAST({col_prod} AS DOUBLE)) AS total_prod FROM {tname} WHERE LOWER({col_state}) LIKE '%{_canon_lower(state)}%' GROUP BY {col_crop} ORDER BY total_prod DESC LIMIT {int(M)};"
    else:
        parsed_years = _distinct_parsed_years(con, tname, col_year)
        if not parsed_years:
            raise HTTPException(status_code=400, detail=f"No parsed year tokens in crop table {tname}.")
        maxy = max(parsed_years)
        starty = maxy - int(last_n) + 1 if last_n else min(parsed_years)
        year_expr = _year_int_expr(col_year)
        sql = f"SELECT {col_crop} AS crop, SUM(CAST({col_prod} AS DOUBLE)) AS total_prod FROM {tname} WHERE {year_expr} BETWEEN {starty} AND {maxy} "
        if state:
            sql += f"AND LOWER({col_state}) LIKE '%{_canon_lower(state)}%' "
        if crop_type:
            sql += f"AND LOWER({col_crop}) LIKE '%{_canon_lower(crop_type)}%' "
        sql += f"GROUP BY {col_crop} ORDER BY total_prod DESC LIMIT {int(M)};"

    df = con.execute(sql).fetchdf()

    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    if df.empty:
        raise HTTPException(status_code=400, detail=f"No crop records found for the given filters ({crop_type or 'any crop'}, {state or 'any state'}) in years or single-year query.")
    crop_col = 'crop' if 'crop' in df.columns else df.columns[0]
    prod_col = 'total_prod' if 'total_prod' in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    rows_text = "; ".join([f"{r[crop_col]}: {round(float(r[prod_col]),2)}" for _, r in df.iterrows()])
    if single_year:
        ans = f"Top {M} crops in {state} (single-year inferred: {inferred_year}): {rows_text}"
    else:
        ans = f"Top {M} crops in {state or 'all states'} ({starty}-{maxy}): {rows_text}"
    return {"answer": ans, "evidence": {"table": tname, "sql": sql, "source": meta.get("source")}}

def _handle_district_extreme(con, crop_meta, state: str, crop: str, extreme: str = "max"):
    tname, meta = crop_meta
    columns = meta['columns']
    col_year = next((c for c in columns if c.lower() in ('year','yr')), None)
    col_state = next((c for c in columns if 'state' in c.lower()), None)
    col_district = next((c for c in columns if 'district' in c.lower()), None)
    col_crop = next((c for c in columns if 'crop' in c.lower()), None)
    col_prod = _find_prod_col(columns)
    if not (col_state and col_district and col_crop and col_prod):
        raise HTTPException(status_code=400, detail=f"Crop table {tname} missing required columns for district query. Found: {columns}")

    # cleanup 'crop' if it accidentally contains trailing phrases
    crop = re.sub(r"\s*(in the most recent year|in the most recent|in recent year|for the most recent year|\d{4})\s*$", "", crop, flags=re.I).strip()

    if col_year:
        parsed_years = _distinct_parsed_years(con, tname, col_year)
        if not parsed_years:
            raise HTTPException(status_code=400, detail=f"No parsed year tokens in crop table {tname}.")
        maxy = max(parsed_years)
        year_expr = _year_int_expr(col_year)
        sql = f"SELECT {col_district} AS district, SUM(CAST({col_prod} AS DOUBLE)) AS total_prod FROM {tname} WHERE {year_expr} = {maxy} AND LOWER({col_state}) LIKE '%{_canon_lower(state)}%' AND LOWER({col_crop}) LIKE '%{_canon_lower(crop)}%' GROUP BY {col_district} ORDER BY total_prod {'DESC' if extreme=='max' else 'ASC'} LIMIT 1;"
    else:
        sql = f"SELECT {col_district} AS district, SUM(CAST({col_prod} AS DOUBLE)) AS total_prod FROM {tname} WHERE LOWER({col_state}) LIKE '%{_canon_lower(state)}%' AND LOWER({col_crop}) LIKE '%{_canon_lower(crop)}%' GROUP BY {col_district} ORDER BY total_prod {'DESC' if extreme=='max' else 'ASC'} LIMIT 1;"
    df = con.execute(sql).fetchdf()
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No district-level production found for crop '{crop}' in {state}.")
    r = df.iloc[0]
    word = "Highest" if extreme == "max" else "Lowest"
    ans = f"{word} production district in {state} for crop '{crop}': {r['district']} ({round(float(r['total_prod']),2)} units)"
    return {"answer": ans, "evidence": {"table": tname, "sql": sql, "source": meta.get("source")}}

def _handle_correlation(con, rain_meta, crop_meta, state: str, crop: str, last_n: Optional[int] = None):
    if pd is None:
        raise HTTPException(status_code=400, detail="Pandas is required for correlation; please install pandas.")
    t_rain, meta_rain = rain_meta
    t_crop, meta_crop = crop_meta
    overlap_years = _get_overlap_years(con, (t_crop, meta_crop), (t_rain, meta_rain), last_n)
    if len(overlap_years) < 3:
        raise HTTPException(status_code=400, detail=f"Not enough overlapping years to compute correlation (need >=3, found {len(overlap_years)}). Overlap sample: {overlap_years[:10]}")
    starty, endy = min(overlap_years), max(overlap_years)
    cols_rain = meta_rain['columns']
    col_year_rain = next((c for c in cols_rain if c.lower() in ('year','yr')), None)
    col_annual = next((c for c in cols_rain if 'annual' in c.lower()), None)
    year_expr_rain = _year_int_expr(col_year_rain) if col_year_rain else str(starty)
    if col_annual:
        sql_rain = f"SELECT {year_expr_rain} as year, AVG(CAST({col_annual} AS DOUBLE)) as avg_rain FROM {t_rain} WHERE {year_expr_rain} BETWEEN {starty} AND {endy} AND {_build_region_where('SUBDIVISION', state, con, t_rain)} GROUP BY year ORDER BY year;"
    else:
        months = [c for c in cols_rain if c[:3].upper() in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
        if not months:
            raise HTTPException(status_code=400, detail="Rain table lacks ANNUAL or monthly columns needed for correlation.")
        months_expr = " + ".join([f"CAST({m} AS DOUBLE)" for m in months])
        sql_rain = f"SELECT {year_expr_rain} as year, AVG(({months_expr})) as avg_rain FROM {t_rain} WHERE {year_expr_rain} BETWEEN {starty} AND {endy} AND {_build_region_where('SUBDIVISION', state, con, t_rain)} GROUP BY year ORDER BY year;"
    cols_crop = meta_crop['columns']
    col_year_crop = next((c for c in cols_crop if c.lower() in ('year','yr')), None)
    col_prod = _find_prod_col(cols_crop)
    col_crop_field = next((c for c in cols_crop if 'crop' in c.lower()), None)
    col_state = next((c for c in cols_crop if 'state' in c.lower()), None)
    if not (col_prod and col_crop_field and col_state):
        raise HTTPException(status_code=400, detail="Crop table missing required columns for correlation (production, crop, state).")
    year_expr_crop = _year_int_expr(col_year_crop) if col_year_crop else str(starty)
    sql_crop = f"SELECT {year_expr_crop} as year, SUM(CAST({col_prod} AS DOUBLE)) as prod FROM {t_crop} WHERE {year_expr_crop} BETWEEN {starty} AND {endy} AND LOWER({col_state}) LIKE '%{_canon_lower(state)}%' AND LOWER({col_crop_field}) LIKE '%{_canon_lower(crop)}%' GROUP BY year ORDER BY year;"
    df_rain = con.execute(sql_rain).fetchdf()
    df_crop = con.execute(sql_crop).fetchdf()
    if df_rain.empty:
        raise HTTPException(status_code=400, detail=f"No rainfall series found for '{state}' in years {starty}-{endy}. SQL: {sql_rain}")
    if df_crop.empty:
        raise HTTPException(status_code=400, detail=f"No crop production series found for '{crop}' in {state} for years {starty}-{endy}. SQL: {sql_crop}")
    df_rain.columns = [c.lower() for c in df_rain.columns]
    df_crop.columns = [c.lower() for c in df_crop.columns]
    df_join = pd.merge(df_rain, df_crop, on='year', how='inner')
    if df_join.empty or len(df_join) < 3:
        raise HTTPException(status_code=400, detail=f"Not enough merged points to compute correlation (need >=3, found {len(df_join)}).")
    corr_val = float(df_join['avg_rain'].corr(df_join['prod']))
    ans = f"Pearson correlation between avg annual rainfall and production for {crop} in {state or ''} ({starty}-{endy}): r = {round(corr_val,3)} (n={len(df_join)})"
    evidence = {"sql_rain": sql_rain, "sql_crop": sql_crop, "table_rain": t_rain, "table_crop": t_crop, "overlap_years": [starty, endy], "n": len(df_join)}
    return {"answer": ans, "evidence": evidence}

def _handle_crop_trend(con, crop_meta, crop_name: str, region: Optional[str], last_n: Optional[int] = 10):
    tname, meta = crop_meta
    columns = meta['columns']
    col_year = next((c for c in columns if c.lower() in ('year','yr')), None)
    col_prod = _find_prod_col(columns)
    col_crop = next((c for c in columns if 'crop' in c.lower()), None)
    col_state = next((c for c in columns if 'state' in c.lower()), None)
    if not (col_year and col_prod and col_crop):
        raise HTTPException(status_code=400, detail=f"Crop table {tname} lacks a YEAR column to compute trend.")
    parsed_years = _distinct_parsed_years(con, tname, col_year)
    if not parsed_years:
        raise HTTPException(status_code=400, detail="No parsed years in crop table.")
    maxy = max(parsed_years)
    starty = maxy - int(last_n) + 1 if last_n else min(parsed_years)
    year_expr = _year_int_expr(col_year)
    sql = f"SELECT {year_expr} as year, SUM(CAST({col_prod} AS DOUBLE)) as prod FROM {tname} WHERE {year_expr} BETWEEN {starty} AND {maxy}"
    if region and col_state:
        sql += f" AND LOWER({col_state}) LIKE '%{_canon_lower(region)}%'"
    if crop_name:
        sql += f" AND LOWER({col_crop}) LIKE '%{_canon_lower(crop_name)}%'"
    sql += " GROUP BY year ORDER BY year;"
    df = con.execute(sql).fetchdf()
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No crop trend data found for the given filters in years {starty}-{maxy}.")
    slope_text = "trend summary unavailable"
    try:
        if np is not None and len(df) >= 2:
            xs = df.iloc[:,0].astype(int).to_numpy()
            ys = df.iloc[:,1].astype(float).to_numpy()
            m = float(np.polyfit(xs, ys, 1)[0])
            slope_text = "increasing" if m > 0 else ("decreasing" if m < 0 else "stable")
    except Exception:
        pass
    ans = f"Trend of production for {crop_name or 'selected crops'} in {region or 'region'} ({starty}-{maxy}): {slope_text}."
    return {"answer": ans, "evidence": {"table": tname, "sql": sql, "source": meta.get("source")}}

def _handle_compare_rain_and_crops(con, rain_meta, crop_meta, a_state: str, b_state: str, last_n: int = 5, top_m: int = 5):
    # fallback extraction if a_state or b_state suspiciously short
    if len((a_state or "").strip()) <= 2 or len((b_state or "").strip()) <= 2:
        fb = _extract_state_from_query_fallback(a_state or "")
        if fb and (' and ' in fb.lower() or ' vs ' in fb.lower() or ' v. ' in fb.lower()):
            parts = re.split(r"\s+and\s+|\s+vs\s+|\s+v\.\s*", fb, flags=re.I)
            if len(parts) >= 2:
                a_state = parts[0].strip()
                b_state = parts[1].strip()
    rain_res = _handle_compare_rain(con, rain_meta, a_state, b_state, last_n)
    crop_res_a = _handle_top_crops(con, crop_meta, top_m, crop_type="", state=a_state, last_n=last_n)
    crop_res_b = _handle_top_crops(con, crop_meta, top_m, crop_type="", state=b_state, last_n=last_n)
    ans_lines = []
    ans_lines.append(rain_res["answer"])
    ans_lines.append("")
    ans_lines.append(f"Top {top_m} crops in {a_state} (last {last_n} years or available):")
    ans_lines.append(crop_res_a["answer"])
    ans_lines.append("")
    ans_lines.append(f"Top {top_m} crops in {b_state} (last {last_n} years or available):")
    ans_lines.append(crop_res_b["answer"])
    combined_answer = "\n".join(ans_lines)
    combined_evidence = {
        "rain_evidence": rain_res.get("evidence"),
        "crop_a_evidence": crop_res_a.get("evidence"),
        "crop_b_evidence": crop_res_b.get("evidence"),
        "notes": "Combined rainfall + top-crops result."
    }
    return {"answer": combined_answer, "evidence": combined_evidence}

# ---------- top-level router ----------
def handle_query(query: str, con, table_meta: Dict[str, dict]) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query provided.")
    rain_table, crop_table = _detect_tables(table_meta)

    # Combined intent first
    m = RE_COMBINED_RAIN_CROPS.search(q)
    if m and rain_table and crop_table:
        top_m = int(m.group(1)) if m.group(1) else 5
        a = m.group(2).strip()
        b = m.group(3).strip()
        years = int(m.group(4)) if m.group(4) else 5
        # fallback if single-letter tokens
        if len(a) <= 2 or len(b) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb and (' and ' in fb.lower() or ' vs ' in fb.lower()):
                parts = re.split(r"\s+and\s+|\s+vs\s+|\s+v\.\s*", fb, flags=re.I)
                if len(parts) >= 2:
                    a, b = parts[0].strip(), parts[1].strip()
        return _handle_compare_rain_and_crops(con, rain_table, crop_table, a, b, last_n=years, top_m=top_m)

    # Other intents
    m = RE_COMPARE_RAIN.search(q)
    if m:
        if not rain_table:
            raise HTTPException(status_code=400, detail="No rainfall table ingested.")
        a = m.group(1).strip()
        b = m.group(2).strip()
        years = int(m.group(3)) if m.group(3) else 5
        # fallback extraction
        if len(a) <= 2 or len(b) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb and (' and ' in fb.lower() or ' vs ' in fb.lower()):
                parts = re.split(r"\s+and\s+|\s+vs\s+|\s+v\.\s*", fb, flags=re.I)
                if len(parts) >= 2:
                    a, b = parts[0].strip(), parts[1].strip()
        return _handle_compare_rain(con, rain_table, a, b, years)

    m = RE_RAIN_TREND.search(q)
    if m:
        if not rain_table:
            raise HTTPException(status_code=400, detail="No rainfall table ingested.")
        region = (m.group(1) or "").strip()
        years = int(m.group(2)) if m.group(2) else 10
        # fallback if region short
        if len(region) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb:
                region = fb
        return _handle_rain_trend(con, rain_table, region, years)

    m = RE_TOP_CROPS.search(q)
    if m:
        if not crop_table:
            raise HTTPException(status_code=400, detail="No crop table ingested.")
        M = int(m.group(1))
        crop_type = (m.group(2) or "").strip()
        state = (m.group(3) or "").strip()
        years = int(m.group(4)) if m.group(4) else 5
        # fallback for short state tokens
        if len(state) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb:
                # try split logic
                if ' and ' in fb.lower():
                    parts = re.split(r"\s+and\s+", fb, flags=re.I)
                    state = parts[0].strip()
                else:
                    state = fb
        return _handle_top_crops(con, crop_table, M, crop_type or None, state or None, years)

    m = RE_DISTRICT_EXTREME.search(q)
    if m:
        if not crop_table:
            raise HTTPException(status_code=400, detail="No crop table ingested.")
        state = (m.group(1) or "").strip()
        extreme_word = m.group(2).strip().lower()
        crop = m.group(3).strip()
        # cleanup trailing phrases in crop
        crop = re.sub(r"\s*(in the most recent year|in the most recent|in recent year|for the most recent year|\d{4})\s*$", "", crop, flags=re.I).strip()
        # fallback for state
        if not state:
            state_guess = re.search(r"in\s+([A-Za-z &,\-()]+?)\s+has|in\s+([A-Za-z &,\-()]+?)\s+which", q, re.I)
            if state_guess:
                state = (state_guess.group(1) or state_guess.group(2) or "").strip()
        if not state:
            raise HTTPException(status_code=400, detail="Please specify the state in the query, e.g. 'Which district in Punjab has the highest production of Wheat?'")
        extreme = "max" if extreme_word in ("highest","maximum","max") else "min"
        return _handle_district_extreme(con, crop_table, state, crop, extreme)

    m = RE_CORRELATE.search(q)
    if m:
        if not (rain_table and crop_table):
            raise HTTPException(status_code=400, detail="Correlation requires both rainfall and crop tables ingested.")
        crop_name = (m.group(1) or "").strip()
        state = (m.group(2) or "").strip()
        years_match = re.search(r"last\s+(\d+)\s+years?", q, re.I)
        years = int(years_match.group(1)) if years_match else None
        if len(state) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb:
                state = fb
        return _handle_correlation(con, rain_table, crop_table, state or "", crop_name or "rice", years)

    m = RE_CROP_TREND.search(q)
    if m:
        if not crop_table:
            raise HTTPException(status_code=400, detail="No crop table ingested.")
        crop_name = (m.group(1) or "").strip()
        region = (m.group(2) or "").strip()
        years = int(m.group(3)) if m.group(3) else 10
        if len(region) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb:
                region = fb
        return _handle_crop_trend(con, crop_table, crop_name or "", region or "", years)

    m = RE_COMPARE_CROP.search(q)
    if m:
        if not crop_table:
            raise HTTPException(status_code=400, detail="No crop table ingested.")
        a = m.group(1).strip()
        b = m.group(2).strip()
        years = int(m.group(3)) if m.group(3) else 5
        # fallback
        if len(a) <= 2 or len(b) <= 2:
            fb = _extract_state_from_query_fallback(q)
            if fb and (' and ' in fb.lower() or ' vs ' in fb.lower()):
                parts = re.split(r"\s+and\s+|\s+vs\s+|\s+v\.\s*", fb, flags=re.I)
                if len(parts) >= 2:
                    a, b = parts[0].strip(), parts[1].strip()
        res_a = _handle_top_crops(con, crop_table, 5, "", a, years)
        res_b = _handle_top_crops(con, crop_table, 5, "", b, years)
        ans = f"Comparison top crops {a} vs {b} ({years} years):\n\n{a}: {res_a['answer']}\n\n{b}: {res_b['answer']}"
        return {"answer": ans, "evidence": {"a": res_a.get("evidence"), "b": res_b.get("evidence")}}

    m = RE_RAINFALL_EXTREME.search(q)
    if m:
        v = m.group(1).strip().lower()
        year = int(m.group(2))
        if not rain_table:
            raise HTTPException(status_code=400, detail="No rainfall table ingested.")
        tname, meta = rain_table
        cols = meta['columns']
        col_year = next((c for c in cols if c.lower() in ('year','yr')), None)
        col_annual = next((c for c in cols if 'annual' in c.lower()), None)
        if not col_annual:
            months = [c for c in cols if c[:3].upper() in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
            months_expr = " + ".join([f"CAST({m} AS DOUBLE)" for m in months])
            year_expr = _year_int_expr(col_year) if col_year else str(year)
            sql = f"SELECT SUBDIVISION, AVG(({months_expr})) as total FROM {tname} WHERE {year_expr} = {year} GROUP BY SUBDIVISION ORDER BY total {'DESC' if v in ('highest','max') else 'ASC'} LIMIT 1;"
        else:
            year_expr = _year_int_expr(col_year) if col_year else str(year)
            sql = f"SELECT SUBDIVISION, AVG(CAST({col_annual} AS DOUBLE)) as total FROM {tname} WHERE {year_expr} = {year} GROUP BY SUBDIVISION ORDER BY total {'DESC' if v in ('highest','max') else 'ASC'} LIMIT 1;"
        df = con.execute(sql).fetchdf()
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No rainfall data found for year {year}.")
        r = df.iloc[0]
        word = "highest" if v in ('highest','max') else "lowest"
        ans = f"Subdivision with {word} rainfall in {year}: {r['SUBDIVISION']} ({round(float(r['total']),2)} mm)"
        return {"answer": ans, "evidence": {"table": tname, "sql": sql}}

    m = RE_GENERIC_TOP.search(q)
    if m and crop_table:
        count = int(m.group(1))
        thing = m.group(2).strip()
        region = m.group(3).strip()
        return _handle_top_crops(con, crop_table, count, thing, region, 5)

    # fallback
    raise HTTPException(status_code=400, detail="Could not parse query with current intents. Examples:\n- Compare average annual rainfall in Kerala and Tamil Nadu for the last 5 years\n- Trend of rainfall in Kerala for the last 10 years\n- Top 5 most produced crops in Maharashtra for the last 5 years\n- Which district in Punjab has the highest production of Wheat in the most recent year\n- Correlate rice production in Andhra Pradesh for the last 10 years with rainfall\n- Compare rainfall and top crops in Maharashtra and Karnataka for the last 5 years")
