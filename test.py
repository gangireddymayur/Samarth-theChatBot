# test.py
# Run this to test many sample queries against your Samarth backend.
# Usage: python test.py

import requests
import json
import time

BASE = "http://127.0.0.1:8765"   # change if your backend runs on different host/port
ENDPT = f"{BASE}/chat"
HEADERS = {"Content-Type": "application/json"}

QUERIES = [
    # Rainfall intents
    "Compare average annual rainfall in Kerala and Tamil Nadu for the last 5 years",
    "Trend of rainfall in Kerala for the last 10 years",
    "Which subdivision had the lowest rainfall in 2015?",
    "Which subdivision had the highest rainfall in 2010?",

    # Crop intents
    "Top 5 most produced crops in Maharashtra for the last 5 years",
    "Top 10 most produced crops in Andhra Pradesh for the last 7 years",

    # Crop trend + compare crop
    "Analyze the trend of Rice production in Andhra Pradesh for the last 15 years",
    "Compare production in Punjab and Haryana for the last 5 years",

    # Correlation intent (crop vs rainfall)
    "Correlate rice production in Andhra Pradesh for the last 10 years with rainfall",
    "Correlate maize production in Karnataka with rainfall (last 8 years)",

    # Combined intent (rain + top crops)
    "Compare rainfall and top crops in Maharashtra and Karnataka for the last 5 years",
    "Compare rainfall and top 10 crops in Kerala and Tamil Nadu for the last 7 years",

    # Generic top / fuzzy tests / edge-cases
    "Top 3 crops in Andaman and Nicobar Islands",
    "Correlate rice production in Andaman and Nicobar Islands with rainfall",
    "Compare rainfall and top crops in Andhra Pradesh and Odisha for the last 5 years",

]

def pretty_print_response(resp):
    print("Status:", resp.status_code)
    try:
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print("Non-JSON response:")
        print(resp.text)

def main():
    print("Samarth diagnostic test starting...")
    for i, q in enumerate(QUERIES, start=1):
        if not q:
            print(f"\n[{i}] Skipping empty query (will still send to test validation)...")
        print("\n" + "="*80)
        print(f"[{i}] Query:")
        print(q if q else "<EMPTY>")
        payload = {"query": q}
        try:
            resp = requests.post(ENDPT, json=payload, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            continue
        pretty_print_response(resp)
        # small pause so logs are readable
        time.sleep(0.4)
    print("\nDone. If any test returned 4xx/5xx, copy the full JSON output and /datasets listing to debug.")

if __name__ == "__main__":
    main()
