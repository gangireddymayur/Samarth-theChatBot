# Samarth: Intelligent Agricultural & Climate Q&A System

![tests](https://github.com/gangireddymayur/Samarth-theChatBot/actions/workflows/tests.yml/badge.svg)

## 🚀 Overview
Samarth is an end‑to‑end intelligent Question‑Answering system built to enable cross‑domain insights from **Indian Government datasets**—primarily rainfall data and crop production data.  
This system unifies inconsistent data sources, performs natural‑language query parsing, executes DuckDB‑backed analytical pipelines, and produces fully traceable, SQL‑driven answers.

It is designed as a prototype for **Bharat Digital Fellowship – Project Samarth**.

---

## 🧠 Key Capabilities
- Natural‑language to structured query conversion  
- Rainfall analysis (IMD subdivision/state level)  
- Crop production analysis (state/district level)  
- Cross-domain reasoning: rainfall ↔ crop correlations  
- Multi‑year trends, comparisons, and top‑M crop statistics  
- Automatic table selection & intelligent fallback mechanisms  
- Full traceability with SQL evidence for every answer  
- Real‑time duckdb queries with flexible schema handling  

---

## 🏗️ System Architecture

### **Backend (FastAPI + DuckDB)**
- Auto‑ingests local/remote CSVs using a configurable `config.py`
- Registers each dataset as a DuckDB table dynamically
- A **super‑intelligent `query_handler.py`** handles:
  - Intent recognition  
  - Entity extraction (states, crops, years, metrics)  
  - SQL generation  
  - Join alignment across mismatched year ranges  
  - Schema normalization (column rename tolerances, fuzzy matching)

### **Frontend (Streamlit Chat UI)**
- ChatGPT‑style conversational interface  
- Dark‑themed responsive layout  
- Sidebar with example queries  
- Evidence viewer for SQL + previews  
- Clean bottom input bar with scrolling history  

---

## 📚 Datasets Used

### **Rainfall Dataset (Kaggle – IMD Rainfall Data)**
- Subdivision-level rainfall from **1901–2017**
- Monthly and annual rainfall totals
- Maps well to Indian states using harmonized names

### **Crop Dataset (data.gov.in – DES District Crop Statistics)**
- State + district crop production data **1997–2020**
- Crop-wise area, production & yield  
- Year format normalized (e.g., "2014-15" → 2014)

---

## 🔍 Example Supported Questions
- “Compare rainfall in Kerala and Tamil Nadu for the last 5 years.”  
- “Top 5 crops in Maharashtra for the last 7 years.”  
- “Correlate rice production in Andhra Pradesh with rainfall trends.”  
- “Which district in Punjab had the highest wheat production last year?”  
- “Compare rainfall and top crops between Maharashtra and Karnataka.”  

---

## 🛠️ Installation

```bash
git clone <your-repo-url>
cd samarth
pip install -r requirements.txt
```

### Run Backend
```bash
python -m uvicorn backend.main:app --reload --port 8765
```

### Run Frontend
```bash
streamlit run frontend/app.py
```

---

## 📁 Project Structure

```
samarth/
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── utils/
│   │   └── query_handler.py
│   └── data/
│       ├── rainfall.csv
│       └── crop_production.csv
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│
└── README.md
```

---

## 🧪 Testing
An automated pytest suite in `tests/` runs every supported intent through the real FastAPI app (in-process, no server needed) and checks that:
- each question type returns an answer **with the SQL that produced it**
- both places in a comparison are kept intact (e.g. "Tamil Nadu", not "T")
- "last N years" windows are applied
- unknown and empty questions are rejected with a clear message

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run on every push via GitHub Actions. `test.py` is still available as a manual diagnostic script against a running backend.

---

## 🔐 Core Values
- **Accuracy** — Every answer includes SQL-level evidence.  
- **Traceability** — Dataset IDs + source URLs included.  
- **Data Sovereignty** — Fully offline-capable, no external LLM required.  
- **Resilience** — Handles schema mismatch, fuzzy names, year misalignment.  

---


## 📄 License
Open-source for educational purposes.

---

## 👤 Author
Samarth Prototype – Powered by FastAPI, DuckDB, and Streamlit.

