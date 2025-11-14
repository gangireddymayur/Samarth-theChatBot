# Samarth: Intelligent Agricultural & Climate Q&A System

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
A provided `test.py` runs 20 diagnostic questions that verify:
- intent recognition  
- rainfall queries  
- crop queries  
- correlation logic  
- mixed-domain logic  

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

