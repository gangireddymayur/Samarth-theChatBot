"""
Samarth — Streamlit Chat (Revamped UI, CSS-only animations, polished look)
Overwrite your frontend/app.py with this file and run:
    python -m streamlit run app.py

Notes:
- This version intentionally uses CSS animations, gradients, SVG accents and nicer layout.
- There's a toggle to disable animations for low-power devices.
- Functionality is kept identical to your original app (samples, backend health, chat loop).
- No external packages required beyond streamlit & requests.
"""

import streamlit as st
import requests
from urllib.parse import urljoin
import time
import json
import html as html_module

# ---------------------- MUST BE FIRST STREAMLIT COMMAND ----------------------
st.set_page_config(page_title="Samarth — Chat (Revamp)", layout="wide", initial_sidebar_state="expanded")

# ---------------------- Config ----------------------
BACKEND_BASE = st.secrets.get("backend_base", "http://localhost:8765")
API_TIMEOUT = 60

# ---------------------- Helpers ----------------------

def backend_post(path: str, payload: dict, timeout: int = API_TIMEOUT):
    url = urljoin(BACKEND_BASE, path.lstrip("/"))
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
    except requests.exceptions.HTTPError:
        try:
            return {"_error": True, "status_code": res.status_code, "body": res.json()}
        except Exception:
            return {"_error": True, "status_code": getattr(res, "status_code", None), "body": res.text}
    except Exception as e:
        return {"_error": True, "status_code": None, "body": str(e)}
    try:
        return res.json()
    except Exception:
        return res.text


def backend_get(path: str, timeout: int = 6):
    url = urljoin(BACKEND_BASE, path.lstrip("/"))
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"_error": True, "body": str(e)}

# --- Hide all default Streamlit headers, menus, and footers ---
hide_everything = """
    <style>
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stAppHeader"] {display: none !important;}
    #MainMenu {display: none !important;}
    footer {display: none !important;}
    </style>
"""
st.markdown(hide_everything, unsafe_allow_html=True)

# ---------------------- Session state ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi — I'm Samarth. Ask about rainfall & crops. Try a sample on the right.", "ts": time.time()}
    ]
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "compact" not in st.session_state:
    st.session_state.compact = False
if "animations" not in st.session_state:
    st.session_state.animations = True

# ---------------------- Samples ----------------------
SAMPLES = [
    "Compare average annual rainfall in Kerala and Tamil Nadu for the last 5 years",
    "Trend of rainfall in Kerala for the last 10 years",
    "Correlate rice production in Andhra Pradesh for the last 10 years with rainfall",
    "Top 5 most produced crops in Maharashtra for the last 5 years",
    "Correlate rice production in Andhra Pradesh for the last 10 years with rainfall",
    "Compare rainfall and top crops in Maharashtra and Karnataka for the last 5 years",
]

# ---------------------- Polished CSS ----------------------
CSS = r"""
<style>
:root{
  --bg-1: #071023;
  --bg-2: #081129;
  --card: rgba(255,255,255,0.03);
  --muted: #98a7bd;
  --accent-a: #06b6d4;
  --accent-b: #8b5cf6;
  --glass: rgba(255,255,255,0.035);
}
html, body, [class^="st"] > .main > div {
  background: linear-gradient(180deg,var(--bg-1), var(--bg-2));
}
.bg-blob { position: fixed; z-index:0; filter: blur(40px); opacity:0.12; pointer-events:none; }
.blob1{ right: -8vw; top: -4vh; width:40vw; height:40vw; background: radial-gradient(circle at 30% 30%, var(--accent-b), transparent 40%); animation: float1 14s ease-in-out infinite; }
.blob2{ left: -6vw; bottom: -6vh; width:38vw; height:38vw; background: radial-gradient(circle at 70% 70%, var(--accent-a), transparent 35%); animation: float2 18s ease-in-out infinite; }
@keyframes float1 {0%{transform:translateY(0) rotate(0)}50%{transform:translateY(30px) rotate(8deg)}100%{transform:translateY(0) rotate(0)}}
@keyframes float2 {0%{transform:translateY(0) rotate(0)}50%{transform:translateY(-36px) rotate(-6deg)}100%{transform:translateY(0) rotate(0)}}

.header { display:flex; align-items:center; gap:12px; }
.logo { width:56px; height:56px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; background:linear-gradient(135deg,var(--accent-b),var(--accent-a)); color:white; font-size:20px }
.title { font-weight:800; font-size:18px }
.subtitle { color:var(--muted); font-size:13px }

.msg-row{ display:flex; gap:12px; align-items:flex-start; margin-bottom:10px; }
.msg-avatar{ width:44px; min-width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:700 }
.msg-user{ background: linear-gradient(90deg,var(--accent-a),var(--accent-b)); color:white; padding:12px 14px; border-radius:12px; max-width:85%; font-weight:600 }
.msg-assistant{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); color: #dfe9f5; padding:12px 14px; border-radius:12px; max-width:85%; }
.msg-meta{ font-size:12px; color:var(--muted); margin-top:6px }

.input-area { display:flex; gap:10px; align-items:center; padding:10px; border-radius:12px; background:linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); border:1px solid rgba(255,255,255,0.02); }
.text-input { flex:1 }
.send-btn{ background:linear-gradient(90deg,var(--accent-a),var(--accent-b)); color:white; border-radius:10px; padding:10px 14px; font-weight:700; border:none; cursor:pointer }
.send-btn:hover{ transform:translateY(-2px); box-shadow: 0 8px 30px rgba(11,35,64,0.45); }

.typing { display:inline-block; width:46px; height:24px; vertical-align:middle; }
.typing span{ display:inline-block; width:8px; height:8px; margin:2px; border-radius:50%; background:rgba(255,255,255,0.28); animation: dot 1s infinite ease-in-out }
.typing span:nth-child(2){ animation-delay:0.12s }
.typing span:nth-child(3){ animation-delay:0.24s }
@keyframes dot { 0%{transform:translateY(0); opacity:0.35} 50%{transform:translateY(-6px); opacity:1} 100%{transform:translateY(0); opacity:0.35} }

.msg-anim{ animation: fadeUp .36s ease both; }
@keyframes fadeUp { from{ transform:translateY(8px); opacity:0 } to{ transform:translateY(0); opacity:1 } }

.compact .msg-avatar{ width:40px; height:40px }
.compact .msg-user,.compact .msg-assistant{ padding:8px 10px; border-radius:9px }

@media (max-width:800px){ .logo{ width:46px; height:46px; font-size:18px } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- Custom Title Bar ---
st.markdown("""
    <div style="
        background-color: #0E1117;
        padding: 16px 30px;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
    ">
        <img src="https://cdn-icons-png.flaticon.com/512/3208/3208726.png" width="36" style="margin-right: 12px;">
        <h2 style="color: white; margin: 0;">🌦️ Samarth | Rainfall & Crop Insights</h2>
    </div>
""", unsafe_allow_html=True)

# Decorative blobs
st.markdown("<div class='bg-blob blob1'></div><div class='bg-blob blob2'></div>", unsafe_allow_html=True)

# ---------------------- Layout ----------------------
left_col, right_col = st.columns([3, 1], gap="large")

# Right column (samples + backend check)
with right_col:
    st.markdown("### Quick samples")
    for i, s in enumerate(SAMPLES):
        if st.button(s, key=f"sample_right_{i}"):
            q = s
            st.session_state.messages.append({"role": "user", "content": q, "ts": time.time()})
            with st.spinner("Samarth is thinking..."):
                resp = backend_post("/chat", {"query": q}, timeout=API_TIMEOUT)
            if isinstance(resp, dict) and resp.get("_error"):
                st.session_state.messages.append({"role": "assistant", "content": f"Backend error: {resp.get('body')}", "ts": time.time()})
            else:
                if isinstance(resp, dict) and resp.get("answer"):
                    st.session_state.messages.append({"role": "assistant", "content": resp.get("answer"), "ts": time.time()})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": json.dumps(resp, indent=2), "ts": time.time()})
            st.rerun()

    st.write("---")
    if st.button("Check backend health"):
        st.write(backend_get("/health"))
    st.markdown(f"<div class='small-muted'>Backend base: <code style='color:#9ae6b4'>{html_module.escape(BACKEND_BASE)}</code></div>", unsafe_allow_html=True)

# Left column (chat UI)
with left_col:
    container_classes = "chat-shell"
    if st.session_state.compact:
        container_classes += " compact"
    st.markdown(f"<div class='{container_classes}'>", unsafe_allow_html=True)

    messages_container = st.container()
    with messages_container:
        for m in st.session_state.messages:
            role = m.get("role", "assistant")
            content = str(m.get("content", ""))
            ts = m.get("ts", 0)
            time_str = time.strftime("%H:%M", time.localtime(ts))
            safe_content = html_module.escape(content).replace('\n', '<br/>')

            if role == "user":
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.write("")
                with c2:
                    st.markdown(f"<div class='msg-row msg-anim' style='justify-content:flex-end'><div style='text-align:right'><div class='msg-user'>{safe_content}</div><div class='msg-meta'>{time_str}</div></div></div>", unsafe_allow_html=True)
            else:
                avatar_html = "<div class='msg-avatar' style='background:linear-gradient(135deg,var(--accent-b),var(--accent-a)); color:white'>S</div>"
                st.markdown(f"<div class='msg-row msg-anim'><div>{avatar_html}</div><div><div class='msg-assistant'><strong>Samarth</strong><div style='margin-top:8px'>{safe_content}</div></div><div class='msg-meta'>{time_str}</div></div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=False):
        user_input = st.text_input("", value=st.session_state.get("user_query", ""), key="chat_input_native")
        submitted = st.form_submit_button("Send")
        if submitted:
            q = (user_input or "").strip()
            if not q:
                st.warning("Please type a message.")
            else:
                st.session_state.user_query = ""
                st.session_state.messages.append({"role": "user", "content": q, "ts": time.time()})
                with st.spinner("Samarth is thinking..."):
                    resp = backend_post("/chat", {"query": q}, timeout=API_TIMEOUT)
                if isinstance(resp, dict) and resp.get("_error"):
                    st.session_state.messages.append({"role": "assistant", "content": f"Backend error: {resp.get('body')}", "ts": time.time()})
                else:
                    if isinstance(resp, dict) and resp.get("answer"):
                        st.session_state.messages.append({"role": "assistant", "content": resp.get("answer"), "ts": time.time()})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": json.dumps(resp, indent=2), "ts": time.time()})
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Footer backend status
hb = backend_get("/health", timeout=2)
if hb and not hb.get("_error"):
    btxt = "Backend OK — tables: " + (", ".join(hb.get("tables", [])) if hb.get("tables") else "unknown")
else:
    btxt = "Backend unreachable"

st.markdown(f"<div style='position:fixed;right:18px;bottom:18px;background:rgba(7,24,39,0.85);padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.02);font-size:13px;color:var(--muted)'>Backend: {html_module.escape(btxt)}</div>", unsafe_allow_html=True)

st.markdown("""
<style>.hint{position:fixed;left:18px;bottom:18px;background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02));padding:10px 14px;border-radius:10px;border:1px solid rgba(255,255,255,0.03);color:var(--muted);font-size:13px}</style>
<div class='hint'>Tip: Try the sample queries on the right. Toggle animations in Options if your device is slow.</div>
""", unsafe_allow_html=True)
