# agent.py
# Streamlit AI Agent — 2-column UI (Ask left, Docs+Search right)
# - Minimal controls (no temperature, no counters)
# - Drag & drop (PDF, DOCX, TXT, CSV, MD)
# - Search in docs: now only AI-assist (summary from snippets)
# - Mistral API via HTTPS (no SDK)
# Requirements (add to requirements.txt as needed):
#   requests
#   pypdf
#   python-docx

from __future__ import annotations
import os, re, time, random
from html import escape
from typing import Dict, List, Tuple

import streamlit as st

# Optional parsers
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore

try:
    from docx import Document
except Exception:
    Document = None  # type: ignore

# requests (soft dep)
_HAVE_REQUESTS = True
try:
    import requests as _requests
except Exception:
    _HAVE_REQUESTS = False

# ---------------------------- Config ----------------------------
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
MAX_CONTEXT_CHARS = 2000     # total chars from uploaded docs (for chat)
MAX_FILE_CHARS = 4000        # per-file cap to avoid huge payloads
HISTORY_TURNS = 6           # number of prior messages to keep

SYSTEM_PROMPT = """You are a helpful, concise AI assistant embedded in a Debt Capital Markets workspace.
- Always answer clearly and directly.
- Use any provided document context when relevant; if not relevant, ignore it.
- If information is missing, say so briefly rather than inventing facts.
"""

# ---------------------------- Utils ----------------------------
def _get_api_key() -> str:
    key = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, "secrets") else ""
    return key or os.environ.get("MISTRAL_API_KEY", "") or ""

def _safe_decode(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")

def _read_pdf(file) -> str:
    if PdfReader is None:
        return "⚠️ PDF support not available: install 'pypdf'."
    try:
        reader = PdfReader(file)
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(text_parts)
    except Exception as e:
        return f"⚠️ Failed to parse PDF: {e}"

def _read_docx(file) -> str:
    if Document is None:
        return "⚠️ DOCX support not available: install 'python-docx'."
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"⚠️ Failed to parse DOCX: {e}"

def _read_text(file) -> str:
    try:
        raw = file.read() if hasattr(file, "read") else file
        if isinstance(raw, (bytes, bytearray)):
            return _safe_decode(raw)
        return str(raw)
    except Exception as e:
        return f"⚠️ Failed to read text: {e}"

def _parse_upload(upload) -> Tuple[str, str]:
    """Return (filename, text) with per-file char capping."""
    name = upload.name
    suffix = (name.split(".")[-1] or "").lower()
    if suffix == "pdf":
        text = _read_pdf(upload)
    elif suffix == "docx":
        text = _read_docx(upload)
    elif suffix in ("txt", "md", "csv"):
        text = _read_text(upload)
    else:
        text = _read_text(upload)
    text = (text or "").strip()
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n\n[...truncated...]"
    return name, text

def _build_context_from_docs(docs: Dict[str, str]) -> str:
    if not docs:
        return ""
    parts, remaining = [], MAX_CONTEXT_CHARS
    for fname, body in docs.items():
        if remaining <= 0:
            break
        chunk = body[:min(len(body), remaining)]
        parts.append(f"# File: {fname}\n{chunk}")
        remaining -= len(chunk)
    return "\n\n".join(parts)

def _call_mistral(messages: List[Dict[str, str]], api_key: str) -> str:
    if not _HAVE_REQUESTS:
        return "⚠️ Missing dependency: install with `pip install requests`."
    if not api_key:
        return ("⚠️ Missing API key. Add `MISTRAL_API_KEY` to `.streamlit/secrets.toml` "
                "or set the environment variable.")

    model = st.secrets.get("MISTRAL_MODEL", os.getenv("MISTRAL_MODEL", MODEL_NAME))
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    max_retries = 4
    base_delay = 1.5

    for attempt in range(max_retries + 1):
        try:
            r = _requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                    continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries:
                return f"⚠️ API error: {e}"
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

    return "⚠️ API error: exhausted retries."

def _init_state():
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_docs" not in st.session_state:
        st.session_state.agent_docs = {}
    if "agent_query_input" not in st.session_state:
        st.session_state.agent_query_input = ""

# ---------- Search helpers ----------
def _find_matches(text: str, pattern: re.Pattern, window: int = 80) -> List[str]:
    """
    Build short HTML snippets around matches.
    Fix: collapse PDF line breaks and odd spacing before highlighting,
    and render with CSS that does NOT preserve hard newlines.
    """
    snippets = []
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        s0 = max(0, start - window)
        e0 = min(len(text), end + window)

        # Raw slice then normalize whitespace (PDFs often insert \n after each word)
        snippet_raw = text[s0:e0]
        snippet = re.sub(r"\s+", " ", snippet_raw).strip()

        # Re-run pattern on the normalized snippet for correct highlight offsets
        out, last = [], 0
        for mm in pattern.finditer(snippet):
            a, b = mm.start(), mm.end()
            out.append(escape(snippet[last:a]))
            out.append(f"<mark>{escape(snippet[a:b])}</mark>")
            last = b
        out.append(escape(snippet[last:]))

        prefix = "…" if s0 > 0 else ""
        suffix = "…" if e0 < len(text) else ""

        # Render with normal white-space (wrap long words if needed)
        snippets.append(
            f'<div class="snippet" style="white-space: normal; word-break: break-word; overflow-wrap: anywhere;">'
            f'{prefix}{"".join(out)}{suffix}</div>'
        )
    return snippets

def _search_keywords(query: str, docs: Dict[str, str], file_filter: str) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    if not query.strip():
        return results
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    items = docs.items()
    if file_filter and file_filter != "All files":
        items = [(file_filter, docs.get(file_filter, ""))]
    for fname, body in items:
        if not body:
            continue
        snippets = _find_matches(body, pattern, window=80)
        if snippets:
            results[fname] = snippets[:10]
    return results

def _ai_assist_summary(query: str, results: Dict[str, List[str]], api_key: str) -> str:
    if not results:
        return ""
    flat = []
    for f, ss in results.items():
        for s in ss[:3]:
            text_only = re.sub("<.*?>", "", s)
            text_only = re.sub(r"\s+", " ", text_only).strip()  # normalize for clean summary
            flat.append((f, text_only))
    flat = flat[:5]
    snippet_block = "\n".join([f"- [{f}] {t}" for f, t in flat])
    messages = [
        {"role": "system", "content": "You summarize only using the provided snippets. If unsure, say you cannot find it."},
        {"role": "user", "content": f"Query: {query}\n\nSnippets:\n{snippet_block}\n\nAnswer succinctly and cite filenames in brackets."},
    ]
    return _call_mistral(messages, api_key)

# ---------------------------- UI ----------------------------
def render_agent():
    _init_state()
    api_key = _get_api_key()

    st.title("AI Agent")
    st.caption("Chat with context. Drag & drop documents to ground the answers.")

    left, right = st.columns([1.05, 1.25])

    # ----- LEFT: Ask the agent -----
    with left:
        st.subheader("Ask the agent")
        with st.form("ask_form", clear_on_submit=True):
            q = st.text_input("Ask the agent", value=st.session_state.agent_query_input, placeholder="Ask something about your documents…")
            sent = st.form_submit_button("Send", use_container_width=True)

    # ----- RIGHT: Docs + actions + search -----
    with right:
        uploads = st.file_uploader(
            "Drag & drop files here",
            type=["pdf", "docx", "txt", "csv", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads:
            for up in uploads:
                name, text = _parse_upload(up)
                st.session_state.agent_docs[name] = text
                with st.expander(f"📎 {name} (preview)"):
                    st.code(text[:2000] if text else "", language="markdown")
            st.success(f"Loaded {len(uploads)} file(s).")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.agent_messages = []
                st.rerun()
        with c2:
            if st.button("Clear docs", use_container_width=True):
                st.session_state.agent_docs = {}
                st.rerun()

        st.subheader("Search in docs")

        file_opts = ["All files"] + list(st.session_state.agent_docs.keys())

        # ✅ Removed radio button "Type"
        with st.form("search_form"):
            s_query = st.text_input("Query", placeholder="Enter your question…")
            s_file = st.selectbox("File filter", file_opts, index=0)
            s_submit = st.form_submit_button("Search", use_container_width=True)

        if s_submit and s_query.strip():
            results = _search_keywords(s_query, st.session_state.agent_docs, s_file)
            if not results:
                st.info("No matches found.")
            else:
                # ✅ Always use AI-assist
                summary = _ai_assist_summary(s_query, results, api_key)
                if summary:
                    st.markdown("**AI-assisted summary**")
                    st.markdown(
                        f'<div style="white-space: normal; word-break: break-word; overflow-wrap: anywhere;">{escape(summary)}</div>',
                        unsafe_allow_html=True,
                    )

                for fname, snippets in results.items():
                    st.markdown(f'<div class="search-file">{escape(fname)} <span class="hits">({len(snippets)} hit(s))</span></div>', unsafe_allow_html=True)
                    for snip in snippets:
                        st.markdown(snip, unsafe_allow_html=True)

    # ----- Handle Ask form submission -----
    if 'sent' in locals() and sent and q.strip():
        user_prompt = q.strip()
        st.session_state.agent_query_input = ""

        st.session_state.agent_messages.append({"role": "user", "content": user_prompt})

        context_block = _build_context_from_docs(st.session_state.agent_docs)
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_block:
            messages.append({
                "role": "system",
                "content": f"Document context (may be relevant):\n\n{context_block}\n\nUse it only when pertinent.",
            })
        history = st.session_state.agent_messages[-(2 * HISTORY_TURNS):]
        messages.extend(history)

        with st.spinner("Thinking..."):
            answer = _call_mistral(messages, api_key=api_key)

        st.session_state.agent_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # ----- Transcript -----
    with left:
        st.subheader("Conversation")
        for msg in st.session_state.agent_messages[-2 * HISTORY_TURNS:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
   

# Public render function expected by app.py
def render():
    render_agent()

# Also allow running as a standalone Streamlit page
if __name__ == "__main__":
    render_agent()
