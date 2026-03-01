# mcp_streamlit_client.py
"""
MCP Streamlit Client for Document Processing Pipeline
Connects to the MCP server and provides a user-friendly interface for all available tools.
"""

import streamlit as st
import json
import subprocess
import sys
import time
import threading
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="MCP Document Processing Client",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    .stButton button:hover { background-color: #0056b3; }
    .success-box  { padding:1rem; border-radius:4px; background:#d4edda; border:1px solid #c3e6cb; color:#155724; }
    .error-box    { padding:1rem; border-radius:4px; background:#f8d7da; border:1px solid #f5c6cb; color:#721c24; }
    .info-box     { padding:1rem; border-radius:4px; background:#d1ecf1; border:1px solid #bee5eb; color:#0c5460; }
    .document-card {
        background:white; border-radius:8px; padding:1rem;
        box-shadow:0 2px 4px rgba(0,0,0,.1); margin:.5rem 0;
        border-left:4px solid #007bff;
    }
    /* ── Research result cards ── */
    .research-summary-box {
        background:#f0f7ff; border-radius:8px; padding:1.2rem;
        border-left:4px solid #007bff; margin-bottom:1rem;
    }
    .research-finding {
        display:flex; align-items:flex-start; gap:.5rem;
        padding:.4rem 0; border-bottom:1px solid #e9ecef;
    }
    .research-finding:last-child { border-bottom:none; }
    .finding-bullet {
        background:#007bff; color:white; border-radius:50%;
        width:22px; height:22px; min-width:22px;
        display:flex; align-items:center; justify-content:center;
        font-size:.7rem; font-weight:700;
    }
    .research-section-card {
        background:white; border-radius:8px; padding:1rem;
        box-shadow:0 1px 4px rgba(0,0,0,.08); margin:.5rem 0;
        border-top:3px solid #28a745;
    }
    .research-conclusion-box {
        background:#fff8e1; border-radius:8px; padding:1rem;
        border-left:4px solid #ffc107; margin-top:1rem;
    }
    .source-chip {
        display:inline-block; padding:.2rem .6rem; border-radius:12px;
        font-size:.75rem; font-weight:600; margin:.1rem; color:white;
    }
    .status-badge { display:inline-block; padding:.25rem .5rem; border-radius:4px; font-size:.8rem; font-weight:500; }
    .status-ready       { background:#d4edda; color:#155724; }
    .status-processing  { background:#fff3cd; color:#856404; }
    .status-vectorized  { background:#cce5ff; color:#004085; }
    .status-uploaded    { background:#e2e3e5; color:#383d41; }
    /* ── Meeting tool cards ── */
    .meeting-card {
        background:white; border-radius:8px; padding:1rem;
        box-shadow:0 2px 4px rgba(0,0,0,.1); margin:.5rem 0;
        border-left:4px solid #6f42c1;
    }
    .task-card {
        background:#f8f4ff; border-radius:6px; padding:.8rem;
        margin:.4rem 0; border-left:3px solid #6f42c1;
    }
    .priority-high   { color:#dc3545; font-weight:700; }
    .priority-medium { color:#fd7e14; font-weight:600; }
    .priority-low    { color:#28a745; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ──────────────────────────────────────────────────
for key, default in [
    ("mcp_connection",  None),
    ("current_document", None),
    ("chat_history",    []),
    ("documents_cache", []),
    ("last_refresh",    time.time()),
    ("research_history", []),
    ("meetings_cache",   []),
    ("meeting_tasks_cache", {}),
    ("email_list_cache", []),          # added for email
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================================
# MCP Client
# ============================================================================

class MCPStreamlitClient:
    def __init__(self, python_path=None, script_path=None, cwd=None):
        self.python_path = python_path or sys.executable
        self.script_path = script_path or str(Path.cwd() / "mcp_server.py")
        self.cwd         = cwd or str(Path.cwd())
        self.process     = None
        self.request_id  = 1
        self.connected   = False
        self.stderr_lines = []
        self._stop_stderr = False

    def _read_stderr(self):
        for line in self.process.stderr:
            if self._stop_stderr:
                break
            self.stderr_lines.append(line.rstrip())

    def connect(self):
        try:
            self.process = subprocess.Popen(
                [self.python_path, self.script_path],
                cwd=self.cwd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            t = threading.Thread(target=self._read_stderr, daemon=True)
            t.start()

            resp = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "streamlit-client", "version": "1.0.0"},
            })
            if "result" in resp:
                self._send_notification("notifications/initialized")
                self.connected = True
                return True, "Connected successfully"
            return False, f"Init failed: {resp.get('error', {}).get('message', 'Unknown')}"
        except Exception as e:
            stderr = "\n".join(self.stderr_lines)
            return False, f"Connection failed: {e}" + (f"\n\nstderr:\n{stderr}" if stderr else "")

    def disconnect(self):
        self._stop_stderr = True
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.connected = False

    def _send_request(self, method, params=None):
        if not self.process:
            raise Exception("Not connected")
        req = {"jsonrpc": "2.0", "id": self.request_id, "method": method}
        if params is not None:
            req["params"] = params
        self.request_id += 1
        self.process.stdin.write(json.dumps(req) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise Exception("No response from server")
        return json.loads(line)

    def _send_notification(self, method, params=None):
        if not self.process:
            raise Exception("Not connected")
        n = {"jsonrpc": "2.0", "method": method}
        if params:
            n["params"] = params
        self.process.stdin.write(json.dumps(n) + "\n")
        self.process.stdin.flush()

    def call_tool(self, name, arguments=None):
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        resp = self._send_request("tools/call", params)
        if "error" in resp:
            raise Exception(resp["error"].get("message", "Unknown error"))
        content = resp.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            text = content[0].get("text", "{}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                try:
                    return json.loads(json.loads(text))
                except Exception:
                    return text
        return resp.get("result", {})

    def list_tools(self):
        resp = self._send_request("tools/list")
        if "result" in resp:
            return resp["result"].get("tools", [])
        raise Exception(resp.get("error", {}).get("message", "Unknown"))

    # ── Document convenience wrappers ────────────────────────────────────────
    def list_documents(self):          return self.call_tool("list_documents")
    def get_document_info(self, d):    return self.call_tool("get_document_info",  {"document_id": d})
    def upload_document(self, p):      return self.call_tool("upload_document",    {"file_path": p})
    def vectorize_document(self, d):   return self.call_tool("vectorize_document", {"document_id": d})
    def get_graph_stats(self, d):      return self.call_tool("get_graph_stats",    {"document_id": d})
    def get_document_markdown(self, d):return self.call_tool("get_document_markdown", {"document_id": d})

    def query_document(self, doc_id, query, include_chunks=True):
        return self.call_tool("query_document", {
            "document_id": doc_id, "query": query, "include_chunks": include_chunks,
        })

    def summarize_page(self, doc_id, page_number):
        return self.call_tool("summarize_page", {
            "document_id": doc_id, "page_number": page_number,
        })

    # ── Web research wrapper ─────────────────────────────────────────────────
    def web_research(self, query: str, sources: str = "all", max_results: int = 7):
        result = self.call_tool("web_research", {
            "query": query,
            "sources": sources,
            "max_results_per_source": max_results,
        })
        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                return {"error": result}
        return result

    # ── Meeting tool wrappers (updated to handle string errors) ──────────────
    def list_meetings(self):
        result = self.call_tool("list_meetings")
        if isinstance(result, str):
            try:
                return json.loads(result)
            except:
                return result
        return result

    def get_meeting_tasks(self, meeting_name: str):
        return self.call_tool("get_meeting_tasks", {"meeting_name": meeting_name})

    def process_meeting(self, audio_file_path: str):
        return self.call_tool("process_meeting", {"audio_file_path": audio_file_path})

    def schedule_meeting_tasks(self, meeting_name: str,
                               credentials_path: str = "~/credentials/google_calendar.json",
                               preview_only: bool = False):
        return self.call_tool("schedule_meeting_tasks", {
            "meeting_name":     meeting_name,
            "credentials_path": credentials_path,
            "preview_only":     preview_only,
        })

    # ── Email tool wrappers ──────────────────────────────────────────────────
    def list_emails(self):
        return self.call_tool("list_emails")

    def fetch_emails_batch(self, max_results=50, label_ids="INBOX"):
        return self.call_tool("fetch_emails_batch", {
            "max_results": max_results,
            "label_ids": label_ids,
        })

    def rebuild_email_collection(self):
        return self.call_tool("rebuild_email_collection")

    def get_email_info(self, email_id):
        return self.call_tool("get_email_info", {"email_id": email_id})

    def get_email_markdown(self, email_id):
        return self.call_tool("get_email_markdown", {"email_id": email_id})

    def get_email_graph_stats(self, email_id):
        return self.call_tool("get_email_graph_stats", {"email_id": email_id})

    def query_email_collection(self, query, top_k=5):
        return self.call_tool("query_email_collection", {
            "query": query,
            "top_k": top_k,
        })


# ============================================================================
# Sidebar – connection
# ============================================================================

def render_connection_panel():
    st.sidebar.title("🔌 MCP Connection")
    with st.sidebar.expander("Connection Settings", expanded=True):
        python_path = st.text_input("Python Path",      value=sys.executable)
        script_path = st.text_input("MCP Server Script",value=str(Path.cwd() / "mcp_server.py"))
        cwd         = st.text_input("Working Directory", value=str(Path.cwd()))

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("🔌 Connect", use_container_width=True):
            with st.spinner("Connecting…"):
                client = MCPStreamlitClient(python_path, script_path, cwd)
                ok, msg = client.connect()
                if ok:
                    st.session_state.mcp_connection = client
                    st.success(msg); st.rerun()
                else:
                    st.error(msg)
    with c2:
        if st.button("🔌 Disconnect", use_container_width=True,
                     disabled=not st.session_state.mcp_connection):
            st.session_state.mcp_connection.disconnect()
            st.session_state.mcp_connection = None
            st.success("Disconnected"); st.rerun()

    if st.session_state.mcp_connection and st.session_state.mcp_connection.connected:
        st.sidebar.markdown('<div class="success-box">✅ Connected to MCP server</div>',
                            unsafe_allow_html=True)
        try:
            tools = st.session_state.mcp_connection.list_tools()
            with st.sidebar.expander("Available Tools", expanded=False):
                for t in tools:
                    st.markdown(f"**{t.get('name')}**")
                    st.caption(t.get("description", ""))
        except Exception as e:
            st.sidebar.warning(f"Could not list tools: {e}")
    else:
        st.sidebar.markdown('<div class="info-box">⚠️ Not connected</div>',
                            unsafe_allow_html=True)


# ============================================================================
# Documents panel
# ============================================================================

def render_documents_panel():
    st.header("📚 Documents")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to view documents"); return

    c1, _, c3 = st.columns([1, 3, 2])
    with c1:
        if st.button("🔄 Refresh", use_container_width=True):
            with st.spinner("Refreshing…"):
                try:
                    st.session_state.documents_cache = st.session_state.mcp_connection.list_documents()
                    st.session_state.last_refresh = time.time()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to refresh: {e}")
    with c3:
        st.caption(f"Last refresh: {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S')}")

    if not st.session_state.documents_cache:
        try:
            with st.spinner("Loading…"):
                st.session_state.documents_cache = st.session_state.mcp_connection.list_documents()
        except Exception as e:
            st.error(f"Failed: {e}"); return

    docs = st.session_state.documents_cache
    if not docs:
        st.info("No documents found."); return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",          len(docs))
    c2.metric("Ready for Chat", sum(1 for d in docs if d.get("status") == "ready"))
    c3.metric("Processing",     sum(1 for d in docs if d.get("status") == "processing"))
    c4.metric("Vectorized",     sum(1 for d in docs if d.get("status") == "vectorized"))

    for doc in docs:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
            with c1:
                st.markdown(f"""
                <div class="document-card">
                    <strong>{doc.get('name', doc.get('document_id','Unknown'))}</strong><br>
                    <span style="font-size:.8rem;color:#666;">ID: {doc.get('document_id','N/A')}</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                status = doc.get("status", "unknown")
                st.markdown(f'<span class="status-badge status-{status}">{status}</span>',
                            unsafe_allow_html=True)
            with c3:
                if st.button("📊", key=f"stats_{doc['document_id']}", use_container_width=True):
                    st.session_state.current_document = doc["document_id"]; st.rerun()
            with c4:
                if status == "processing":
                    if st.button("⚙️ Vectorize", key=f"vec_{doc['document_id']}", use_container_width=True):
                        with st.spinner("Vectorizing…"):
                            try:
                                r = st.session_state.mcp_connection.vectorize_document(doc["document_id"])
                                if "error" not in r:
                                    st.success("Done!")
                                    st.session_state.documents_cache = st.session_state.mcp_connection.list_documents()
                                    st.rerun()
                                else:
                                    st.error(r.get("error"))
                            except Exception as e:
                                st.error(str(e))
                else:
                    st.button("⚙️ Vectorize", key=f"vec_{doc['document_id']}", disabled=True, use_container_width=True)
            with c5:
                if status == "ready":
                    if st.button("💬 Chat", key=f"chat_{doc['document_id']}", use_container_width=True):
                        st.session_state.current_document = doc["document_id"]; st.rerun()
                else:
                    st.button("💬 Chat", key=f"chat_{doc['document_id']}", disabled=True, use_container_width=True)


# ============================================================================
# Upload panel
# ============================================================================

def render_upload_panel():
    st.header("📤 Upload Document")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to upload documents"); return

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file:
        temp_dir = Path("temp_uploads"); temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"**File:** {uploaded_file.name}  |  **Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("📤 Upload", use_container_width=True):
                with st.spinner("Uploading…"):
                    try:
                        r = st.session_state.mcp_connection.upload_document(str(temp_path))
                        if "error" not in r:
                            st.success("✅ Uploaded!"); st.json(r)
                            st.session_state.documents_cache = []; time.sleep(1); st.rerun()
                        else:
                            st.error(r.get("error"))
                    except Exception as e:
                        st.error(str(e))
                    finally:
                        if temp_path.exists(): temp_path.unlink()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                if temp_path.exists(): temp_path.unlink()
                st.rerun()


# ============================================================================
# Chat panel
# ============================================================================

def render_chat_panel():
    st.header("💬 Document Chat")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to chat"); return

    docs      = st.session_state.documents_cache
    ready     = [d for d in docs if d.get("status") == "ready"]
    if not ready:
        st.info("No ready documents."); return

    options   = {f"{d.get('name', d['document_id'])} ({d['document_id'][:8]})": d["document_id"]
                 for d in ready}
    sel_name  = st.selectbox("Select Document", list(options), key="chat_doc_select")
    sel_id    = options[sel_name]
    st.session_state.current_document = sel_id

    info = next((d for d in ready if d["document_id"] == sel_id), None)
    if info:
        with st.expander("Document Info", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages",  info.get("total_pages",  "N/A"))
            c2.metric("Chunks", info.get("total_chunks", "N/A"))
            c3.metric("Status", info.get("status",       "N/A"))

    st.markdown("---")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if "stats" in msg:
                    with st.expander("Retrieval Statistics", expanded=False):
                        st.json(msg["stats"])

    if prompt := st.chat_input("Ask a question about the document…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            try:
                r = st.session_state.mcp_connection.query_document(sel_id, prompt, include_chunks=False)
                if "error" not in r:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": r.get("answer", "No answer"),
                        "stats":   r.get("retrieval_stats", {}),
                    })
                else:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"Error: {r.get('error')}"})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()


# ============================================================================
# Statistics panel
# ============================================================================

def render_statistics_panel():
    st.header("📊 Document Statistics")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to view statistics"); return

    docs = st.session_state.documents_cache
    vdocs = [d for d in docs if d.get("status") in ["vectorized", "ready"]]
    if not vdocs:
        st.info("No vectorized documents found."); return

    options = {f"{d.get('name', d['document_id'])} ({d['document_id'][:8]})": d["document_id"]
               for d in vdocs}
    default = 0
    if st.session_state.current_document:
        for i, did in enumerate(options.values()):
            if did == st.session_state.current_document:
                default = i; break

    sel_name = st.selectbox("Select Document", list(options), index=default, key="stats_doc_select")
    sel_id   = options[sel_name]

    if st.button("📊 Get Graph Statistics", use_container_width=True):
        with st.spinner("Loading…"):
            try:
                stats = st.session_state.mcp_connection.get_graph_stats(sel_id)
                if "error" not in stats:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Nodes",   stats.get("total_nodes", 0))
                    c2.metric("Edges",   stats.get("total_edges", 0))
                    c3.metric("Density", f"{stats.get('density', 0):.4f}")
                    if nt := stats.get("node_types", {}):
                        st.plotly_chart(px.pie(values=list(nt.values()), names=list(nt.keys()),
                                               title="Node Types"), use_container_width=True)
                    if er := stats.get("edge_relations", {}):
                        st.plotly_chart(px.bar(x=list(er.keys()), y=list(er.values()),
                                               title="Edge Relations"), use_container_width=True)
                    with st.expander("Raw JSON", expanded=False):
                        st.json(stats)
                else:
                    st.error(stats.get("error"))
            except Exception as e:
                st.error(str(e))


# ============================================================================
# Page Summary panel
# ============================================================================

def render_page_summary_panel():
    st.header("📄 Page Summary")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to generate summaries"); return

    docs  = st.session_state.documents_cache
    vdocs = [d for d in docs if d.get("status") in ["vectorized", "ready"]]
    if not vdocs:
        st.info("No vectorized documents found."); return

    options  = {f"{d.get('name', d['document_id'])} ({d['document_id'][:8]})": d["document_id"]
                for d in vdocs}
    sel_name = st.selectbox("Select Document", list(options), key="summary_doc_select")
    sel_id   = options[sel_name]
    info     = next((d for d in vdocs if d["document_id"] == sel_id), None)
    total_pages = info.get("total_pages", 0) if info else 0

    c1, c2 = st.columns([1, 2])
    with c1:
        page = st.number_input("Page Number", min_value=1, max_value=max(total_pages, 1), value=1)
    with c2:
        st.caption(f"Document has {total_pages} pages")

    if st.button("📄 Generate Summary", use_container_width=True):
        with st.spinner(f"Summarising page {page}…"):
            try:
                r = st.session_state.mcp_connection.summarize_page(sel_id, page)
                if "error" not in r:
                    st.success("✅ Summary generated!")
                    st.markdown("### Summary"); st.markdown(r.get("summary", ""))
                    if kp := r.get("key_points", []):
                        st.markdown("### Key Points")
                        for p in kp: st.markdown(f"• {p}")
                else:
                    st.error(r.get("error"))
            except Exception as e:
                st.error(str(e))


# ============================================================================
# Markdown panel
# ============================================================================

def render_markdown_panel():
    st.header("📝 Document Markdown")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to view markdown"); return

    docs = st.session_state.documents_cache
    pdocs = [d for d in docs if d.get("markdown_path")]
    if not pdocs:
        st.info("No processed documents found."); return

    options  = {f"{d.get('name', d['document_id'])} ({d['document_id'][:8]})": d["document_id"]
                for d in pdocs}
    sel_name = st.selectbox("Select Document", list(options), key="markdown_doc_select")
    sel_id   = options[sel_name]

    if st.button("📝 Load Markdown", use_container_width=True):
        with st.spinner("Loading…"):
            try:
                md = st.session_state.mcp_connection.get_document_markdown(sel_id)
                if "error" not in md:
                    st.markdown("### Document Content"); st.markdown(md)
                else:
                    st.error(md.get("error"))
            except Exception as e:
                st.error(str(e))


# ============================================================================
# Web Research panel
# ============================================================================

_SOURCE_COLORS = {
    "Tavily":     "#17a2b8",
    "Exa":        "#6f42c1",
    "Google":     "#dc3545",
    "DuckDuckGo": "#fd7e14",
    "Wikipedia":  "#28a745",
}

def _chip(source: str) -> str:
    color = _SOURCE_COLORS.get(source, "#6c757d")
    return (f'<span class="source-chip" style="background:{color};">{source}</span>')


def _render_research_result(item: dict, expanded: bool = True):
    """Render a single structured research result."""
    ts = item["timestamp"][:19].replace("T", " ")

    with st.expander(f"**{item['query'][:80]}**  —  {ts}", expanded=expanded):

        # ── Meta row ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Sources Used", item["num_sources"])
        c2.metric("Sections",     len(item.get("sections", [])))
        c3.metric("Key Findings", len(item.get("key_findings", [])))

        # Source chips
        if item.get("sources"):
            seen: set = set()
            chips = ""
            for _, _, src in item["sources"]:
                if src not in seen:
                    chips += _chip(src); seen.add(src)
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown("---")

        # ── Summary ───────────────────────────────────────────────────────
        if item.get("summary"):
            st.markdown(
                f'<div class="research-summary-box">'
                f'<strong>Overview</strong><br>{item["summary"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Key Findings ──────────────────────────────────────────────────
        if item.get("key_findings"):
            st.markdown("#### Key Findings")
            findings_html = ""
            for i, f in enumerate(item["key_findings"], 1):
                findings_html += (
                    f'<div class="research-finding">'
                    f'<div class="finding-bullet">{i}</div>'
                    f'<div>{f}</div></div>'
                )
            st.markdown(findings_html, unsafe_allow_html=True)
            st.markdown("")   # spacer

        # ── Deep-Dive Sections ────────────────────────────────────────────
        if item.get("sections"):
            st.markdown("#### Deep Dive")
            for sec in item["sections"]:
                heading = sec.get("heading", "")
                content = sec.get("content", "")
                st.markdown(
                    f'<div class="research-section-card">'
                    f'<strong>{heading}</strong><br>{content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Conclusion ────────────────────────────────────────────────────
        if item.get("conclusion"):
            st.markdown(
                f'<div class="research-conclusion-box">'
                f'<strong>Conclusion</strong><br>{item["conclusion"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Full Markdown (collapsible) ───────────────────────────────────
        if item.get("raw_markdown"):
            with st.expander("Full Markdown Answer", expanded=False):
                st.markdown(item["raw_markdown"])

        # ── Source table ──────────────────────────────────────────────────
        if item.get("sources"):
            st.markdown("#### Sources")
            df = pd.DataFrame(
                [{"#": i+1, "Title": t or "(no title)", "Engine": s, "URL": u}
                 for i, (t, u, s) in enumerate(item["sources"])],
            )
            st.dataframe(
                df,
                column_config={"URL": st.column_config.LinkColumn("URL")},
                use_container_width=True,
                hide_index=True,
            )

        # ── Non-fatal errors ──────────────────────────────────────────────
        if item.get("errors"):
            with st.expander("⚠️ Non-fatal errors", expanded=False):
                for src, msg in item["errors"].items():
                    st.warning(f"**{src}:** {msg}")

        # ── Download ──────────────────────────────────────────────────────
        md_export = item.get("raw_markdown") or item.get("summary", "")
        if md_export:
            st.download_button(
                "💾 Download as Markdown",
                data=md_export,
                file_name=f"research_{ts[:10]}.md",
                mime="text/markdown",
                key=f"dl_{item['timestamp']}",
            )


def render_web_research_panel():
    st.header("🌐 Web Research")
    st.caption("Search multiple engines simultaneously — results are parsed and structured by OpenAI.")

    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to use web research"); return

    # ── Query form ────────────────────────────────────────────────────────
    with st.form("research_form"):
        query = st.text_area(
            "Research Query",
            placeholder="e.g. What are the latest advancements in multimodal AI models?",
            height=90,
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_sources = st.multiselect(
                "Sources  (leave empty = all)",
                options=["tavily", "exa", "google", "duckduckgo", "wikipedia"],
                default=[],
            )
        with c2:
            max_results = st.slider("Results / source", 3, 10, 7)

        submitted = st.form_submit_button("🔍 Research", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Please enter a research query.")
        else:
            sources_str = ",".join(selected_sources) if selected_sources else "all"
            with st.spinner(f"Searching & analysing with OpenAI…"):
                try:
                    result = st.session_state.mcp_connection.web_research(
                        query=query.strip(),
                        sources=sources_str,
                        max_results=max_results,
                    )

                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except Exception:
                            st.error(f"Unparseable response: {result}")
                            st.stop()

                    if not isinstance(result, dict):
                        st.error(f"Unexpected type from server: {type(result)}")
                        st.stop()

                    if "error" in result:
                        st.error(f"Research failed: {result['error']}")
                    else:
                        st.session_state.research_history.insert(0, {
                            "query":        result.get("query",        query),
                            "summary":      result.get("summary",      ""),
                            "key_findings": result.get("key_findings", []),
                            "sections":     result.get("sections",     []),
                            "conclusion":   result.get("conclusion",   ""),
                            "raw_markdown": result.get("raw_markdown", ""),
                            "sources":      result.get("sources",      []),
                            "num_sources":  result.get("num_sources",  0),
                            "timestamp":    result.get("timestamp",    datetime.now().isoformat()),
                            "errors":       result.get("errors",       {}),
                        })
                        st.rerun()

                except Exception as e:
                    st.error(f"Error calling web_research tool: {e}")

    # ── History ───────────────────────────────────────────────────────────
    if not st.session_state.research_history:
        st.markdown("---")
        st.markdown(
            '<div class="info-box">💡 Enter a query above to start researching. '
            'Results are structured into <strong>summary → key findings → '
            'sections → conclusion</strong> by OpenAI.</div>',
            unsafe_allow_html=True,
        )
        return

    hc1, hc2 = st.columns([4, 1])
    hc1.markdown(f"**{len(st.session_state.research_history)} result(s) in history**")
    with hc2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.research_history = []
            st.rerun()

    for idx, item in enumerate(st.session_state.research_history):
        _render_research_result(item, expanded=(idx == 0))


# ============================================================================
# Meeting Tools Panel (with robust error handling)
# ============================================================================

def _priority_badge(priority: str) -> str:
    cls = {
        "high":   "priority-high",
        "medium": "priority-medium",
        "low":    "priority-low",
    }.get(str(priority).lower(), "priority-medium")
    return f'<span class="{cls}">▲ {priority.upper()}</span>'


def _render_task_card(task: dict, idx: int):
    desc     = task.get("description", "No description")
    assignee = task.get("assignee", "—")
    due      = task.get("due_date",  "—")
    location = task.get("location",  "")
    priority = task.get("priority",  "medium")
    status   = task.get("status",    "pending")

    st.markdown(
        f"""
        <div class="task-card">
          <strong>#{idx} — {desc}</strong><br>
          👤 <em>{assignee}</em> &nbsp;|&nbsp;
          📅 {due} &nbsp;|&nbsp;
          {_priority_badge(priority)} &nbsp;|&nbsp;
          🔖 {status}
          {"<br>📍 " + location if location else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meeting_tools_panel():
    st.header("🎙️ Meeting Tools")
    st.caption(
        "Process audio recordings to extract tasks & knowledge graphs, "
        "then schedule them directly into Google Calendar."
    )

    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to use meeting tools")
        return

    client: MCPStreamlitClient = st.session_state.mcp_connection

    # ── Tabs inside the panel ─────────────────────────────────────────────
    mtab1, mtab2, mtab3 = st.tabs(["🎤 Process Audio", "📋 View Tasks", "📅 Schedule"])

    # ====================================================================
    # Tab 1 – Process Audio
    # ====================================================================
    with mtab1:
        st.subheader("Transcribe & Extract Tasks from a Meeting Recording")

        with st.form("meeting_process_form"):
            audio_path = st.text_input(
                "Audio file path",
                placeholder="/absolute/path/to/meeting.mp3  or  relative/path.wav",
                help="Supported formats: mp3, mp4, wav, m4a, ogg, flac",
            )
            submitted = st.form_submit_button("🚀 Process Meeting", use_container_width=True)

        if submitted:
            if not audio_path.strip():
                st.warning("Please enter a file path.")
            else:
                with st.spinner("Transcribing and extracting knowledge… (this may take a few minutes)"):
                    try:
                        result = client.process_meeting(audio_path.strip())
                        if isinstance(result, str):
                            result = json.loads(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        result = None

                if result:
                    if isinstance(result, dict) and "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"✅ Meeting **{result.get('meeting_name')}** processed!")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Tasks Found",   result.get("tasks", 0))
                        col2.metric("Entities",      result.get("entities", 0))
                        col3.metric("Graph Nodes",   result.get("graph_nodes", 0))
                        col4.metric("Transcript len",result.get("transcript_length", 0))
                        st.info(f"📁 Output saved to: `{result.get('output_dir')}`")
                        # Refresh meetings cache
                        try:
                            st.session_state.meetings_cache = client.list_meetings()
                        except Exception:
                            pass
                        st.rerun()

        st.markdown("---")
        st.markdown(
            '<div class="info-box">💡 <strong>Requirements:</strong> '
            '<code>faster-whisper</code>, <code>networkx</code>, '
            '<code>langchain-openai</code> and an <code>OPENAI_API_KEY</code> '
            'in your <code>.env</code> for LLM-powered extraction.</div>',
            unsafe_allow_html=True,
        )

    # ====================================================================
    # Tab 2 – View Tasks
    # ====================================================================
    with mtab2:
        st.subheader("Extracted Meeting Tasks")

        rc1, rc2 = st.columns([3, 1])
        with rc2:
            if st.button("🔄 Refresh Meetings", use_container_width=True):
                with st.spinner("Loading…"):
                    try:
                        st.session_state.meetings_cache = client.list_meetings()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # Auto-load on first visit
        if not st.session_state.meetings_cache:
            try:
                with st.spinner("Loading meetings…"):
                    st.session_state.meetings_cache = client.list_meetings()
            except Exception as e:
                st.error(f"Could not list meetings: {e}")

        meetings = st.session_state.meetings_cache
        # --- FIX: Check if meetings is a list of dicts, not a string error ---
        if isinstance(meetings, str):
            st.error(f"Error from server: {meetings}")
            return
        if not isinstance(meetings, list):
            st.error(f"Unexpected meetings data type: {type(meetings)}")
            return
        # --------------------------------------------------------------------
        if not meetings:
            st.info("No processed meetings found. Use the **Process Audio** tab first.")
        else:
            meeting_names = [m["meeting_name"] for m in meetings if isinstance(m, dict)]
            if not meeting_names:
                st.error("No valid meeting names found in response.")
                return
            selected = st.selectbox(
                "Select meeting",
                options=meeting_names,
                format_func=lambda n: f"{n}  ({next((m['tasks_count'] for m in meetings if m.get('meeting_name')==n), 0)} tasks)",
            )

            if selected:
                if st.button("📥 Load Tasks", use_container_width=False):
                    with st.spinner("Loading tasks…"):
                        try:
                            tasks = client.get_meeting_tasks(selected)
                            if isinstance(tasks, str):
                                tasks = json.loads(tasks)
                            st.session_state.meeting_tasks_cache[selected] = tasks
                        except Exception as e:
                            st.error(f"Failed: {e}")

                tasks_data = st.session_state.meeting_tasks_cache.get(selected)
                if tasks_data is not None:
                    if isinstance(tasks_data, dict) and "error" in tasks_data:
                        st.error(tasks_data["error"])
                    elif not tasks_data:
                        st.info("No tasks extracted for this meeting.")
                    else:
                        tasks_list = tasks_data if isinstance(tasks_data, list) else [tasks_data]
                        st.markdown(f"**{len(tasks_list)} task(s) found**")
                        for i, task in enumerate(tasks_list, 1):
                            _render_task_card(task, i)

                        # Download as JSON
                        st.download_button(
                            "💾 Download Tasks JSON",
                            data=json.dumps(tasks_list, indent=2),
                            file_name=f"{selected}_tasks.json",
                            mime="application/json",
                        )

    # ====================================================================
    # Tab 3 – Schedule
    # ====================================================================
    with mtab3:
        st.subheader("Schedule Tasks to Google Calendar")

        if not st.session_state.meetings_cache:
            try:
                st.session_state.meetings_cache = client.list_meetings()
            except Exception:
                pass

        meetings = st.session_state.meetings_cache
        if isinstance(meetings, str):
            st.error(f"Error from server: {meetings}")
            return
        if not isinstance(meetings, list):
            st.error(f"Unexpected meetings data type: {type(meetings)}")
            return
        if not meetings:
            st.info("No processed meetings found. Use the **Process Audio** tab first.")
        else:
            with st.form("schedule_form"):
                meeting_names = [m["meeting_name"] for m in meetings if isinstance(m, dict)]
                if not meeting_names:
                    st.error("No valid meeting names found in response.")
                    return
                sched_meeting = st.selectbox("Meeting", options=meeting_names)
                cred_path = st.text_input(
                    "Google Calendar credentials path",
                    value="~/credentials/google_calendar.json",
                    help="OAuth 2.0 credentials JSON from Google Cloud Console",
                )
                preview_only = st.checkbox(
                    "Preview only (don't create real calendar events)",
                    value=True,
                )
                sched_submitted = st.form_submit_button("📅 Schedule / Preview", use_container_width=True)

            if sched_submitted:
                with st.spinner("Scheduling tasks…"):
                    try:
                        result = client.schedule_meeting_tasks(
                            meeting_name=sched_meeting,
                            credentials_path=cred_path.strip(),
                            preview_only=preview_only,
                        )
                        if isinstance(result, str):
                            result = json.loads(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        result = None

                if result:
                    if isinstance(result, dict) and "error" in result:
                        st.error(f"Scheduling failed: {result['error']}")
                    elif result.get("mode") == "preview":
                        st.info(result.get("message", "Preview mode"))
                        for ev in result.get("events", []):
                            st.markdown(
                                f"""
                                <div class="task-card">
                                  📌 <strong>{ev.get('summary')}</strong><br>
                                  ▶ Start: {ev.get('start', '—')}<br>
                                  ■ End:&nbsp;&nbsp; {ev.get('end',   '—')}<br>
                                  {"📍 " + ev['location'] if ev.get('location') else ""}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        sc = result.get("scheduled", 0)
                        fa = result.get("failed",    0)
                        sk = result.get("skipped",   0)
                        st.success(f"✅ Scheduled {sc} event(s)  |  ❌ Failed: {fa}  |  ⏭️ Skipped: {sk}")

            st.markdown("---")
            st.markdown(
                '<div class="info-box">💡 <strong>Setup:</strong> '
                'Create OAuth credentials in Google Cloud Console, enable the '
                'Calendar API, download the JSON, and place it at the credentials path above. '
                'Install <code>phidata</code>, <code>tzlocal</code>, and <code>python-dateutil</code>.</div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# Email Panel
# ============================================================================

def render_email_panel():
    st.header("📧 Email Processing")
    if not st.session_state.mcp_connection:
        st.info("Connect to MCP server to use email tools")
        return

    client: MCPStreamlitClient = st.session_state.mcp_connection

    etab1, etab2, etab3, etab4 = st.tabs(["📥 Fetch", "📋 List", "🔄 Collection", "🔍 Query"])

    # ====================================================================
    # Tab 1 – Fetch Batch
    # ====================================================================
    with etab1:
        st.subheader("Fetch Emails from Gmail")
        st.caption("Requires `credentials.json` and `token.json` in the working directory.")
        with st.form("email_fetch_form"):
            max_results = st.number_input("Max results", min_value=1, max_value=200, value=50)
            label_ids = st.text_input("Label IDs (comma separated)", value="INBOX",
                                      help="e.g. INBOX,UNREAD,IMPORTANT")
            submitted = st.form_submit_button("🚀 Fetch Batch", use_container_width=True)
        if submitted:
            with st.spinner("Fetching and processing emails… (this may take a while)"):
                try:
                    result = client.fetch_emails_batch(max_results=int(max_results), label_ids=label_ids)
                    if isinstance(result, str):
                        result = json.loads(result)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Batch fetch completed. Check server logs for details.")
                        st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ====================================================================
    # Tab 2 – List Emails
    # ====================================================================
    with etab2:
        st.subheader("Processed Emails")
        if st.button("🔄 Refresh", use_container_width=False):
            with st.spinner("Loading…"):
                try:
                    emails = client.list_emails()
                    if isinstance(emails, str):
                        emails = json.loads(emails)
                    st.session_state["email_list_cache"] = emails
                except Exception as e:
                    st.error(f"Failed: {e}")

        if "email_list_cache" not in st.session_state:
            st.session_state["email_list_cache"] = []

        emails = st.session_state["email_list_cache"]
        if not emails:
            st.info("No emails processed yet. Use the Fetch tab to get emails.")
        else:
            for email in emails:
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    with c1:
                        st.markdown(f"**{email['email_id']}**")
                    with c2:
                        st.caption(email['status'])
                    with c3:
                        if st.button("📄 Info", key=f"info_{email['email_id']}"):
                            info = client.get_email_info(email['email_id'])
                            if isinstance(info, str):
                                info = json.loads(info)
                            st.json(info)
                    with c4:
                        if st.button("📊 Graph", key=f"graph_{email['email_id']}"):
                            stats = client.get_email_graph_stats(email['email_id'])
                            if isinstance(stats, str):
                                stats = json.loads(stats)
                            st.json(stats)

    # ====================================================================
    # Tab 3 – Collection
    # ====================================================================
    with etab3:
        st.subheader("Email Collection")
        st.markdown("Rebuild the merged collection (vector store + graph) from all processed emails.")
        if st.button("🔨 Rebuild Collection", use_container_width=True):
            with st.spinner("Rebuilding collection… (may take a few minutes)"):
                try:
                    result = client.rebuild_email_collection()
                    if isinstance(result, str):
                        result = json.loads(result)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Collection rebuilt!")
                        st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ====================================================================
    # Tab 4 – Query Collection
    # ====================================================================
    with etab4:
        st.subheader("Query Across All Emails")
        with st.form("email_query_form"):
            query = st.text_area("Your question", placeholder="e.g. What was discussed about project X?")
            top_k = st.slider("Top K chunks", 1, 20, 5)
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)
        if submitted:
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching…"):
                    try:
                        result = client.query_email_collection(query, top_k=top_k)
                        if isinstance(result, str):
                            result = json.loads(result)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"Found {result['count']} relevant chunks")
                            for i, r in enumerate(result.get("results", [])):
                                with st.expander(f"Chunk {i+1} – Email: {r['email_id']}"):
                                    st.markdown(r['content'])
                                    st.caption(f"Metadata: {r['metadata']}")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    st.title("📄 MCP Document Processing Client")
    st.caption("Streamlit client for the Document Processing Pipeline MCP server")

    render_connection_panel()

    if st.session_state.mcp_connection:
        tabs = st.tabs([
            "📚 Documents", "📤 Upload", "💬 Chat",
            "📊 Statistics", "📄 Page Summary", "📝 Markdown",
            "🌐 Web Research", "🎙️ Meeting Tools", "📧 Email",   # added Email tab
        ])
        with tabs[0]: render_documents_panel()
        with tabs[1]: render_upload_panel()
        with tabs[2]: render_chat_panel()
        with tabs[3]: render_statistics_panel()
        with tabs[4]: render_page_summary_panel()
        with tabs[5]: render_markdown_panel()
        with tabs[6]: render_web_research_panel()
        with tabs[7]: render_meeting_tools_panel()
        with tabs[8]: render_email_panel()                         # new tab

        # Auto-refresh every 30 s
        if time.time() - st.session_state.last_refresh > 30:
            try:
                st.session_state.documents_cache = st.session_state.mcp_connection.list_documents()
                st.session_state.last_refresh = time.time()
            except Exception:
                pass
    else:
        st.info("👈 Connect to MCP server using the sidebar to get started")
        with st.expander("Quick Start Guide", expanded=True):
            st.markdown("""
            ### How to use:
            1. **Connect** via the sidebar
            2. **Upload** a PDF and wait for it to become *ready*
            3. **Chat** with your document or explore statistics / summaries
            4. **🌐 Web Research** – research any topic; results are parsed by OpenAI into
               summary, key findings, deep-dive sections and a conclusion
            5. **🎙️ Meeting Tools** – upload an audio recording, extract tasks/knowledge graph,
               then schedule events directly to Google Calendar
            6. **📧 Email** – fetch emails from Gmail, list them, rebuild the collection,
               and query across all emails

            ### API keys needed (in `.env`):
            | Key | Used for |
            |-----|----------|
            | `OPENAI_API_KEY` | Structuring research results & meeting extraction |
            | `TAVILY_API_KEY` | Tavily search |
            | `EXA_API_KEY` | Exa search |
            | `SERPER_API_KEY` | Google search |
            | DuckDuckGo & Wikipedia | No key needed |

            ### Meeting tool dependencies:
            pip install faster-whisper networkx langchain langchain-openai
            pip install phidata python-dateutil tzlocal

            ### Email tool dependencies:
            pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
            pip install html2text langchain-community langgraph ollama

    """)


if __name__ == "__main__":
    main()