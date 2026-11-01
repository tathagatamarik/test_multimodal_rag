"""Streamlit front end for the RAG API.

    streamlit run services/ui.py            # expects the API at RAG_API_URL (default http://localhost:8010)

The UI only talks to the HTTP API, so it can run on another machine and never
loads a model itself.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("RAG_API_URL", "http://localhost:8010").rstrip("/")
GEN_TIMEOUT_S = 900  # CPU generation can take minutes

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1200px;}
  h1 {font-weight: 600; letter-spacing: -0.01em;}
  .evidence {border: 1px solid rgba(128,128,128,.25); border-radius: 6px; padding: .6rem .8rem;
             margin-bottom: .5rem; font-size: .9rem;}
  .tag {font-family: ui-monospace, monospace; font-weight: 600;}
  .muted {color: rgba(128,128,128,.95); font-size: .85rem;}
</style>""", unsafe_allow_html=True)


def api(method: str, path: str, timeout: float = 30, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=timeout, **kw)
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API}: {exc}")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        st.error(f"Request failed ({r.status_code}): {detail}")
        st.stop()
    return r


def visual_url(visual_id: str) -> str:
    return f"{API}/visuals/{visual_id}"


st.title("Multimodal RAG")
st.caption("Question answering over indexed research papers. Retrieval: Jina CLIP v1 and ChromaDB. "
           "Generation: Phi-3-Vision.")

tab_ask, tab_retrieve, tab_docs, tab_system = st.tabs(["Ask", "Retrieval explorer", "Documents", "System"])

# ------------------------------------------------------------------- Ask
with tab_ask:
    q = st.text_area("Question", placeholder="e.g. Which optimizer and learning rate schedule were used?",
                     height=80)
    c1, c2, c3 = st.columns([1, 1, 2])
    mode = c1.selectbox("Retrieval mode", ["multimodal", "text", "image"])
    routing = c2.checkbox("Structure routing", value=False,
                          help="Route through the document's table of contents before chunk retrieval.")
    c3.markdown('<p class="muted">Answers are generated on the server; on CPU this takes 1-4 minutes.</p>',
                unsafe_allow_html=True)
    # Buttons stay enabled and validate on click: Streamlit commits a text box's value only
    # on Enter/blur, so a disabled-while-empty button looks broken right after typing.
    ask_clicked = st.button("Answer", type="primary")
    if ask_clicked and not q.strip():
        st.warning("Enter a question first.")
    elif ask_clicked:
        with st.spinner("Retrieving evidence and generating the answer"):
            resp = api("POST", "/query", timeout=GEN_TIMEOUT_S,
                       json={"question": q, "mode": mode, "structure_routing": routing}).json()
        st.subheader("Answer")
        if resp["insufficient_evidence"]:
            st.warning(resp["answer"])
        else:
            st.write(resp["answer"])
        m = st.columns(5)
        m[0].metric("Total latency", f"{resp['timings']['total']:.1f} s")
        m[1].metric("Generation", f"{resp['timings'].get('generation_generate', 0):.1f} s")
        m[2].metric("Input tokens", resp["tokens"].get("input", 0))
        m[3].metric("Output tokens", resp["tokens"].get("output", 0))
        m[4].metric("Citations", f"{len(resp['citations'])} valid / {len(resp['invalid_citations'])} invalid")
        st.subheader("Evidence")
        cited = {c["tag"] for c in resp["citations"]}
        for e in resp["evidence"]:
            mark = "cited" if e["tag"] in cited else "not cited"
            st.markdown(f'<div class="evidence"><span class="tag">[{e["tag"]}]</span> '
                        f'{e["modality"]} &middot; {e["label"]} &middot; {e["page_ref"]} &middot; '
                        f'similarity {e["similarity"]:.3f} &middot; <span class="muted">{mark}</span><br>'
                        f'{e["snippet"]}</div>', unsafe_allow_html=True)
            if e["modality"] == "image" and e["source_ids"]:
                st.image(visual_url(e["source_ids"][0]), width=560)
        with st.expander("Raw response"):
            st.json(resp)

# ------------------------------------------------------------------- Retrieval
with tab_retrieve:
    rq = st.text_input("Query", key="rq", placeholder="e.g. scaled dot-product attention diagram")
    c1, c2 = st.columns(2)
    rmode = c1.selectbox("Mode", ["multimodal", "text", "image"], key="rmode")
    rrouting = c2.checkbox("Structure routing", key="rrouting")
    retrieve_clicked = st.button("Retrieve")
    if retrieve_clicked and not rq.strip():
        st.warning("Enter a query first.")
    elif retrieve_clicked:
        res = api("POST", "/retrieve", json={"question": rq, "mode": rmode, "generate": False,
                                             "structure_routing": rrouting}).json()
        st.markdown(f'<p class="muted">Query embedding {res["timings"].get("query_embedding", 0):.3f} s, '
                    f'vector search {res["timings"].get("vector_search", 0):.3f} s</p>', unsafe_allow_html=True)
        if res["sections"]:
            st.markdown("**Routed sections**")
            st.table([{"rank": s["rank"], "section": s["label"], "similarity": s["similarity"]}
                      for s in res["sections"]])
        st.markdown("**Text chunks**")
        st.table([{"rank": t["rank"], "similarity": t["similarity"], "page": t["page"],
                   "section": t["label"], "document": t["source"],
                   # the "[section path] " prefix is already shown in the section column
                   "text": t["content"].split("] ", 1)[-1][:220]}
                  for t in res["text"]])
        if res["images"] or res["rejected_images"]:
            st.markdown("**Visuals** (fused score = mean of pixel and caption similarity)")
            cols = st.columns(max(1, len(res["images"])))
            for col, im in zip(cols, res["images"]):
                col.image(visual_url(im["visual_id"]), caption=f'{im["label"]} (p.{im["page"]}) '
                          f'score {im["similarity"]:.3f}')
            for im in res["rejected_images"]:
                st.markdown(f'<p class="muted">Below threshold: {im["label"]} score {im["similarity"]:.3f}</p>',
                            unsafe_allow_html=True)

# ------------------------------------------------------------------- Documents
with tab_docs:
    status = api("GET", "/index/status").json()
    c = st.columns(3)
    c[0].metric("Indexed documents", status["indexed_documents"])
    c[1].metric("Records in collection", status["collection_records"])
    c[2].metric("Index fingerprint", status["fingerprint_hash"])
    docs = status["documents"]
    if docs:
        st.table([{"doc_id": d["doc_id"], "file": d["filename"], "status": d["status"], "pages": d["page_count"],
                   "chunks": d["text_chunks"], "sections": d["section_records"], "visuals": d["visual_records"],
                   "seconds": d["timings"].get("total"), "updated": d["updated_at"]} for d in docs])
    st.subheader("Add documents")
    up = st.file_uploader("Upload a PDF (processed by the background ingestion worker)", type=["pdf"])
    if up is not None and st.button("Ingest upload"):
        job = api("POST", "/ingest/upload", files={"file": (up.name, up.getvalue(), "application/pdf")}).json()
        st.success(f"Queued job {job['job_id']}")
    d1, d2 = st.columns([3, 1])
    directory = d1.text_input("Server directory to synchronise (incremental)")
    prune = d2.checkbox("Remove missing files")
    sync_clicked = st.button("Synchronise directory")
    if sync_clicked and not directory.strip():
        st.warning("Enter a server directory first.")
    elif sync_clicked:
        job = api("POST", "/ingest/batch", json={"directory": directory, "prune": prune}).json()
        st.success(f"Queued job {job['job_id']}")
    st.subheader("Ingestion jobs")
    jobs = api("GET", "/ingest/jobs").json()
    if jobs:
        st.table([{"job": j["job_id"], "kind": j["kind"], "status": j["status"], "created": j["created_at"],
                   "finished": j["finished_at"], "error": j["error"][:120]} for j in jobs[:20]])
    else:
        st.markdown('<p class="muted">No jobs yet.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------- System
with tab_system:
    h = api("GET", "/health").json()
    c = st.columns(4)
    c[0].metric("API status", h["status"])
    c[1].metric("Embedding model", "loaded" if h["embedding_model_loaded"] else "not loaded")
    c[2].metric("Generation model", "busy" if h["generation_busy"] else
                ("loaded" if h["generation_model_loaded"] else "not loaded"))
    c[3].metric("Ingestion queue", h["ingestion_queue"])
    s = api("GET", "/metrics/summary").json()["live"]
    if s.get("n"):
        c = st.columns(4)
        c[0].metric("Queries", s["n"])
        c[1].metric("Latency p50 / p95", f'{s["latency_total_p50"]:.0f} / {s["latency_total_p95"]:.0f} s')
        c[2].metric("Tokens in / out (total)", f'{s["input_tokens_total"]} / {s["output_tokens_total"]}')
        c[3].metric("Abstention rate", f'{s["abstention_rate"]:.0%}')
    st.markdown('<p class="muted">Detailed monitoring and drift: <code>streamlit run services/dashboard.py</code>; '
                'experiment history: <code>mlflow ui</code>.</p>', unsafe_allow_html=True)
