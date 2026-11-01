"""Monitoring dashboard: latency, token usage, retrieval quality, drift and evaluation history.

    streamlit run services/dashboard.py --server.port 8502

Reads the telemetry SQLite store and the MLflow tracking store directly, so it
works independently of the API process.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run services/dashboard.py` puts services/ (not the project root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.drift import detect_drift  # noqa: E402
from src.telemetry import TelemetryStore, summarize  # noqa: E402

st.set_page_config(page_title="RAG Monitoring", layout="wide")
st.markdown("""<style>.block-container {padding-top: 1.6rem;}
.status {display:inline-block; padding:.15rem .6rem; border-radius:4px; font-weight:600; font-size:.85rem;}
.stable {background:#e6f4ea; color:#1e6b34;} .moderate {background:#fff4e0; color:#8a5300;}
.significant {background:#fde8e8; color:#9b1c1c;} .insufficient_data {background:#eef0f3; color:#4a5160;}
</style>""", unsafe_allow_html=True)

cfg = load_config()
store = TelemetryStore(cfg.path(cfg.monitoring.telemetry_path))

st.title("RAG Monitoring")
c1, c2 = st.columns([1, 3])
window = c1.selectbox("Live window", ["Last 200 queries", "Last 24 hours", "Last 7 days", "All"])
since = {"Last 24 hours": time.time() - 86400, "Last 7 days": time.time() - 7 * 86400}.get(window)
limit = 200 if window == "Last 200 queries" else None

live_rows = store.rows(exclude_source="evaluation", since=since, limit=limit)
eval_rows = store.rows(source="evaluation")


def frame(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "embedding"} for r in rows])
    if not df.empty:
        local_tz = datetime.now().astimezone().tzinfo
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(local_tz).dt.tz_localize(None)
    return df


live, ref = frame(live_rows), frame(eval_rows)
s_live, s_ref = summarize(live_rows), summarize(eval_rows)

st.subheader("Service health")
if not s_live.get("n"):
    st.info("No live queries recorded yet. Ask questions through the API or CLI to populate this view.")
else:
    m = st.columns(6)
    m[0].metric("Queries", s_live["n"])
    m[1].metric("Latency p50", f'{s_live["latency_total_p50"]:.1f} s',
                delta=(f'{s_live["latency_total_p50"] - s_ref["latency_total_p50"]:+.1f} s vs eval'
                       if s_ref.get("n") else None), delta_color="inverse")
    m[2].metric("Latency p95", f'{s_live["latency_total_p95"]:.1f} s')
    m[3].metric("Input tokens (mean)", s_live["input_tokens_mean"])
    m[4].metric("Abstention rate", f'{s_live["abstention_rate"]:.0%}')
    m[5].metric("Error rate", f'{s_live["error_rate"]:.0%}')

    st.subheader("Latency and token usage over time")
    g1, g2 = st.columns(2)
    # One point per query: scatter (a line chart draws nothing for a single query).
    lat = live.melt(id_vars="time", value_vars=["latency_total", "latency_generation", "latency_retrieval"],
                    var_name="stage", value_name="seconds").dropna()
    g1.scatter_chart(lat, x="time", y="seconds", color="stage")
    tok = live.melt(id_vars="time", value_vars=["input_tokens", "output_tokens"],
                    var_name="kind", value_name="tokens").dropna()
    g2.scatter_chart(tok, x="time", y="tokens", color="kind")
    g3, g4 = st.columns(2)
    g3.markdown("Top text similarity per query")
    g3.scatter_chart(live.dropna(subset=["top_text_similarity"]), x="time", y="top_text_similarity")
    g4.markdown("Token totals")
    g4.table(pd.DataFrame({"input": [s_live["input_tokens_total"]], "output": [s_live["output_tokens_total"]],
                           "queries": [s_live["n"]]}))

st.subheader("Drift")
report = detect_drift(store, current_since=since, current_limit=limit or 10_000)
st.markdown(f'Overall status: <span class="status {report["status"]}">{report["status"].replace("_", " ")}</span> '
            f'&nbsp; reference (evaluation) n={report["reference_n"]}, current n={report["current_n"]}',
            unsafe_allow_html=True)
if report.get("message"):
    st.caption(report["message"])
if report.get("signals"):
    rows = []
    for name, sig in report["signals"].items():
        detail = ", ".join(f"{k}={v}" for k, v in sig.items() if k != "level")
        rows.append({"signal": name, "level": sig["level"], "detail": detail})
    st.table(pd.DataFrame(rows))
st.caption("PSI: <0.1 stable, 0.1-0.25 moderate, >0.25 significant (confirmed by a KS test at p<0.05). "
           "Out-of-domain share: current queries less similar to every evaluation query than the 5th "
           "percentile of within-reference nearest-neighbour similarity.")

st.subheader("Evaluation history (MLflow)")
try:
    import mlflow
    from services.tracking import _setup
    _setup(cfg)
    runs = mlflow.search_runs(experiment_names=[cfg.monitoring.mlflow_experiment],
                              filter_string="tags.kind = 'evaluation'", order_by=["start_time DESC"])
    if runs.empty:
        st.caption("No evaluation runs logged yet: python -m services.tracking log-eval <dir>")
    else:
        cols = [c for c in runs.columns if c.startswith("metrics.generation.") or
                c.endswith("multimodal_fused_gated@0.4.mrr") or c.endswith("visual_in_context")]
        keep = ["tags.mlflow.runName", "start_time", "params.parser", "tags.prompt_version"] + sorted(cols)[:14]
        st.dataframe(runs[[c for c in keep if c in runs.columns]], hide_index=True)
except Exception as exc:  # dashboard must render without MLflow; show why
    st.caption(f"MLflow unavailable: {exc}")

st.subheader("Recent queries")
if not live.empty:
    st.dataframe(live.sort_values("time", ascending=False)[
        ["time", "source", "question", "latency_total", "input_tokens", "output_tokens",
         "top_text_similarity", "n_images", "insufficient", "n_citations", "error"]].head(50), hide_index=True)
st.caption(f"Rendered {datetime.now():%Y-%m-%d %H:%M:%S}")
