import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout
import plotly.express as px
from hugging_face_authentication import hugging_face_auth


st.set_page_config(
    page_title="AI Sentiment Analysis Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── session state ────────────────────────────────────────────────────────────
for _k, _v in {
    "gpu_ok": None,
    "gpu_name": None,
    "model_manager": None,
    "dataset_loader": None,
    "pipeline": None,
    "selected_column": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── sidebar ──────────────────────────────────────────────────────────────────
def _dot(active):
    return "🟢" if active else "⚪"


with st.sidebar:
    st.title("Sentiment Pipeline")

    page = st.radio(
        "Navigate",
        [
            "System Status",
            "Load Model",
            "Load Dataset",
            "Run Pipeline",
            "Generate Report",
            "Browse Reports",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Session")

    mm = st.session_state.model_manager
    dl = st.session_state.dataset_loader
    pl = st.session_state.pipeline

    st.markdown(
        f"{_dot(st.session_state.gpu_ok)} GPU: "
        + (st.session_state.gpu_name if st.session_state.gpu_ok else "not checked")
    )
    st.markdown(f"{_dot(mm)} Model: " + (mm.model_id if mm else "none"))
    st.markdown(f"{_dot(dl)} Dataset: " + (dl.dataset_id if dl else "none"))
    st.markdown(f"{_dot(pl)} Pipeline: " + ("ready" if pl else "not ready"))


# ─── helper ───────────────────────────────────────────────────────────────────
def _latest_report_dir():
    root = Path("reports")
    dirs = sorted(root.glob("sentiment_report_*"), reverse=True)
    return dirs[0] if dirs else None


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM STATUS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "System Status":
    st.title("System Status")

    col_hw, col_env = st.columns(2)

    with col_hw:
        st.subheader("Hardware")
        if st.button("Check GPU"):
            try:
                import torch
                ok = torch.cuda.is_available()
                st.session_state.gpu_ok = ok
                st.session_state.gpu_name = torch.cuda.get_device_name(0) if ok else None
                st.rerun()
            except Exception as e:
                st.error(f"Error checking GPU: {e}")

        if st.session_state.gpu_ok is True:
            import torch
            props = torch.cuda.get_device_properties(0)
            st.success(f"GPU detected: **{st.session_state.gpu_name}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("VRAM Total", f"{props.total_memory / 1e9:.1f} GB")
            c2.metric("VRAM Allocated", f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            c3.metric("VRAM Reserved", f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        elif st.session_state.gpu_ok is False:
            st.error("No CUDA GPU detected. This pipeline requires one.")
        else:
            st.info("Click 'Check GPU' to detect available hardware.")

    with col_env:
        st.subheader("Environment")
        for pkg in ["torch", "transformers", "datasets", "bitsandbytes"]:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
                st.markdown(f"✅ **{pkg}** `{ver}`")
            except ImportError:
                st.markdown(f"❌ **{pkg}** — not installed")

        st.divider()
        if hugging_face_auth:
            st.markdown(f"✅ we are logged in the HuggingFace hub")
        else:
            st.warning("HF_TOKEN not found in environment. Gated models require it.")


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Load Model":
    st.title("Load Model")

    model_id = st.text_input(
        "HuggingFace Model ID",
        value="Qwen/Qwen2.5-3B-Instruct",
        placeholder="e.g. Qwen/Qwen2.5-3B-Instruct",
    )
    use_quant = st.checkbox(
        "Enable 4-bit NF4 quantization (recommended for GPUs with < 12 GB VRAM)",
        value=True,
    )
    st.caption(
        "Quantization shrinks a 3B model from ~6 GB to ~2 GB VRAM. "
        "Disable only if you have sufficient VRAM and need maximum precision."
    )

    if st.session_state.model_manager:
        st.info(f"Currently loaded: **{st.session_state.model_manager.model_id}** — submit a new ID to replace it.")

    if st.button("Load Model", type="primary"):
        with st.spinner(f"Loading **{model_id}**… First run downloads model weights, which may take several minutes."):
            try:
                from model_manager import ModelManager
                new_mm = ModelManager(model_id=model_id, quantization_setting=use_quant)
                st.session_state.model_manager = new_mm
                st.session_state.pipeline = None  # stale pipeline
                st.success(f"Model loaded: **{model_id}**")
            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if st.session_state.model_manager:
        loaded_mm = st.session_state.model_manager
        st.subheader("Loaded model info")
        st.write(f"**Model ID:** `{loaded_mm.model_id}`")
        try:
            import torch
            st.write(f"**Device:** {torch.cuda.get_device_name(0)}")
            st.write(f"**VRAM in use:** {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Load Dataset":
    st.title("Load Dataset")

    dataset_id = st.text_input(
        "HuggingFace Dataset ID",
        value="mteb/tweet_sentiment_extraction",
        placeholder="e.g. mteb/tweet_sentiment_extraction",
    )

    if st.session_state.dataset_loader:
        st.info(f"Currently loaded: **{st.session_state.dataset_loader.dataset_id}**")

    if st.button("Load Dataset", type="primary"):
        with st.spinner(f"Loading **{dataset_id}**…"):
            try:
                from dataset_loader import DatasetLoader
                new_dl = DatasetLoader(dataset_id=dataset_id)
                st.session_state.dataset_loader = new_dl
                st.session_state.selected_column = None
                st.session_state.pipeline = None  # stale pipeline
                st.success(f"Loaded: **{dataset_id}**")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

    if st.session_state.dataset_loader:
        loaded_dl = st.session_state.dataset_loader

        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(loaded_dl.dataset):,}")
        c2.metric("Columns", len(loaded_dl.dataset.column_names))
        st.write("**Columns:** " + "  ·  ".join(f"`{c}`" for c in loaded_dl.dataset.column_names))

        st.divider()
        st.subheader("Dataset preview")
        n_preview = st.slider("Rows to show", 5, 100, 10, step=5)
        df_preview = loaded_dl.dataset.select(range(min(n_preview, len(loaded_dl.dataset)))).to_pandas()
        st.dataframe(df_preview, use_container_width=True)

        st.divider()
        st.subheader("Select text column for pipeline")
        col_names = loaded_dl.dataset.column_names
        default_idx = (
            col_names.index(st.session_state.selected_column)
            if st.session_state.selected_column in col_names
            else 0
        )
        col_choice = st.selectbox("Column to analyze for sentiment", col_names, index=default_idx)
        if st.button("Confirm column selection"):
            st.session_state.selected_column = col_choice
            st.success(f"Column confirmed: **{col_choice}**")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Run Pipeline":
    st.title("Run Pipeline")

    missing = []
    if not st.session_state.model_manager:
        missing.append("Model — go to **Load Model**")
    if not st.session_state.dataset_loader:
        missing.append("Dataset — go to **Load Dataset**")
    for m in missing:
        st.warning(f"Missing: {m}")
    if missing:
        st.stop()

    run_mm = st.session_state.model_manager
    run_dl = st.session_state.dataset_loader

    # initialize pipeline
    if st.session_state.pipeline is None:
        st.info("Pipeline not yet initialized. Click below to build it from the loaded model and dataset.")
        if st.button("Initialize Pipeline", type="primary"):
            with st.spinner("Building text-generation pipeline…"):
                try:
                    from sentiment_pipeline import SentimentPipeline
                    new_pl = SentimentPipeline(run_mm, run_dl)
                    st.session_state.pipeline = new_pl
                    st.success("Pipeline ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        st.stop()

    run_pl = st.session_state.pipeline
    st.success("Pipeline is ready.")
    st.divider()

    st.subheader("Run configuration")
    col_a, col_b = st.columns(2)

    with col_a:
        col_names = run_dl.dataset.column_names
        default_col_idx = (
            col_names.index(st.session_state.selected_column)
            if st.session_state.selected_column in col_names
            else 0
        )
        run_col = st.selectbox("Text column to classify", col_names, index=default_col_idx)

    with col_b:
        total_rows = len(run_dl.dataset)
        max_rows = st.number_input(
            f"Rows to process (dataset has {total_rows:,})",
            min_value=1,
            max_value=total_rows,
            value=min(200, total_rows),
        )

    output_csv = st.text_input("Save annotated CSV to", value=str(Path("reports_csv_files") / "sentiment_debug_output.csv"))

    if st.button("Run Sentiment Analysis", type="primary"):
        run_slice = run_dl.dataset.select(range(int(max_rows)))
        sentiments = []
        failed = 0

        prog = st.progress(0.0)
        status = st.empty()

        for i, example in enumerate(run_slice):
            phrase = str(example[run_col])
            result = run_pl.running_pipeline_on_a_record(phrase)
            if result is None:
                failed += 1
                result = 0
            sentiments.append(result)
            prog.progress((i + 1) / max_rows, text=f"Row {i + 1:,} / {max_rows:,}  |  parse failures: {failed}")

        prog.empty()
        status.empty()

        df_out = run_slice.to_pandas()
        df_out["sentiment"] = sentiments
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False)

        # update loader dataset so generating_report can use it
        import datasets as _hf
        run_dl.dataset = _hf.Dataset.from_pandas(df_out)

        st.success(
            f"Finished {max_rows:,} rows — parse failures defaulted to 0: **{failed}**. "
            f"Saved to `{output_csv}`."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Positive (1)", sentiments.count(1))
        c2.metric("Neutral (0)", sentiments.count(0))
        c3.metric("Negative (-1)", sentiments.count(-1))

        st.subheader("Sample results (first 30 rows)")
        st.dataframe(df_out[[run_col, "sentiment"]].head(30), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Generate Report":
    st.title("Generate Report")

    st.markdown(
        "Reads an annotated CSV (produced by **Run Pipeline**) and generates a full "
        "analytical report: sentiment distribution, polarity score, top keywords, "
        "confusion matrix (if ground-truth labels exist), and example rows."
    )

    csv_dir = Path("reports_csv_files")
    available_csvs = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []

    if not available_csvs:
        st.warning("No CSV files found in `reports_csv_files/`. Run the pipeline first to generate one.")
        st.stop()

    selected_csv_name = st.selectbox(
        "Select annotated CSV to analyze",
        [f.name for f in available_csvs],
    )
    csv_path = str(csv_dir / selected_csv_name)

    df_peek = pd.read_csv(csv_path, nrows=5)
    st.write("**CSV preview (first 5 rows):**")
    st.dataframe(df_peek, use_container_width=True)

    if "sentiment" not in df_peek.columns:
        st.error("The CSV has no `sentiment` column. Run the pipeline on a dataset first.")
        st.stop()

    _exclude = {"sentiment", "sentiment_name", "char_count", "word_count"}
    _text_candidates = [c for c in df_peek.columns.tolist() if c not in _exclude]
    if not _text_candidates:
        st.error("No usable text column found in this CSV.")
        st.stop()
    _default_idx = _text_candidates.index("text") if "text" in _text_candidates else 0
    chosen_text_col = st.selectbox(
        "Text column to analyze",
        _text_candidates,
        index=_default_idx,
        help="Select the column containing the text the model classified.",
    )

    st.write(f"**Full CSV:** {pd.read_csv(csv_path).shape[0]:,} rows")

    if st.button("Generate Report", type="primary"):
        from sentiment_pipeline import SentimentPipeline

        # generating_report reads only the CSV — it does not call the model.
        # If the live pipeline instance is available use it; otherwise use a
        # lightweight proxy so we don't require a loaded model just for reporting.
        if st.session_state.pipeline is not None:
            reporter = st.session_state.pipeline
        else:
            class _ReportProxy:
                pass
            reporter = _ReportProxy()

        reporter.selected_column = chosen_text_col

        buf = StringIO()
        with st.spinner("Generating report…"):
            try:
                with redirect_stdout(buf):
                    SentimentPipeline.generating_report(reporter, csv_path)
                report_text = buf.getvalue()
                st.success("Report generated!")
                st.code(report_text, language="text")
                rdir = _latest_report_dir()
                if rdir:
                    st.info(f"Artifacts saved to `{rdir}` — browse them in **Browse Reports**.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════════
# BROWSE REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Browse Reports":
    st.title("Browse Reports")

    reports_root = Path("reports")

    if not reports_root.exists():
        st.info("No reports directory found. Generate a report first.")
        st.stop()

    report_dirs = sorted(
        [d for d in reports_root.iterdir() if d.is_dir()],
        reverse=True,
    )

    if not report_dirs:
        st.info("No reports found yet. Use **Generate Report** to create one.")
        st.stop()

    selected_name = st.selectbox(
        "Select a report",
        [d.name for d in report_dirs],
        format_func=lambda name: name.replace("sentiment_report_", "").replace("_", " "),
    )
    rdir = reports_root / selected_name

    tabs = st.tabs(["Report", "Summary & Charts", "Full Dataset", "Top Words", "Confusion Matrix", "Downloads"])

    # ── Tab: Report text ──────────────────────────────────────────────────────
    with tabs[0]:
        rtxt = rdir / "report.txt"
        if rtxt.exists():
            st.code(rtxt.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("report.txt not found in this report directory.")

    # ── Tab: Summary & Charts ─────────────────────────────────────────────────
    with tabs[1]:
        sjson = rdir / "summary.json"
        if sjson.exists():
            summary = json.loads(sjson.read_text(encoding="utf-8"))

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows analyzed", f"{summary.get('total_rows_analyzed', '?'):,}")
            polarity = summary.get("polarity_score", "?")
            if isinstance(polarity, (int, float)):
                c2.metric("Polarity score", polarity, delta=polarity, delta_color="normal")
            else:
                c2.metric("Polarity score", polarity)
            agr = summary.get("agreement_with_labels_percent")
            c3.metric("Label agreement", f"{agr}%" if agr is not None else "N/A")

            counts = summary.get("sentiment_counts", {})
            pcts = summary.get("sentiment_percentages", {})
            dist_df = pd.DataFrame({
                "Sentiment": list(counts.keys()),
                "Count": [int(v) for v in counts.values()],
                "Percentage (%)": [round(pcts.get(k, 0.0), 2) for k in counts.keys()],
            })

            _sent_colors = {"negative": "#EF5350", "neutral": "#9E9E9E", "positive": "#4CAF50"}
            col_t, col_c = st.columns([1, 2])
            with col_t:
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
            with col_c:
                fig_dist = px.bar(
                    dist_df,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map=_sent_colors,
                    text="Count",
                )
                fig_dist.update_traces(textposition="outside")
                fig_dist.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)

            avg = summary.get("average_lengths", {})
            if avg:
                st.subheader("Average text length by sentiment")
                avg_df = pd.DataFrame([
                    {
                        "Sentiment": k,
                        "Avg chars": v.get("char_count"),
                        "Avg words": v.get("word_count"),
                    }
                    for k, v in avg.items()
                ])
                st.dataframe(avg_df, use_container_width=True, hide_index=True)

            st.subheader("Raw JSON")
            st.json(summary)
        else:
            st.warning("summary.json not found.")

    # ── Tab: Full dataset ─────────────────────────────────────────────────────
    with tabs[2]:
        dcsv = rdir / "sentiment_dataset.csv"
        if dcsv.exists():
            df_full = pd.read_csv(dcsv)
            st.write(f"Shape: **{df_full.shape[0]:,} rows × {df_full.shape[1]} columns**")

            if "sentiment" in df_full.columns:
                sentiment_map = {-1: "Negative (-1)", 0: "Neutral (0)", 1: "Positive (1)"}
                available = sorted(df_full["sentiment"].dropna().unique().astype(int).tolist())
                chosen_sentiments = st.multiselect(
                    "Filter by sentiment",
                    options=available,
                    default=available,
                    format_func=lambda x: sentiment_map.get(x, str(x)),
                )
                df_show = df_full[df_full["sentiment"].isin(chosen_sentiments)]
            else:
                df_show = df_full

            def _color_sentiment_cell(val):
                _c = {
                    "positive": ("#4CAF50", "rgba(76,175,80,0.15)"),
                    "negative": ("#EF5350", "rgba(239,83,80,0.15)"),
                    "neutral": ("#9E9E9E", "rgba(158,158,158,0.15)"),
                    1: ("#4CAF50", "rgba(76,175,80,0.15)"),
                    -1: ("#EF5350", "rgba(239,83,80,0.15)"),
                    0: ("#9E9E9E", "rgba(158,158,158,0.15)"),
                }
                if val in _c:
                    fg, bg = _c[val]
                    return f"color:{fg};background-color:{bg};font-weight:600"
                return ""

            _style_cols = [c for c in ["sentiment", "sentiment_name"] if c in df_show.columns]
            if _style_cols:
                try:
                    st.dataframe(df_show.style.map(_color_sentiment_cell, subset=_style_cols), use_container_width=True)
                except Exception:
                    st.dataframe(df_show, use_container_width=True)
            else:
                st.dataframe(df_show, use_container_width=True)

            st.download_button(
                "Download filtered CSV",
                df_show.to_csv(index=False).encode(),
                file_name="filtered_sentiment_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("sentiment_dataset.csv not found.")

    # ── Tab: Top words ────────────────────────────────────────────────────────
    with tabs[3]:
        _word_palette = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#EF5350"}
        any_found = False
        for label in ["positive", "neutral", "negative"]:
            wfile = rdir / f"top_words_{label}.csv"
            if wfile.exists():
                any_found = True
                wdf = pd.read_csv(wfile)
                with st.expander(f"{label.capitalize()} — top keywords", expanded=True):
                    col_t, col_c = st.columns([1, 2])
                    with col_t:
                        st.dataframe(wdf, use_container_width=True, hide_index=True)
                    with col_c:
                        fig_w = px.bar(
                            wdf.sort_values("count"),
                            x="count",
                            y="word",
                            orientation="h",
                            color_discrete_sequence=[_word_palette.get(label, "#2196F3")],
                            text="count",
                        )
                        fig_w.update_traces(textposition="outside")
                        fig_w.update_layout(
                            showlegend=False,
                            height=max(250, len(wdf) * 28),
                            margin=dict(l=0, r=0, t=10, b=10),
                            yaxis_title="",
                            xaxis_title="Count",
                        )
                        st.plotly_chart(fig_w, use_container_width=True)
        if not any_found:
            st.info("No top-words files found in this report.")

    # ── Tab: Confusion matrix ─────────────────────────────────────────────────
    with tabs[4]:
        cmfile = rdir / "confusion_matrix.csv"
        if cmfile.exists():
            cm_df = pd.read_csv(cmfile, index_col=0)
            st.subheader("Confusion Matrix")
            st.caption("Rows = ground-truth labels · Columns = model predictions")
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="RdYlGn",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                aspect="auto",
            )
            fig_cm.update_traces(textfont_size=18)
            fig_cm.update_layout(
                height=380,
                coloraxis_showscale=False,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info(
                "No confusion matrix for this report. "
                "It is computed only when the dataset contains a ground-truth `label` column."
            )

    # ── Tab: Downloads ────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Download individual report files")
        download_files = [
            ("report.txt", "text/plain"),
            ("summary.json", "application/json"),
            ("sentiment_dataset.csv", "text/csv"),
            ("top_words_positive.csv", "text/csv"),
            ("top_words_neutral.csv", "text/csv"),
            ("top_words_negative.csv", "text/csv"),
            ("confusion_matrix.csv", "text/csv"),
        ]
        found_any = False
        for fname, mime in download_files:
            fpath = rdir / fname
            if fpath.exists():
                found_any = True
                st.download_button(
                    label=f"Download {fname}",
                    data=fpath.read_bytes(),
                    file_name=fname,
                    mime=mime,
                    key=f"dl_{selected_name}_{fname}",
                )
        if not found_any:
            st.info("No downloadable files found in this report.")
