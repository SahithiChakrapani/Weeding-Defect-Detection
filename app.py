"""
app.py
============================================================
Streamlit front-end for the Weeding Defect Detector
(HARP/INC test sheets only).

Run with:
    streamlit run app.py

The user uploads one image. The app processes it and shows:
- The annotated image with defect bounding boxes
- A summary tally (minor / major / missing)
- A per-defect breakdown table
- The reference templates the algorithm used
"""

import os
import tempfile

import streamlit as st
from PIL import Image

from pipeline import run_full_pipeline


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Weeding Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .big-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f3864;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #595959;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: #f5f8fc;
        border-left: 4px solid #2e75b6;
        padding: 0.9rem 1.1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .clean-tag {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        padding: 0.4rem 0.9rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .defect-tag {
        display: inline-block;
        background: #f8d7da;
        color: #721c24;
        padding: 0.4rem 0.9rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================
st.markdown('<div class="big-header">🔍 Weeding Defect Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a HARP/INC test sheet image to detect '
    'cut graphic defects — missing letters, broken edges, weeding errors, '
    'and other quality issues.</div>',
    unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
<strong>ℹ️ How it works:</strong> The app detects every letter on your test sheet, 
builds a reference template from the cleanest examples, then compares each letter 
against that reference. Defects are flagged when a letter deviates by area, shape, 
or position. Missing letters are detected by checking the expected grid layout.
</div>
""", unsafe_allow_html=True)


# ============================================================
# Session state
# ============================================================
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None


# ============================================================
# Upload widget
# ============================================================
uploaded = st.file_uploader(
    "📁 Upload a HARP/INC test sheet image (JPG / PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Image should show the printed sheet on a darker background, "
         "with multiple rows of HARP and INC letters clearly visible.",
)


# ============================================================
# Run analysis
# ============================================================
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp.write(uploaded.read())
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    col_preview, col_action = st.columns([3, 1])
    with col_preview:
        st.image(Image.open(tmp_path), caption=f"📷 {uploaded.name}",
                 use_container_width=True)
    with col_action:
        st.write("")
        st.write("")
        run_clicked = st.button("▶️ Run Detection",
                                type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Analyzing test sheet …"):
            progress = st.progress(0.0)
            status = st.empty()

            steps_total = 11.0
            current = [0]
            def cb(msg):
                current[0] += 1
                status.text(f"Step {current[0]}/{int(steps_total)}: {msg}")
                progress.progress(min(current[0] / steps_total, 1.0))

            try:
                results = run_full_pipeline(tmp_path, progress_callback=cb)
                st.session_state.results = results
                st.session_state.uploaded_filename = uploaded.name
                progress.empty()
                status.empty()
            except Exception as e:
                progress.empty()
                status.empty()
                st.error(f"❌ Detection failed: {e}")
                st.info("💡 Tips:\n"
                        "- The image must be a HARP/INC test sheet\n"
                        "- The sheet should be brighter than the background\n"
                        "- Multiple rows of letters must be visible\n"
                        "- Try a clearer, well-lit photo")


# ============================================================
# Display results
# ============================================================
if st.session_state.results is not None:
    r = st.session_state.results
    tally = r["tally"]

    st.markdown("---")
    st.subheader("📊 Detection Result")

    # Annotated image
    st.image(r["annotated_image"],
             caption="🟡 Yellow = minor defects   🔴 Red = major defects   ⚠️ Missing = red box at expected position",
             use_container_width=True)

    # Banner
    if tally["total_defects"] == 0 and tally["missing"] == 0:
        st.markdown('<span class="clean-tag">✅ NO DEFECTS DETECTED</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="defect-tag">⚠️ {tally["total_defects"]} '
            f'DEFECT{"S" if tally["total_defects"] != 1 else ""} DETECTED'
            f'{f" + {tally['missing']} MISSING" if tally['missing'] else ""}'
            '</span>',
            unsafe_allow_html=True)

    st.markdown("")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Letters Inspected", tally["total_blobs_inspected"])
    with col2:
        st.metric("✅ Clean", tally["clean"])
    with col3:
        st.metric("🟡 Minor Defects", tally["minor"])
    with col4:
        st.metric("🔴 Major Defects", tally["major"] - tally["missing"]
                  if tally["major"] >= tally["missing"] else tally["major"])

    if tally["missing"] > 0:
        st.warning(
            f"⚠️ **{tally['missing']} missing letter(s) detected** — "
            "these are letters expected by the grid layout but not present "
            "on the sheet. They are shown as red boxes at the expected position.")

    # ============================================================
    # Per-defect breakdown table
    # ============================================================
    if r["defects_only"]:
        st.markdown("### 📋 Per-Defect Breakdown")

        rows = []
        for idx, d in enumerate(sorted(r["defects_only"],
                                       key=lambda x: -x["score"]), 1):
            b = d["blob"]
            if d["type"] == "large":
                type_label = f"Letter (column {d['col']})"
            elif d["type"] == "super":
                type_label = "INC group"
            elif d["type"] == "missing":
                type_label = f"Missing letter (row {d['row']}, col {d['col']})"
            else:
                type_label = d["type"]

            rows.append({
                "#": idx,
                "Severity": d["verdict"].upper(),
                "Type": type_label,
                "Position (x,y)": f"({b['x']},{b['y']})",
                "Size (w×h)": f"{b['w']}×{b['h']}",
                "Score": d["score"],
                "Reasons": "; ".join(d["reasons"]) if d["reasons"] else "—",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ============================================================
    # Reference templates
    # ============================================================
    with st.expander("🔬 Reference Templates Used (advanced)", expanded=False):
        st.caption("These are the 'ideal' letter shapes the algorithm built "
                   "from your image's cleanest examples. Each upload analyzes "
                   "a defect by comparing against these references.")

        n_large = len(r["large_refs"])
        n_super = len(r["super_refs"])
        total = n_large + n_super

        if total > 0:
            cols = st.columns(min(total, 6))
            idx = 0
            for col_idx in sorted(r["large_refs"].keys()):
                ref = r["large_refs"][col_idx]
                with cols[idx % len(cols)]:
                    st.image(ref["template"],
                             caption=f"Large col {col_idx}\n"
                                     f"{ref['count']} blobs · "
                                     f"area={ref['ref_area']:,}",
                             use_container_width=True, clamp=True)
                idx += 1
            for col_idx in sorted(r["super_refs"].keys()):
                ref = r["super_refs"][col_idx]
                with cols[idx % len(cols)]:
                    comp = ref.get("ref_component_count", "N/A")
                    st.image(ref["template"],
                             caption=f"INC group\n"
                                     f"{ref['count']} blobs · "
                                     f"components={comp}",
                             use_container_width=True, clamp=True)
                idx += 1
        else:
            st.info("No reference templates available.")

    # ============================================================
    # Row anomalies (if any)
    # ============================================================
    if r["row_anomalies"]:
        with st.expander("🔍 Row anomalies detected", expanded=False):
            for ridx, reason in r["row_anomalies"].items():
                st.write(f"- Row {ridx}: {reason}")
