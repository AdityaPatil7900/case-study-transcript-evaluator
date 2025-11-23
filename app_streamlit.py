# app_streamlit.py (fixed)
import streamlit as st
import pandas as pd
import json
import tempfile
import os

# Import scoring functions
from score_engine import load_rubric, score_transcript

st.set_page_config(page_title="Case Study Scorer", layout="wide")
st.title("Transcript Scorer — Case Study")

# default path (update if your cleaned rubric is in a different location)
# I recommend using the cleaned CSV produced by parse_rubric_custom.py
DEFAULT_RUBRIC = r"D:\PROJECTS\evaluates a student’s spoken introduction using the given transcript\rubric_clean.csv"
DEFAULT_EXCEL = r"D:\PROJECTS\evaluates a student’s spoken introduction using the given transcript\Case study for interns.xlsx"

# Sidebar controls
st.sidebar.markdown("## Rubric")
rubric_path_input = st.sidebar.text_input("Rubric path (CSV or Excel)", value=DEFAULT_RUBRIC)

# Reload button: load immediately when clicked, but do NOT call st.experimental_rerun()
if st.sidebar.button("Reload rubric"):
    try:
        # Try CSV first (rubric_clean), else Excel
        if rubric_path_input.lower().endswith(".csv") and os.path.exists(rubric_path_input):
            rubric_df = pd.read_csv(rubric_path_input)
        else:
            # attempt to load with our loader (it accepts Excel or CSV paths)
            rubric_df = load_rubric(rubric_path_input)
        st.sidebar.success("Rubric reloaded successfully.")
        # store to session state so the value persists for this session
        st.session_state['rubric_df'] = rubric_df
    except Exception as e:
        st.sidebar.error(f"Failed to load rubric: {e}")
        st.session_state['rubric_df'] = None

# Allow upload of a rubric file (Excel)
st.sidebar.markdown("### Or upload rubric Excel (will override path)")
rubric_file = st.sidebar.file_uploader("Upload rubric (.xlsx or .csv)", type=['xlsx', 'csv'])
if rubric_file is not None:
    try:
        # if CSV uploaded
        if rubric_file.type == "text/csv" or rubric_file.name.lower().endswith(".csv"):
            rubric_df = pd.read_csv(rubric_file)
        else:
            # save to temp file and call load_rubric to normalize
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp_name = tmp.name
                tmp.write(rubric_file.getbuffer())
            rubric_df = load_rubric(tmp_name)
            # cleanup temp file
            try:
                os.remove(tmp_name)
            except:
                pass
        st.sidebar.success("Rubric loaded from upload")
        st.session_state['rubric_df'] = rubric_df
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded rubric: {e}")
        st.session_state['rubric_df'] = None

# Initialize rubric_df from session state or attempt auto-load
if 'rubric_df' in st.session_state:
    rubric_df = st.session_state['rubric_df']
else:
    try:
        # try CSV first, then use load_rubric (Excel)
        if os.path.exists(DEFAULT_RUBRIC):
            rubric_df = pd.read_csv(DEFAULT_RUBRIC)
        elif os.path.exists(DEFAULT_EXCEL):
            rubric_df = load_rubric(DEFAULT_EXCEL)
        else:
            # try user-provided path if exists
            if os.path.exists(rubric_path_input):
                if rubric_path_input.lower().endswith(".csv"):
                    rubric_df = pd.read_csv(rubric_path_input)
                else:
                    rubric_df = load_rubric(rubric_path_input)
            else:
                rubric_df = None
    except Exception as e:
        st.sidebar.error(f"Failed to load default rubric: {e}")
        rubric_df = None
    st.session_state['rubric_df'] = rubric_df

# UI for transcript input
st.markdown("**Paste transcript text below or upload .txt file**")
text = st.text_area("Transcript text", height=300)
uploaded = st.file_uploader("Or upload transcript (.txt)", type=['txt'])
if uploaded is not None and not text:
    try:
        text = uploaded.read().decode('utf-8')
    except Exception:
        text = uploaded.getvalue().decode('utf-8')

# Scoring
if st.button("Score"):
    if rubric_df is None:
        st.warning("Rubric is not loaded. Upload a rubric or provide a valid rubric path in the sidebar.")
    elif not text or text.strip() == "":
        st.warning("Please paste or upload a transcript first.")
    else:
        with st.spinner("Scoring..."):
            try:
                out = score_transcript(text, rubric_df)
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                raise

        st.metric("Overall Score", f"{out['overall_score']:.2f}/100")

        # Result table
        df_rows = []
        for pc in out['per_criterion']:
            df_rows.append({
                "Criterion": pc.get('criterion', ''),
                "RuleScore": round(pc.get('rule_score', 0), 3),
                "SemanticScore": round(pc.get('semantic_score', 0), 3),
                "Combined": round(pc.get('combined_raw', 0), 3),
                "KeywordsFound": ", ".join(pc.get('keywords_found', [])) if pc.get('keywords_found') else "",
                "Words": pc.get('nwords', 0),
                "LengthFeedback": pc.get('length_feedback') or ""
            })
        df = pd.DataFrame(df_rows)
        st.dataframe(df)

        st.markdown("### Per-criterion feedback (short)")
        for pc in out['per_criterion']:
            kf = ", ".join(pc.get('keywords_found')) if pc.get('keywords_found') else 'None'
            st.markdown(f"**{pc.get('criterion','')}** — Keywords: {kf}; Words: {pc.get('nwords')}; {pc.get('length_feedback') or 'OK'}; Sem: {pc.get('semantic_score'):.2f}")

        st.download_button("Download JSON result", data=json.dumps(out, ensure_ascii=False, indent=2), file_name="scoring_result.json")
