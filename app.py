
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import re

# === Bring in your existing logic ===
# These should exist in your repo; if the function names differ, update below.
try:
    from canvas_api import get_material_order  # def get_material_order(course_id: str) -> pd.DataFrame | list
except Exception:
    def get_material_order(course_id: str):
        st.warning("Stub get_material_order used. Replace with your canvas_api.get_material_order implementation.")
        return pd.DataFrame({"position": [1, 2, 3], "title": ["Intro", "Module 1", "Module 2"]})

try:
    from echo360_formatter import process_echo360  # def process_echo360(df: pd.DataFrame) -> pd.DataFrame
except Exception:
    def process_echo360(df: pd.DataFrame) -> pd.DataFrame:
        st.warning("Stub process_echo360 used. Replace with your echo360_formatter.process_echo360 implementation.")
        return df.copy()

try:
    from gradebook_formatter import process_gradebook  # def process_gradebook(df: pd.DataFrame) -> pd.DataFrame
except Exception:
    def process_gradebook(df: pd.DataFrame) -> pd.DataFrame:
        st.warning("Stub process_gradebook used. Replace with your gradebook_formatter.process_gradebook implementation.")
        return df.copy()

# === Streamlit page config ===
st.set_page_config(
    page_title="Course Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Course Analytics Dashboard")
st.caption("Unifies Canvas materials order, Echo360 engagement, and Gradebook formatting into a live dashboard.")

# === Helpers ===
def _drop_name_cols_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Hide personally-identifying name columns in UI table display."""
    if df is None or df.empty:
        return df
    drop_cols = []
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in {"student", "student name", "name", "user name"} or "student" in lc and "points possible" not in lc:
            drop_cols.append(c)
    out = df.drop(columns=drop_cols, errors="ignore")
    return out

def _parse_percent_like(series: pd.Series) -> pd.Series:
    """Convert strings like '88%' or numbers in 0-1/0-100 into 0-100 floats."""
    if series is None:
        return pd.Series(dtype=float)
    s = series.astype(str).str.strip()
    has_pct = s.str.endswith("%", na=False)
    s = s.str.rstrip("%")
    vals = pd.to_numeric(s, errors="coerce")
    # If values look like 0-1, scale to 0-100
    if not has_pct.any() and pd.notna(vals).sum() and vals.mean(skipna=True) <= 1.5:
        vals = vals * 100.0
    return vals

def _letter_from_percent(pct: float) -> str:
    """Map numeric percent to letter grade (typical US scale)."""
    if pct is None or not np.isfinite(pct):
        return "—"
    p = float(pct)
    # Standard-ish: A 93+, A- 90-92.99, B+ 87-89.99, B 83-86.99, B- 80-82.99, etc.
    bins = [
        (93, "A"), (90, "A-"),
        (87, "B+"), (83, "B"), (80, "B-"),
        (77, "C+"), (73, "C"), (70, "C-"),
        (67, "D+"), (63, "D"), (60, "D-"),
        (-1e9, "F"),
    ]
    for thr, letter in bins:
        if p >= thr:
            return letter
    return "F"

def _extract_numeric_grade_df(grade_df: Optional[pd.DataFrame]) -> pd.Series:
    """Return a numeric percent series for students' final grade when possible."""
    if grade_df is None or grade_df.empty:
        return pd.Series(dtype=float)

    df = grade_df.copy()

    # Drop Canvas pseudo rows like 'Points Possible' and 'Student, Test'
    if "Student" in df.columns:
        mask_pp = df["Student"].astype(str).str.contains("Points Possible", na=False)
        mask_test = df["Student"].astype(str).str.contains("Student, Test", na=False)
        df = df[~(mask_pp | mask_test)]

    # Candidate numeric columns
    num_candidates = [
        "Final Score", "Current Score", "Unposted Final Score", "Unposted Current Score",
        "Score", "Total Score", "Percentage", "Final Percent"
    ]

    # Prefer numeric score columns
    for col in num_candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if pd.notna(s).sum():
                # Normalize if 0-1 scale
                if s.mean(skipna=True) <= 1.5:
                    s = s * 100.0
                return s

    # Fall back to "Final Grade"/"Current Grade" which might be % strings or letters
    text_candidates = ["Final Grade", "Current Grade", "Unposted Final Grade", "Unposted Current Grade"]
    for col in text_candidates:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # If looks like percentage strings
            if s.str.endswith("%", na=False).any() or s.str.match(r"^\d+(\.\d+)?$").any():
                vals = _parse_percent_like(s)
                if pd.notna(vals).sum():
                    return vals
            # If looks like letters, map to midpoint of the band so we can compute a median
            letters = s.str.upper().str.extract(r"([A-D][+-]?|F)")[0]
            if letters.notna().any():
                # Map letters to numeric midpoints (approx) to allow median; we won't show this number, only the letter later
                ladder = {
                    "A": 95, "A-": 91.5,
                    "B+": 88, "B": 85, "B-": 81.5,
                    "C+": 78, "C": 75, "C-": 71.5,
                    "D+": 68, "D": 65, "D-": 61.5,
                    "F": 50,
                }
                mapped = letters.map(ladder)
                if pd.notna(mapped).sum():
                    return mapped

    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def compute_overview_metrics(echo_df: Optional[pd.DataFrame], grade_df: Optional[pd.DataFrame], order_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "Total Students": None,
        "Avg Echo View %": None,
        "Assignments Graded": None,
        "Materials Count": None,
        "Median Letter Grade": None,   # NEW
    }

    # --- Total Students (try common hints) ---
    if grade_df is not None and not grade_df.empty:
        # Heuristics for student identifier column
        student_cols = [c for c in grade_df.columns if c.lower() in {"student", "student name", "name", "user"} or "student" in c.lower()]
        if student_cols:
            # exclude 'Points Possible' and 'Student, Test'
            s = grade_df[student_cols[0]].astype(str)
            total = int((~s.str.contains("Points Possible", na=False) & ~s.str.contains("Student, Test", na=False)).sum())
            metrics["Total Students"] = total
        else:
            metrics["Total Students"] = int(len(grade_df))

        # Rough count of graded assignments (non-empty grade-like columns)
        grade_like_cols = [c for c in grade_df.columns if any(k in c.lower() for k in ["points", "score", "grade"]) and not grade_df[c].isna().all()]
        metrics["Assignments Graded"] = int(len(grade_like_cols)) if grade_like_cols else None

        # --- Median Letter Grade (NEW) ---
        numeric_grades = _extract_numeric_grade_df(grade_df)
        if pd.notna(numeric_grades).sum():
            median_pct = float(numeric_grades.median(skipna=True))
            metrics["Median Letter Grade"] = _letter_from_percent(median_pct)

    # --- Avg Echo View % ---
    if echo_df is not None and not echo_df.empty:
        # Try common column names for average view percent
        candidates = [
            "avg_view_%", "average view %", "avg view %", "average_view_pct", "avg_view_pct",
            "overall view %", "overall_view_pct", "overall_view_percent",
        ]
        col = None
        lower_map = {c.lower(): c for c in echo_df.columns}
        for cand in candidates:
            if cand in lower_map:
                col = lower_map[cand]
                break
        if not col:
            # heuristic: pick any column that looks like a percent
            perc_candidates = [c for c in echo_df.columns if "%" in c or c.lower().endswith(("_pct", "_percent", "_percentage"))]
            if perc_candidates:
                col = perc_candidates[0]
        if col:
            s = pd.to_numeric(echo_df[col], errors="coerce").dropna()
            if len(s):
                # assume values are either 0-100 or 0-1; normalize to percent
                m = s.mean()
                metrics["Avg Echo View %"] = float(m if m > 1.5 else m * 100)

    # --- Materials Count ---
    if order_df is not None and not order_df.empty:
        metrics["Materials Count"] = int(len(order_df))

    return metrics

# === Session State ===
if "echo_df" not in st.session_state:
    st.session_state.echo_df = None
if "grade_df" not in st.session_state:
    st.session_state.grade_df = None
if "order_df" not in st.session_state:
    st.session_state.order_df = None

# === Sidebar Inputs ===
with st.sidebar:
    st.header("Inputs")

    with st.expander("1) Canvas Materials Order", expanded=True):
        st.write("**Source**")
        source = st.radio("Canvas order source", ["Use API", "Upload CSV"], horizontal=True, key="canvas_source")

        # Read server secrets/env if present
        DEFAULT_BASE = st.secrets.get("canvas", {}).get("BASE_URL") or os.environ.get("CANVAS_BASE_URL") or "https://colostate.instructure.com"
        SERVER_TOKEN = st.secrets.get("canvas", {}).get("API_TOKEN") or os.environ.get("CANVAS_API_TOKEN")

        if source == "Use API":
            use_secret = st.toggle("Use server secret", value=bool(SERVER_TOKEN), help="If off, paste a token for this session only.")
            if use_secret and SERVER_TOKEN:
                base_url = DEFAULT_BASE
                api_token = SERVER_TOKEN
                st.caption("Using server-side secret.")
            else:
                base_url = st.text_input("Canvas Base URL", value=DEFAULT_BASE)
                api_token = st.text_input("Canvas API Token", type="password")

            course_id = st.text_input("Canvas Course ID", placeholder="e.g., 213019")
            if st.button("Fetch Order", use_container_width=True, type="primary"):
                if course_id.strip() and base_url and api_token:
                    with st.spinner("Fetching materials order from Canvas…"):
                        try:
                            # Provide creds to your underlying API code via env for this session only
                            os.environ["CANVAS_BASE_URL"] = base_url.strip().rstrip("/")
                            os.environ["CANVAS_API_TOKEN"] = api_token.strip()
                            order = get_material_order(course_id.strip())
                            st.session_state.order_df = pd.DataFrame(order) if not isinstance(order, pd.DataFrame) else order
                            st.success("Materials order loaded.")
                        except Exception:
                            st.error("Failed to fetch order. (Details are in server logs)")
                else:
                    st.warning("Base URL, API token, and Course ID are required.")
        else:
            canvas_csv = st.file_uploader("Upload canvas_order.csv", type=["csv"], key="canvas_order_upl")
            if canvas_csv is not None:
                try:
                    st.session_state.order_df = pd.read_csv(canvas_csv)
                    st.success("Canvas order loaded from CSV.")
                except Exception as e:
                    st.error(f"Canvas CSV read error: {e}")

    with st.expander("2) Echo360 CSV", expanded=True):
        echo_file = st.file_uploader("Upload Echo360 CSV", type=["csv"], key="echo_uploader")
        if echo_file is not None:
            try:
                raw_echo = read_csv(echo_file)
                with st.spinner("Formatting Echo360 data…"):
                    st.session_state.echo_df = process_echo360(raw_echo)
                st.success("Echo360 data processed.")
            except Exception as e:
                st.error(f"Echo360 processing error: {e}")
                st.session_state.echo_df = None

        if st.session_state.echo_df is not None:
            st.download_button(
                "Download Echo360 Processed CSV",
                st.session_state.echo_df.to_csv(index=False),
                file_name="echo360_processed.csv",
                use_container_width=True,
            )

    with st.expander("3) Gradebook CSV", expanded=True):
        grade_file = st.file_uploader("Upload Canvas Gradebook CSV", type=["csv"], key="grade_uploader")
        if grade_file is not None:
            try:
                raw_grade = read_csv(grade_file)
                with st.spinner("Formatting Gradebook…"):
                    st.session_state.grade_df = process_gradebook(raw_grade)
                st.success("Gradebook processed.")
            except Exception as e:
                st.error(f"Gradebook processing error: {e}")
                st.session_state.grade_df = None

        if st.session_state.grade_df is not None:
            st.download_button(
                "Download Gradebook Processed CSV",
                st.session_state.grade_df.to_csv(index=False),
                file_name="gradebook_processed.csv",
                use_container_width=True,
            )

# === Main Content Tabs ===
overview_tab, tables_tab, charts_tab = st.tabs(["Overview", "Tables", "Charts"])

with overview_tab:
    st.subheader("Numbers at a glance")
    metrics = compute_overview_metrics(st.session_state.echo_df, st.session_state.grade_df, st.session_state.order_df)

    # NOW FIVE KPIs (added Median Letter Grade)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", value=(metrics["Total Students"] if metrics["Total Students"] is not None else "—"))
    c2.metric("Avg Echo View %", value=(f"{metrics['Avg Echo View %']:.1f}%" if metrics["Avg Echo View %"] is not None else "—"))
    c3.metric("Assignments Graded", value=(metrics["Assignments Graded"] if metrics["Assignments Graded"] is not None else "—"))
    c4.metric("Materials Count", value=(metrics["Materials Count"] if metrics["Materials Count"] is not None else "—"))
    c5.metric("Median Letter Grade", value=(metrics["Median Letter Grade"] if metrics["Median Letter Grade"] is not None else "—"))

with tables_tab:
    st.subheader("Tables")
    t1, t2, t3 = st.tabs(["Echo360", "Gradebook (names hidden)", "Materials Order"])

    with t1:
        if st.session_state.echo_df is not None:
            st.dataframe(st.session_state.echo_df, use_container_width=True)
        else:
            st.caption("Upload an Echo360 CSV in the sidebar to see this table.")

    with t2:
        if st.session_state.grade_df is not None:
            # HIDE NAMES HERE (UI ONLY)
            clean_for_display = _drop_name_cols_for_display(st.session_state.grade_df)
            st.dataframe(clean_for_display, use_container_width=True)
            st.caption("Student names are hidden in this view. Use the download button in the sidebar to export the full processed gradebook if needed.")
        else:
            st.caption("Upload a Gradebook CSV in the sidebar to see this table.")

    with t3:
        if st.session_state.order_df is not None:
            st.dataframe(st.session_state.order_df, use_container_width=True)
        else:
            st.caption("Fetch Canvas materials order in the sidebar to see this table.")

with charts_tab:
    st.subheader("Charts")

    # === Chart 1: Engagement vs Grade (scatter) ===
    st.markdown("**Chart 1: Echo engagement vs final grade**")
    if st.session_state.echo_df is not None and st.session_state.grade_df is not None:
        echo = st.session_state.echo_df.copy()
        grade = st.session_state.grade_df.copy()

        # --- Detect columns heuristically ---
        def pick_col(df: pd.DataFrame, candidates):
            lowmap = {c.lower(): c for c in df.columns}
            for cand in candidates:
                if cand.lower() in lowmap:
                    return lowmap[cand.lower()]
            # fallback: contains
            for c in df.columns:
                if any(cand.lower() in c.lower() for cand in candidates):
                    return c
            return None

        # Common identifiers
        student_key_echo = pick_col(echo, ["student", "student name", "user", "id", "sis id", "lms_user_id"])  # best-effort
        student_key_grade = pick_col(grade, ["student", "student name", "user", "id", "sis id", "lms_user_id"])  # best-effort
        echo_pct_col = pick_col(echo, ["avg view %", "average view %", "overall view %", "avg_view_pct", "overall_view_pct"])  # best-effort
        final_grade_col = pick_col(grade, ["final score", "current score", "total score", "final grade", "current grade", "percentage"])  # best-effort

        if student_key_echo and student_key_grade and echo_pct_col and final_grade_col:
            # Aggregate Echo to one row per student
            echo_student = (
                echo[[student_key_echo, echo_pct_col]]
                .assign(_pct=pd.to_numeric(echo[echo_pct_col], errors="coerce"))
                .dropna(subset=["_pct"]) 
                .groupby(student_key_echo, as_index=False)["_pct"].mean()
                .rename(columns={"_pct": "echo_view_pct"})
            )

            grade_student = grade[[student_key_grade, final_grade_col]].copy()
            grade_student["grade_val"] = pd.to_numeric(grade_student[final_grade_col], errors="coerce")

            merged = echo_student.merge(grade_student[[student_key_grade, "grade_val"]], left_on=student_key_echo, right_on=student_key_grade, how="inner")
            if len(merged) >= 2:
                try:
                    import altair as alt
                    # Normalize percentages if needed
                    if merged["echo_view_pct"].mean() <= 1.5:
                        merged["echo_view_pct"] = merged["echo_view_pct"] * 100
                    if merged["grade_val"].mean() <= 1.5:
                        merged["grade_val"] = merged["grade_val"] * 100

                    chart1 = alt.Chart(merged).mark_circle(size=60).encode(
                        x=alt.X("echo_view_pct", title="Avg Echo View %"),
                        y=alt.Y("grade_val", title="Final/Current Grade %"),
                        tooltip=list(merged.columns),
                    ).interactive()

                    st.altair_chart(chart1, use_container_width=True)
                except Exception as e:
                    st.error(f"Altair failed to render Chart 1: {e}")
            else:
                st.info("Not enough merged rows to plot Chart 1.")
        else:
            st.info("To render Chart 1, the processed data must include: a student identifier in both Echo and Gradebook, an Echo view % column, and a final/current grade column.")
    else:
        st.caption("Upload Echo360 and Gradebook data to enable this chart.")

    st.divider()

    # === Chart 2: Materials Order Coverage (bar) ===
    st.markdown("**Chart 2: Materials order coverage / position**")
    if st.session_state.order_df is not None and not st.session_state.order_df.empty:
        odf = st.session_state.order_df.copy()
        # Try to find position & title columns
        pos_col = None
        title_col = None
        for c in odf.columns:
            lc = c.lower()
            if pos_col is None and (lc == "position" or "position" in lc or lc.endswith("order") or lc == "index"):
                pos_col = c
            if title_col is None and (lc in {"title", "name"} or "title" in lc or "module" in lc):
                title_col = c
        if pos_col is None:
            odf["position"] = np.arange(1, len(odf) + 1)
            pos_col = "position"
        if title_col is None:
            title_col = odf.columns[0]

        try:
            import altair as alt
            chart2 = alt.Chart(odf).mark_bar().encode(
                x=alt.X(f"{pos_col}:O", title="Position"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[pos_col, title_col],
            )
            st.altair_chart(chart2, use_container_width=True)
        except Exception as e:
            st.error(f"Altair failed to render Chart 2: {e}")
    else:
        st.caption("Fetch the Canvas materials order to enable this chart.")

# === Footer ===
st.markdown("---")
st.caption("Student names are hidden in the Gradebook table above; the sidebar download preserves the original processed file. Adjust the letter-grade bands in code if your scale differs.")
