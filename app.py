
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os
from typing import Optional, Tuple

st.set_page_config(page_title="Course Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Course Analytics Dashboard")
st.caption("Live view combining Canvas module order, Echo360 engagement, and gradebook metrics — with secure API handling.")

# ---------- Utilities ----------
def _normalize_col(col: str) -> str:
    c = col.strip().lower()
    c = re.sub(r"[_\-]+", " ", c)
    c = re.sub(r"[^a-z0-9 %:]+", "", c)
    c = re.sub(r"\s+", " ", c)
    return c

def _find_col(df: pd.DataFrame, candidates, required=False):
    if df is None or df.empty:
        if required:
            raise KeyError(f"Missing column: tried {candidates} but DF is empty")
        return None
    norm_map = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    for k, orig in norm_map.items():
        for cand in candidates:
            if cand in k:
                return orig
    if required:
        raise KeyError(f"Missing column. Looked for any of: {candidates}\nAvailable: {list(df.columns)}")
    return None

def _parse_hms_to_seconds(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        pass
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except Exception:
        return np.nan
    return np.nan

def _coerce_time_to_seconds(df: pd.DataFrame, colname: str) -> pd.Series:
    s = df[colname]
    return s.astype(float) if s.dtype.kind in "if" else s.apply(_parse_hms_to_seconds)

def _first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan

# ---------- Canvas API ----------
def _duration_to_seconds(dur_str: str) -> Optional[int]:
    if not dur_str:
        return None
    parts = dur_str.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    else:
        return None
    return h * 3600 + m * 60 + s

def _normalize_title(title: str) -> Optional[str]:
    if not title:
        return None
    t = title.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.\-–—:;]+$", "", t)
    return t

def _parse_title_and_duration(raw_title: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if not raw_title:
        return None, None, None
    title = raw_title.strip()
    duration_seconds = None
    if "(" in raw_title and raw_title.endswith(")"):
        try:
            base, dur = raw_title.rsplit("(", 1)
            title = base.strip()
            dur = dur.strip(")")
            duration_seconds = _duration_to_seconds(dur)
        except Exception:
            pass
    return title, _normalize_title(title), duration_seconds

def _extract_echo_embeds_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if "echo360.org" not in src and "external_tools/retrieve" not in src:
            continue
        iframe_title = iframe.get("title") or ""
        raw, norm, dur_s = _parse_title_and_duration(iframe_title)
        results.append({
            "video_title_raw": raw,
            "video_title_normalized": norm,
            "video_duration_seconds": dur_s,
        })
    return results

def fetch_canvas_order_df(base_url: str, token: str, course_id: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}

    def get_modules(cid):
        url = f"{base_url}/api/v1/courses/{cid}/modules?include=items&per_page=100"
        modules = []
        while url:
            r = requests.get(url, headers=headers); r.raise_for_status()
            modules.extend(r.json())
            url = r.links.get("next", {}).get("url")
        return modules

    def get_page_body(cid, page_url):
        url = f"{base_url}/api/v1/courses/{cid}/pages/{page_url}"
        r = requests.get(url, headers=headers); r.raise_for_status()
        return r.json().get("body") or ""

    def get_student_count(cid, states=("active",)):
        url = f"{base_url}/api/v1/courses/{cid}/enrollments?type[]=StudentEnrollment&per_page=100"
        if states:
            for stt in states:
                url += f"&state[]={stt}"
        users = set()
        while url:
            r = requests.get(url, headers=headers); r.raise_for_status()
            for enr in r.json():
                uid = enr.get("user_id")
                if uid is not None:
                    users.add(uid)
            url = r.links.get("next", {}).get("url")
        return len(users)

    videos, assignments = [], []
    modules = get_modules(course_id)
    for m in modules:
        mod_name = m.get("name")
        has_video, has_assign = False, False
        for it in m.get("items", []):
            item_type = it.get("type")
            title = (it.get("title") or "").strip()
            if item_type == "ExternalTool" and "echo360.org" in (it.get("external_url") or ""):
                has_video = True
                raw, norm, dur_s = _parse_title_and_duration(title)
                videos.append({"module": mod_name, "video_title_raw": raw, "video_title_normalized": norm, "video_duration_seconds": dur_s})
            elif item_type == "Page":
                html = get_page_body(course_id, it.get("page_url"))
                embeds = _extract_echo_embeds_from_html(html)
                if embeds:
                    has_video = True
                    for e in embeds:
                        videos.append({"module": mod_name, **e})
            elif item_type in ("Assignment", "Quiz", "Discussion"):
                has_assign = True
                assignments.append({"module": mod_name, "assignment_title": title, "assignment_type": item_type})
        if not has_video:
            videos.append({"module": mod_name, "video_title_raw": "none", "video_title_normalized": "none", "video_duration_seconds": "none"})
        if not has_assign:
            assignments.append({"module": mod_name, "assignment_title": "none", "assignment_type": "none"})

    vdf = pd.DataFrame(videos, columns=["module", "video_title_raw", "video_title_normalized", "video_duration_seconds"])
    adf = pd.DataFrame(assignments, columns=["module", "assignment_title", "assignment_type"])

    try:
        student_count = get_student_count(course_id)
        vdf["# of Students"] = student_count
    except Exception:
        vdf["# of Students"] = np.nan

    combined = pd.concat([vdf.reset_index(drop=True), adf.reset_index(drop=True)], axis=1)
    return combined

# ---------- Echo + Gradebook processors ----------
def process_echo_with_order(echo_df: pd.DataFrame, canvas_df: pd.DataFrame, grades_df: Optional[pd.DataFrame]):
    media_col_e = _find_col(echo_df, ["media name","media","media title","video title","title","name"], required=True)
    duration_col = _find_col(echo_df, ["duration","video duration","media duration","length"], required=True)
    total_view_time_col = _find_col(echo_df, ["total view time","total viewtime","total watch time","view time"], required=True)
    avg_view_time_col = _find_col(echo_df, ["average view time","avg view time","avg watch time","average watch time"], required=False)

    echo_df = echo_df.copy()
    echo_df[duration_col] = _coerce_time_to_seconds(echo_df, duration_col)
    echo_df[total_view_time_col] = _coerce_time_to_seconds(echo_df, total_view_time_col)
    if avg_view_time_col:
        echo_df[avg_view_time_col] = _coerce_time_to_seconds(echo_df, avg_view_time_col)

    echo_df["True View %"] = np.where(echo_df[duration_col] > 0, (echo_df[total_view_time_col] / echo_df[duration_col]) * 100.0, np.nan)
    echo_df["True View %"] = pd.to_numeric(echo_df["True View %"], errors="coerce")

    students_col = _find_col(canvas_df, ["# of students","number of students","num students","students","enrollment","total students"], required=False)
    media_col_c  = _find_col(canvas_df, ["media name","media","media title","video title","title","name"], required=False)
    module_col   = _find_col(canvas_df, ["module","module name","module title"], required=False)

    def _gradebook_student_count(gb: Optional[pd.DataFrame]) -> Optional[int]:
        if gb is None or gb.empty:
            return None
        if "Student" in gb.columns:
            tmp = gb[~gb["Student"].isin(["Points Possible","Student, Test"])].copy()
            for c in ["Final Grade","Current Grade","Unposted Final Grade","Final Score","Current Score","Unposted Final Score"]:
                if c in tmp.columns:
                    tmp = tmp[~tmp[c].astype(str).str.contains("read only", case=False, na=False)]
            return int(tmp.shape[0])
        return None

    if students_col is not None and students_col in canvas_df.columns:
        class_students_series = pd.to_numeric(canvas_df[students_col], errors="coerce").dropna()
        if len(class_students_series):
            mode_vals = class_students_series.mode()
            class_total_students = int(mode_vals.iloc[0]) if len(mode_vals) else int(class_students_series.max())
        else:
            class_total_students = _gradebook_student_count(grades_df)
    else:
        class_total_students = _gradebook_student_count(grades_df)

    grouped = echo_df.groupby(media_col_e, dropna=False)
    if avg_view_time_col:
        avg_view_time_series = grouped[avg_view_time_col].mean()
    else:
        avg_view_time_series = grouped[total_view_time_col].sum() / grouped.size()

    echo_summary_df = pd.DataFrame({
        "Media Title": grouped[media_col_e].apply(_first_non_null),
        "Video Duration": grouped[duration_col].apply(_first_non_null),
        "# of Unique Viewers": grouped.size(),
        "Average View %": (grouped["True View %"].mean() / 100.0),
        "Average View Time": avg_view_time_series,
    })

    if class_total_students and class_total_students > 0:
        echo_summary_df["% of Students Viewing"] = (echo_summary_df["# of Unique Viewers"] / class_total_students)
    else:
        echo_summary_df["% of Students Viewing"] = np.nan

    sum_total_view_time = grouped[total_view_time_col].sum()
    echo_summary_df["% of Video Viewed Overall"] = np.nan
    if class_total_students and class_total_students > 0:
        denom = echo_summary_df["Video Duration"] * class_total_students
        with np.errstate(divide="ignore", invalid="ignore"):
            echo_summary_df["% of Video Viewed Overall"] = (sum_total_view_time.reindex(echo_summary_df.index).values / denom.values)

    if media_col_c is not None and media_col_c in canvas_df.columns:
        ordered_media = canvas_df[media_col_c].dropna().astype(str).tolist()
        echo_summary_df = echo_summary_df.reset_index(drop=True)
        echo_summary_df["__order__"] = pd.Categorical(echo_summary_df["Media Title"].astype(str), categories=ordered_media, ordered=True)
        echo_summary_df = echo_summary_df.sort_values(["__order__", "Media Title"], na_position="last").drop(columns="__order__")
    else:
        echo_summary_df = echo_summary_df.sort_values("Media Title")

    grand_row = {"Media Title": "Grand Total"}
    for col in ["Video Duration", "# of Unique Viewers", "Average View %", "Average View Time", "% of Students Viewing", "% of Video Viewed Overall"]:
        grand_row[col] = echo_summary_df[col].mean(skipna=True) if col in echo_summary_df.columns else np.nan
    echo_summary_df = pd.concat([echo_summary_df.reset_index(drop=True), pd.DataFrame([grand_row])], ignore_index=True)

    module_table_cols = ["Module", "Average View %", "# of Students Viewing", "Overall View %", "# of Students"]
    module_table = pd.DataFrame(columns=module_table_cols)

    if module_col is not None and media_col_c is not None and module_col in canvas_df.columns and media_col_c in canvas_df.columns:
        mod_groups = canvas_df[[module_col, media_col_c]].dropna(subset=[module_col]).copy()
        mod_groups[module_col] = mod_groups[module_col].astype(str)
        mod_groups[media_col_c] = mod_groups[media_col_c].astype(str)

        rows = []
        for module_name, sub in mod_groups.groupby(module_col, sort=False):
            media_list = sub[media_col_c].dropna().astype(str).tolist()
            sub_summary = echo_summary_df[echo_summary_df["Media Title"].astype(str).isin(media_list)]
            if not sub_summary.empty:
                avg_view_pct = sub_summary["Average View %"].mean(skipna=True)
                avg_students_viewing = sub_summary["# of Unique Viewers"].mean(skipna=True)
                overall_view_pct = sub_summary["% of Video Viewed Overall"].mean(skipna=True)
            else:
                avg_view_pct = np.nan
                avg_students_viewing = np.nan
                overall_view_pct = np.nan

            if students_col is not None and students_col in canvas_df.columns:
                mod_students_series = pd.to_numeric(canvas_df.loc[canvas_df[module_col].astype(str) == module_name, students_col], errors="coerce").dropna()
                if len(mod_students_series):
                    mode_vals = mod_students_series.mode()
                    mod_students = int(mode_vals.iloc[0]) if len(mode_vals) else int(round(mod_students_series.mean()))
                else:
                    mod_students = class_total_students if class_total_students else np.nan
            else:
                mod_students = class_total_students if class_total_students else np.nan

            rows.append({
                "Module": module_name,
                "Average View %": avg_view_pct,
                "# of Students Viewing": avg_students_viewing,
                "Overall View %": overall_view_pct,
                "# of Students": mod_students,
            })
        module_table = pd.DataFrame(rows, columns=module_table_cols)

    student_table = pd.DataFrame(columns=["Student","Final Grade","Average View % When Watched","View % of Total Video"]) if grades_df is None else None

    if grades_df is not None:
        gb = grades_df.copy()
        if "Student" in gb.columns:
            gb = gb[~gb["Student"].isin(["Points Possible","Student, Test"])].copy()
        for c in ["Final Grade","Current Grade","Unposted Final Grade","Final Score","Current Score","Unposted Final Score"]:
            if c in gb.columns:
                gb = gb[~gb[c].astype(str).str.contains("read only", case=False, na=False)]

        final_grade_col = None
        for c in ["Final Grade","Current Grade","Unposted Final Grade"]:
            if c in gb.columns:
                final_grade_col = c; break
        if final_grade_col is None:
            for c in ["Final Score","Current Score","Unposted Final Score"]:
                if c in gb.columns:
                    final_grade_col = c; break

        echo_name_col = None
        for c in ["User Name","user name","Student","student","Name","name"]:
            if c in echo_df.columns:
                echo_name_col = c; break

        if echo_name_col is not None:
            per_student = echo_df.groupby(echo_name_col, dropna=False).agg(
                avg_true_view_pct=("True View %", lambda x: pd.to_numeric(x, errors="coerce").mean()/100.0),
                sum_view_time=(total_view_time_col, "sum")
            )
        else:
            per_student = pd.DataFrame(columns=["avg_true_view_pct","sum_view_time"])

        total_video_seconds = pd.to_numeric(
            echo_summary_df.loc[echo_summary_df["Media Title"].astype(str) != "Grand Total", "Video Duration"], errors="coerce"
        ).sum()

        out_rows = []
        sid = 1
        gb_name_col = "Student" if "Student" in gb.columns else None
        for _, r in gb.iterrows():
            name = str(r[gb_name_col]) if gb_name_col else None
            final_val = r[final_grade_col] if final_grade_col else None
            avg_view_when_watched = None
            view_pct_of_total = None
            if name and name in per_student.index:
                avg_view_when_watched = per_student.loc[name, "avg_true_view_pct"]
                sum_view_time = per_student.loc[name, "sum_view_time"]
                if total_video_seconds and total_video_seconds > 0:
                    view_pct_of_total = (sum_view_time / total_video_seconds)
            out_rows.append({
                "Student": sid,
                "Final Grade": final_val,
                "Average View % When Watched": avg_view_when_watched,
                "View % of Total Video": view_pct_of_total,
            })
            sid += 1
        student_table = pd.DataFrame(out_rows, columns=["Student","Final Grade","Average View % When Watched","View % of Total Video"])

    return echo_summary_df, module_table, student_table, class_total_students

def process_gradebook_with_order(gradebook_df: pd.DataFrame, order_df: pd.DataFrame):
    df = gradebook_df.copy()
    mask = df.iloc[:, 0].astype(str).str.contains("Student, Test", na=False)
    df = df[~mask].reset_index(drop=True)

    to_drop = ["Student", "ID", "SIS User ID", "SIS Login ID", "Current Grade", "Unposted Current Grade", "Unposted Final Grade"]
    df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True, errors="ignore")

    drop_cols = []
    for col in df.columns:
        if col == "Final Grade":
            continue
        s = pd.to_numeric(df[col].iloc[2:], errors='coerce')
        if s.fillna(0).eq(0).all():
            drop_cols.append(col)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    df2 = gradebook_df.copy()
    df2 = df2[~df2.iloc[:,0].astype(str).str.contains("Student, Test", na=False)].reset_index(drop=True)

    def _clean_name(name):
        if isinstance(name, str):
            return re.sub(r"\s*\(\d+\)$", "", name).strip()
        return name

    df2.columns = [_clean_name(c) for c in df2.columns]
    points_row = df2.iloc[0]
    student_rows = df2.iloc[1:]
    total_students = len(student_rows)

    metrics = []
    if not order_df.empty and "assignment_title" in order_df.columns and "module" in order_df.columns:
        for _, row in order_df.iterrows():
            assignment = row["assignment_title"]
            module = row["module"]
            if isinstance(assignment, str) and assignment.lower() == "none":
                continue
            if assignment not in df2.columns:
                continue
            points_possible = pd.to_numeric(points_row[assignment], errors="coerce")
            if not points_possible or points_possible == 0:
                continue
            grades = pd.to_numeric(student_rows[assignment], errors="coerce").fillna(0)
            avg_excl_zeros = grades[grades > 0].mean() / points_possible if (grades > 0).any() else 0
            pct_turned_in = 1 - (grades.eq(0).sum() / total_students)
            metrics.append([module, assignment, avg_excl_zeros, pct_turned_in])
    module_summary = pd.DataFrame(metrics, columns=["Module","Assignment","Average Excluding Zeros","% Turned In"]) if metrics else pd.DataFrame(columns=["Module","Assignment","Average Excluding Zeros","% Turned In"])

    if not module_summary.empty:
        summary = (
            module_summary.groupby("Module", sort=False)[["Average Excluding Zeros","% Turned In"]]
            .mean().reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["Module","Average Excluding Zeros","% Turned In"])

    return df, summary

# ---------- Sidebar (secure) ----------
with st.sidebar:
    st.header("Inputs")
    st.subheader("1) Canvas Materials Order")
    DEFAULT_BASE = st.secrets.get("canvas", {}).get("BASE_URL") or os.environ.get("CANVAS_BASE_URL") or "https://colostate.instructure.com"
    SERVER_TOKEN = st.secrets.get("canvas", {}).get("API_TOKEN") or os.environ.get("CANVAS_API_TOKEN")

    canvas_mode = st.radio("Canvas order source", ["Use API", "Upload CSV"], horizontal=True)

    if canvas_mode == "Use API":
        use_secret = st.toggle("Use server secret", value=bool(SERVER_TOKEN), help="If off, paste a token for this session only.")
        if use_secret and SERVER_TOKEN:
            base_url = DEFAULT_BASE
            api_token = SERVER_TOKEN
            st.caption("Using server-side secret.")
        else:
            base_url = st.text_input("Canvas Base URL", value=DEFAULT_BASE)
            api_token = st.text_input("Canvas API Token", type="password")
        course_id = st.text_input("Course ID", placeholder="e.g., 213019")
    else:
        base_url = None
        api_token = None
        course_id = None

    canvas_csv_upl = None
    if canvas_mode == "Upload CSV":
        canvas_csv_upl = st.file_uploader("Upload canvas_order.csv", type=["csv"], key="canvas_upl")

    st.divider()
    st.subheader("2) Echo360 CSV")
    echo_upl = st.file_uploader("Upload Echo360 CSV", type=["csv"], key="echo_upl")

    st.subheader("3) Gradebook CSV")
    grade_upl = st.file_uploader("Upload Gradebook CSV", type=["csv"], key="grade_upl")

@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

canvas_df = None
if canvas_mode == "Use API":
    if st.sidebar.button("Fetch Canvas Order", type="primary", use_container_width=True):
        if not base_url or not api_token or not course_id:
            st.sidebar.error("Base URL, API token, and Course ID are required.")
        else:
            with st.spinner("Fetching modules & assignments from Canvas…"):
                try:
                    canvas_df = fetch_canvas_order_df(base_url.strip().rstrip("/"), api_token.strip(), course_id.strip())
                    st.session_state["canvas_df"] = canvas_df
                    st.sidebar.success("Canvas order loaded from API.")
                except Exception:
                    st.sidebar.error("Canvas fetch failed. (See server logs for details)")
else:
    if canvas_csv_upl is not None:
        try:
            canvas_df = _read_csv(canvas_csv_upl)
            st.session_state["canvas_df"] = canvas_df
            st.sidebar.success("Canvas order loaded from CSV.")
        except Exception as e:
            st.sidebar.error(f"Canvas CSV read error: {e}")

canvas_df = st.session_state.get("canvas_df")

echo_df = _read_csv(echo_upl) if echo_upl is not None else None
grade_df = _read_csv(grade_upl) if grade_upl is not None else None

# ---------- Compute ----------
echo_summary_df = module_table = student_table = None
class_total_students = None
if echo_df is not None and canvas_df is not None:
    with st.spinner("Processing Echo + Gradebook with Canvas order…"):
        try:
            echo_summary_df, module_table, student_table, class_total_students = process_echo_with_order(echo_df, canvas_df, grade_df)
        except Exception as e:
            st.error(f"Processing error: {e}")

cleaned_gradebook_df = grade_module_summary = None
if grade_df is not None and canvas_df is not None:
    with st.spinner("Cleaning gradebook and building module summary…"):
        try:
            cleaned_gradebook_df, grade_module_summary = process_gradebook_with_order(grade_df, canvas_df)
        except Exception as e:
            st.error(f"Gradebook processing error: {e}")

# ---------- KPIs ----------
with st.container():
    st.subheader("Numbers at a glance")
    total_students = None
    if class_total_students:
        total_students = class_total_students
    elif grade_df is not None and "Student" in grade_df.columns:
        total_students = int((grade_df["Student"].astype(str).ne("Points Possible") & grade_df["Student"].astype(str).ne("Student, Test")).sum())

    avg_echo_view = None
    if echo_summary_df is not None and not echo_summary_df.empty:
        s = echo_summary_df.loc[echo_summary_df["Media Title"].astype(str) != "Grand Total", "Average View %"].dropna()
        if len(s):
            avg_echo_view = float(s.mean() * 100)

    pct_students_viewing = None
    if echo_summary_df is not None and "% of Students Viewing" in echo_summary_df.columns:
        s2 = echo_summary_df.loc[echo_summary_df["Media Title"].astype(str) != "Grand Total", "% of Students Viewing"].dropna()
        if len(s2):
            pct_students_viewing = float(s2.mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", value=(total_students if total_students is not None else "—"))
    c2.metric("Avg Echo View %", value=(f"{avg_echo_view:.1f}%" if avg_echo_view is not None else "—"))
    c3.metric("% Students Viewing", value=(f"{pct_students_viewing:.1f}%" if pct_students_viewing is not None else "—"))
    c4.metric("Media Count", value=(int((echo_summary_df["Media Title"].astype(str) != "Grand Total").sum()) if echo_summary_df is not None else "—"))

st.divider()

# ---------- Tables & Charts ----------
(tab_tables, tab_charts) = st.tabs(["Tables", "Charts"])

with tab_tables:
    t1, t2, t3, t4 = st.tabs(["Canvas Order", "Echo Summary", "Module Table", "Gradebook (clean)"])

    with t1:
        if canvas_df is not None:
            st.dataframe(canvas_df, use_container_width=True)
            st.download_button("Download canvas_order.csv", canvas_df.to_csv(index=False), "canvas_order.csv")
        else:
            st.caption("Load Canvas order from API or CSV.")

    with t2:
        if echo_summary_df is not None:
            show_df = echo_summary_df.copy()
            for col in ["Average View %","% of Students Viewing","% of Video Viewed Overall"]:
                if col in show_df.columns:
                    show_df[col] = pd.to_numeric(show_df[col], errors="coerce")
                    show_df[col] = show_df[col].map(lambda x: f"{x:.1%}" if pd.notnull(x) else "")
            st.dataframe(show_df, use_container_width=True)
            st.download_button("Download echo_summary.csv", echo_summary_df.to_csv(index=False), "echo_summary.csv")
        else:
            st.caption("Upload Echo + Canvas order to see this table.")

    with t3:
        if module_table is not None and not module_table.empty:
            show_mod = module_table.copy()
            for col in ["Average View %","Overall View %"]:
                if col in show_mod.columns:
                    show_mod[col] = pd.to_numeric(show_mod[col], errors="coerce").map(lambda x: f"{x:.1%}" if pd.notnull(x) else "")
            st.dataframe(show_mod, use_container_width=True)
            st.download_button("Download module_table.csv", module_table.to_csv(index=False), "module_table.csv")
        else:
            st.caption("Module rollup appears once Echo + Canvas order are loaded.")

    with t4:
        if cleaned_gradebook_df is not None:
            st.dataframe(cleaned_gradebook_df, use_container_width=True)
            st.download_button("Download gradebook_cleaned.csv", cleaned_gradebook_df.to_csv(index=False), "gradebook_cleaned.csv")
        else:
            st.caption("Upload Gradebook CSV to see the cleaned table.")

with tab_charts:
    st.subheader("Chart A · Module coverage & viewing")
    if module_table is not None and not module_table.empty:
        try:
            import altair as alt
            mt = module_table.copy()
            mt["students_viewing"] = pd.to_numeric(mt["# of Students Viewing"], errors="coerce")
            mt["students_total"] = pd.to_numeric(mt["# of Students"], errors="coerce")
            mt["students_not_viewing"] = mt["students_total"] - mt["students_viewing"]
            mt["avg_view"] = pd.to_numeric(mt["Average View %"], errors="coerce")
            mt["overall_view"] = pd.to_numeric(mt["Overall View %"], errors="coerce")

            base = alt.Chart(mt).encode(x=alt.X("Module:N", sort=None))

            bars1 = alt.Chart(mt).mark_bar().encode(
                y=alt.Y("students_viewing:Q", stack="zero", title="Students"),
                tooltip=["Module","students_viewing","students_total"]
            )
            bars2 = alt.Chart(mt).mark_bar().encode(
                y=alt.Y("students_not_viewing:Q", stack="zero", title="Students"),
                tooltip=["Module","students_not_viewing","students_total"]
            )
            line1 = base.mark_line(point=True).encode(
                y=alt.Y("avg_view:Q", axis=alt.Axis(title="Percent", format=".0%"), scale=alt.Scale(domain=[0,1])),
                tooltip=["Module", alt.Tooltip("avg_view:Q", format=".0%")]
            )
            line2 = base.mark_line(point=True).encode(
                y=alt.Y("overall_view:Q", axis=alt.Axis(title=None), scale=alt.Scale(domain=[0,1])),
                tooltip=["Module", alt.Tooltip("overall_view:Q", format=".0%")]
            )

            chartA = alt.layer(bars1, bars2, line1, line2).resolve_scale(y='independent').properties(height=360)
            st.altair_chart(chartA, use_container_width=True)
        except Exception as e:
            st.error(f"Altair failed to render Chart A: {e}")
    else:
        st.caption("Load Echo + Canvas order to render Chart A.")

    st.subheader("Chart B · Gradebook module averages")
    if 'grade_module_summary' in globals() and grade_module_summary is not None and not grade_module_summary.empty:
        try:
            import altair as alt
            gms = grade_module_summary.copy()
            gms["Average Excluding Zeros"] = pd.to_numeric(gms["Average Excluding Zeros"], errors="coerce")
            gms["% Turned In"] = pd.to_numeric(gms["% Turned In"], errors="coerce")
            base = alt.Chart(gms).encode(x=alt.X("Module:N", sort=None))
            lineA = base.mark_line(point=True).encode(y=alt.Y("Average Excluding Zeros:Q", axis=alt.Axis(format=".0%", title="Percent")))
            lineB = base.mark_line(point=True).encode(y=alt.Y("% Turned In:Q", axis=alt.Axis(format=".0%", title=None)))
            chartB = alt.layer(lineA, lineB).resolve_scale(y='independent').properties(height=320)
            st.altair_chart(chartB, use_container_width=True)
        except Exception as e:
            st.error(f"Altair failed to render Chart B: {e}")
    else:
        st.caption("Upload Gradebook + Canvas order to render Chart B.")

st.markdown("---")
st.caption("Secrets best-practice: add BASE_URL and API_TOKEN under [canvas] in .streamlit/secrets.toml or as environment variables. Tokens are never logged or cached.")
