import re
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import uuid

def clickable_kpi_card(label: str, value, on_click_key: str, color="#f0f2f6"):
    """Renders a KPI 'card' that behaves like a button. Sets st.session_state[on_click_key] to True when clicked."""
    card_id = f"card_{uuid.uuid4().hex}"       # unique DOM id
    form_key = f"form_{card_id}"               # unique Streamlit form key

    # The visual card
    st.markdown(f"""
        <div onclick="document.getElementById('{card_id}').click()" style="
            cursor: pointer;
            padding: 1rem;
            background-color: {color};
            border-radius: 0.75rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            margin-bottom: 0.5rem;
            height: 110px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: box-shadow 120ms ease, transform 120ms ease;
        " onmouseover="this.style.boxShadow='0 4px 16px rgba(0,0,0,0.10)'; this.style.transform='translateY(-2px)';"
          onmouseout="this.style.boxShadow='0 2px 10px rgba(0,0,0,0.06)'; this.style.transform='translateY(0)';">
            <div style="font-size: 0.85rem; font-weight: 600; color: #6c757d;">{label}</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #000000;">{value}</div>
        </div>
        <form>
            <input type="submit" id="{card_id}" style="display:none;">
        </form>
    """, unsafe_allow_html=True)

    # Hidden form to capture the click safely in Streamlit
    with st.form(key=form_key):
        if st.form_submit_button(label=""):  # invisible submit
            st.session_state[on_click_key] = True

# -------------------- Page Setup --------------------
st.set_page_config(page_title="Email Bot Evaluator", layout="wide")

# ---- Make buttons look like KPI cards ----
st.markdown("""
<style>
/* All KPI buttons share this look */
div.stButton > button.kpi {
  width: 100%;
  text-align: left;
  border: 0;
  background: #f0f2f6;
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  height: 110px;
  transition: box-shadow .12s ease, transform .12s ease;
  white-space: pre-line;             /* honor \n in label */
  font-weight: 700;
  font-size: 1.05rem;
  color: #111;
}
div.stButton > button.kpi:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.10);
  transform: translateY(-2px);
}
/* Lighter label line inside button */
button.kpi span.kpi-label {
  display: block;
  font-size: .85rem;
  font-weight: 600;
  color: #6c757d;
  margin-bottom: .25rem;
}
button.kpi span.kpi-value {
  display: block;
  font-size: 1.6rem;
  font-weight: 800;
  color: #111;
}
</style>
""", unsafe_allow_html=True)

def kpi_button(label, value, key):
    # Use \n to create two lines; CSS will handle layout
    # NOTE: Streamlit doesn't support HTML inside button text, so we mimic with two lines.
    return st.button(f"{label}\n\n{value}", key=key, use_container_width=True)

# -------------------- Cached Data Loader --------------------
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

# -------------------- UI Helpers --------------------
def kpi_card(label, value, delta=None, color="#f0f2f6"):
    st.markdown(f"""
        <div style="
            padding: 1rem;
            background-color: {color};
            border-radius: 0.75rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            margin-bottom: 0.5rem;
            height: 110px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="font-size: 0.85rem; font-weight: 600; color: #6c757d;">
                {label}
            </div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #000000;">
                {value}
            </div>
            {f'<div style="font-size: 0.8rem; color: green;">+{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)

def highlight_entities(text, entity_dict):
    if isinstance(entity_dict, str):
        try:
            entity_dict = eval(entity_dict)
        except:
            return text
    if not isinstance(entity_dict, dict):
        return text

    spans = []
    for color, words in entity_dict.items():
        for word in set(words):
            if not word.strip():
                continue
            color_code = "#d4edda" if color == "green" else "#f8d7da"
            text_color = "green" if color == "green" else "red"
            escaped_word = re.escape(word)
            replacement = f"<span style='font-weight:bold; background-color:{color_code}; color:{text_color};'>{word}</span>"
            spans.append((escaped_word, replacement))

    spans.sort(key=lambda x: len(x[0]), reverse=True)
    for escaped_word, replacement in spans:
        text = re.sub(escaped_word, replacement, text, flags=re.IGNORECASE)
    return text

def compute_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)

# -------------------- State (init BEFORE widgets) --------------------
if "detail_filter" not in st.session_state:
    st.session_state["detail_filter"] = None  # pass/fail/cat_pass/cat_fail/ent_pass/ent_fail
if "main_filter" not in st.session_state:
    st.session_state["main_filter"] = None    # 'pass' or 'fail' from Row 1 buttons
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "📊 Overview"  # internal page state
if "jump_to_detail" not in st.session_state:
    st.session_state["jump_to_detail"] = False  # programmatic redirect trigger

# consume programmatic redirect BEFORE nav widget is created
if st.session_state.get("jump_to_detail"):
    st.session_state["active_page"] = "📋 Detailed Table"
    st.session_state["jump_to_detail"] = False

# -------------------- Data --------------------
CSV_PATH = "procurement_emails_with_slm_score.csv"
df = load_data(CSV_PATH)

st.title("📧 Automated Email Processing Evaluation Suite")

# -------------------- Sidebar Filters --------------------
with st.sidebar:
    st.header("Filters")
    st.markdown("### Date Range")

    date_option = st.selectbox(
        "Quick Ranges",
        options=[
            "All Time", "Today", "Yesterday", "Last 7 Days", "Last 30 Days",
            "Past Go-Live Phase 1", "Past Go-Live Phase 2", "Custom Range"
        ],
        index=0
    )

    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    today = pd.to_datetime("today").normalize()

    # Go-Live dates (adjust if needed)
    go_live_phase1 = pd.to_datetime("2025-06-01")
    go_live_phase2 = pd.to_datetime("2025-07-20")

    if date_option == "All Time":
        start_date, end_date = min_date, max_date
    elif date_option == "Today":
        start_date = end_date = today
    elif date_option == "Yesterday":
        start_date = end_date = today - pd.Timedelta(days=1)
    elif date_option == "Last 7 Days":
        end_date = today
        start_date = end_date - pd.Timedelta(days=7)
    elif date_option == "Last 30 Days":
        end_date = today
        start_date = end_date - pd.Timedelta(days=30)
    elif date_option == "Past Go-Live Phase 1":
        start_date = go_live_phase1
        end_date = max_date
    elif date_option == "Past Go-Live Phase 2":
        start_date = go_live_phase2
        end_date = max_date
    elif date_option == "Custom Range":
        start_date = st.date_input("Start Date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
        end_date = st.date_input("End Date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

    # Normalize SLM score if it was in [0,100]
    if "slm_score" in df_filtered.columns and df_filtered["slm_score"].max() > 1.01:
        df_filtered["slm_score"] = df_filtered["slm_score"] / 100

    # Category filter (procurement_function)
    if "procurement_function" in df_filtered.columns:
        all_functions = sorted(df_filtered["procurement_function"].dropna().unique())
        selected_functions = st.multiselect("Category", options=["All"] + all_functions, default=["All"])
        if "All" not in selected_functions:
            df_filtered = df_filtered[df_filtered["procurement_function"].isin(selected_functions)]

# -------------------- Derived Metrics --------------------
# Binary category correctness
df_filtered["category_correct"] = (df_filtered["category"] == df_filtered["category_identified"]).astype(int)

# Entity accuracy per email
df_filtered["entity_accuracy"] = df_filtered.apply(
    lambda row: row["number_of_entities_identified"] / row["number_of_entities"]
    if row["number_of_entities"] > 0 else 0,
    axis=1
)

# Email-level accuracy = (category_correct + entity_accuracy) / 2
df_filtered["email_accuracy"] = (df_filtered["category_correct"] + df_filtered["entity_accuracy"]) / 2

# -------------------- NAV (radio we can control) --------------------
PAGES = ["📊 Overview", "📋 Detailed Table", "🛠️ Technical View"]
st.markdown("### View")
nav = st.radio(
    label="Select view",
    options=PAGES,
    horizontal=True,
    key="nav_radio",
    index=PAGES.index(st.session_state.get("active_page", "📊 Overview")),
)
# keep internal page state synced
if nav != st.session_state["active_page"]:
    st.session_state["active_page"] = nav

# -------------------- OVERVIEW --------------------
if st.session_state["active_page"] == "📊 Overview":
    st.subheader("📊 Email Classification Summary")
    left_col, right_col = st.columns([1, 2])

    # Left: Pie chart
    with left_col:
        correct = int(df_filtered["is_correct"].sum()) if "is_correct" in df_filtered.columns else 0
        incorrect = int(len(df_filtered) - correct)
        pie_data = pd.DataFrame({
            "Classification": ["Correctly Classified", "Incorrectly Classified"],
            "Count": [correct, incorrect]
        })
        fig_pie = px.pie(pie_data, values="Count", names="Classification", hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Right: KPIs + Drilldown
    with right_col:
        # Totals
        total_emails = len(df_filtered)
        correct_emails = int(df_filtered["is_correct"].sum()) if "is_correct" in df_filtered.columns else 0
        incorrect = total_emails - correct_emails

        total_categories = total_emails
        correct_categories = int((df_filtered["category"] == df_filtered["category_identified"]).sum())
        total_entities = int(df_filtered["number_of_entities"].sum())
        correct_entities = int(df_filtered["number_of_entities_identified"].sum())

        avg_llm_score = df_filtered["llm_judge_score"].mean() if "llm_judge_score" in df_filtered.columns else np.nan
        avg_slm_accuracy = df_filtered["slm_score"].mean() if "slm_score" in df_filtered.columns else np.nan
        category_accuracy = (avg_llm_score + avg_slm_accuracy) / 2 if pd.notna(avg_llm_score) and pd.notna(avg_slm_accuracy) else np.nan
        avg_accuracy = df_filtered["email_accuracy"].mean()

        # Row 1 – Totals (KPI cards only; not clickable)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            kpi_card("📬 Total Emails Processed", total_emails)
        with r1c2:
            kpi_card("✅ Emails Passed", correct_emails)
        with r1c3:
            kpi_card("❌ Emails Failed", incorrect)
        with r1c4:
            success_rate = correct_emails / total_emails if total_emails else 0
            kpi_card("📊 Success Rate", f"{success_rate:.2%}")

        # Spacer
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 2 – Accuracy KPIs
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            kpi_card("📈 Email Accuracy", f"{avg_accuracy:.2%}")
        with r2c2:
            kpi_card("🏷️ Category Accuracy", "N/A" if pd.isna(category_accuracy) else f"{category_accuracy:.2%}")
        with r2c3:
            kpi_card("🔢 Entity Accuracy", f"{(correct_entities / total_entities):.2%}" if total_entities else "N/A")

        # ---- Breakdown controls (buttons at bottom row) ----
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns([1,1,4])
        with bc1:
            if st.button("Show Pass Breakdown", use_container_width=True):
                st.session_state["main_filter"] = "pass"
        with bc2:
            if st.button("Show Fail Breakdown", use_container_width=True):
                st.session_state["main_filter"] = "fail"
        # optional clear
        with bc3:
            if st.button("Clear Breakdown", use_container_width=False):
                st.session_state["main_filter"] = None
        # ---- Styled breakdown panels (appear based on main_filter) ----
        if st.session_state.get("main_filter") == "pass":
            st.markdown("""
            <div style="border:1px solid #e5e7eb;background:#fafbff;border-radius:12px;
                        padding:1rem 1rem .5rem 1rem;margin-top:.25rem;">
            <div style="font-weight:700;margin-bottom:.75rem;">✅ Pass Breakdown</div>
            """, unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                cat_pass_rate = (correct_categories / total_categories) if total_categories else 0
                kpi_card("🏷 Category Pass Rate", f"{cat_pass_rate:.2%}")
                if st.button("Show Emails", key="btn_cat_pass", use_container_width=True):
                    st.session_state["detail_filter"] = "cat_pass"
                    st.session_state["jump_to_detail"] = True
                    st.rerun()
            with pc2:
                ent_pass_rate = (correct_entities / total_entities) if total_entities else 0
                kpi_card("🔢 Entity Pass Rate", f"{ent_pass_rate:.2%}" if total_entities else "N/A")
                if st.button("Show Emails", key="btn_ent_pass", use_container_width=True):
                    st.session_state["detail_filter"] = "ent_pass"
                    st.session_state["jump_to_detail"] = True
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.get("main_filter") == "fail":
            st.markdown("""
            <div style="border:1px solid #e5e7eb;background:#fff7f7;border-radius:12px;
                        padding:1rem 1rem .5rem 1rem;margin-top:.25rem;">
            <div style="font-weight:700;margin-bottom:.75rem;">❌ Fail Breakdown</div>
            """, unsafe_allow_html=True)

            fc1, fc2 = st.columns(2)
            with fc1:
                fail_cat_rate = ((total_categories - correct_categories) / total_categories) if total_categories else 0
                kpi_card("🏷 Category Fail Rate", f"{fail_cat_rate:.2%}" if total_categories else "N/A")
                if st.button("Show Emails", key="btn_cat_fail", use_container_width=True):
                    st.session_state["detail_filter"] = "cat_fail"
                    st.session_state["jump_to_detail"] = True
                    st.rerun()
            with fc2:
                fail_ent_rate = ((total_entities - correct_entities) / total_entities) if total_entities else 0
                kpi_card("🔢 Entity Fail Rate", f"{fail_ent_rate:.2%}" if total_entities else "N/A")
                if st.button("Show Emails", key="btn_ent_fail", use_container_width=True):
                    st.session_state["detail_filter"] = "ent_fail"
                    st.session_state["jump_to_detail"] = True
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- DETAILED TABLE --------------------
elif st.session_state["active_page"] == "📋 Detailed Table":
    st.subheader("Detailed Table")

    mode = st.session_state.get("detail_filter")

    # Build filtered subset per mode
    if mode == "cat_pass":
        df2 = df_filtered[df_filtered["category"] == df_filtered["category_identified"]].copy()
        badge = "Category PASS (identified matches actual)"
    elif mode == "cat_fail":
        df2 = df_filtered[df_filtered["category"] != df_filtered["category_identified"]].copy()
        badge = "Category FAIL (identified ≠ actual)"
    elif mode == "ent_pass":
        df2 = df_filtered[(df_filtered["number_of_entities"] > 0) &
                          (df_filtered["number_of_entities_identified"] == df_filtered["number_of_entities"])].copy()
        badge = "Entity PASS (all entities found)"
    elif mode == "ent_fail":
        df2 = df_filtered[(df_filtered["number_of_entities"] > 0) &
                          (df_filtered["number_of_entities_identified"] < df_filtered["number_of_entities"])].copy()
        badge = "Entity FAIL (missed entities)"
    elif mode == "pass":
        df2 = df_filtered[df_filtered["is_correct"] == 1].copy()
        badge = "Email PASS (is_correct = 1)"
    elif mode == "fail":
        df2 = df_filtered[df_filtered["is_correct"] == 0].copy()
        badge = "Email FAIL (is_correct = 0)"
    else:
        df2 = df_filtered.copy()
        badge = "No breakdown filter applied"

    # Badge + clear
    col_badge, col_clear = st.columns([4,1])
    with col_badge:
        st.markdown(
            f"<div style='display:inline-block;padding:.35rem .6rem;border-radius:.5rem;"
            f"background:#eef3ff;border:1px solid #d6defc;font-size:.9rem;'>{badge}</div>",
            unsafe_allow_html=True
        )

    with col_clear:
        st.markdown("""
            <style>
            div[data-testid="stButton"] > button {
                background-color: #2e86de;  /* Nice blue */
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.25rem 0.4rem; /* smaller size */
                font-size: 0.85rem;        /* slightly smaller text */
                font-weight: 600;
                min-height: 1.8rem;
            }
            div[data-testid="stButton"] > button:hover {
                background-color: #1b4f72;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("Clear Filter", use_container_width=True):
            st.session_state["detail_filter"] = None
            st.session_state["main_filter"] = None
            st.rerun()

    # Table build
    df2["sl_number"] = range(1, len(df2) + 1)
    df2["from"] = df2["from_email"]
    df2["to"] = df2["to_email"]
    table_cols = [
        "sl_number", "timestamp", "from", "to", "subject",
        "category", "category_identified", "pred_confidence",
        "number_of_entities", "number_of_entities_identified",
        "entities_identified", "cluster_id", "llm_judge_score", "slm_score"
    ]
    df_display = df2[table_cols].copy()

    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection("single", use_checkbox=False)
    gb.configure_default_column(filter=True, sortable=True, resizable=True, wrapText=True, autoHeight=True)
    grid_options = gb.build()

    grid_response = AgGrid(
        df_display, gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="balham", height=400, fit_columns_on_grid_load=True
    )
    selected = grid_response.get("selected_rows", [])
    if isinstance(selected, pd.DataFrame):
        selected = selected.to_dict("records")

    if selected:
        row = selected[0]
        email_row = df2[df2["sl_number"] == row["sl_number"]].iloc[0]
        st.markdown("----")
        st.markdown(f"### 📧 Full Email - Subject: `{email_row['subject']}`")
        st.markdown(f"**From:** `{email_row['from']}`")
        st.markdown(f"**To:** `{email_row['to']}`")
        st.markdown(f"**Category:** `{email_row['category']}`  |  **Identified:** `{email_row['category_identified']}`")
        st.markdown("**Email Body:**")
        highlighted = highlight_entities(email_row["original_email"], email_row["entities_identified"])
        st.markdown(
            f"<div style='background-color: #f0f0f0; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; white-space: pre-wrap; font-family: sans-serif; font-size: 0.95rem; line-height: 1.6;'>{highlighted}</div>",
            unsafe_allow_html=True
        )

    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download Table as CSV", data=csv, file_name="detailed_emails.csv", mime="text/csv")

# -------------------- TECHNICAL VIEW --------------------
elif st.session_state["active_page"] == "🛠️ Technical View":
    st.subheader("Model Evaluation")

    selected_categories = sorted(df_filtered["category"].dropna().unique())
    if not selected_categories:
        st.warning("⚠️ No categories found after filter.")
    else:
        # Confusion Matrices
        st.markdown("#### Confusion Matrix per Selected Category")
        matrix_cols = st.columns(2)
        col_idx = 0
        for i, cat in enumerate(selected_categories):
            df_cat = df_filtered[df_filtered["category"] == cat]
            if df_cat.empty:
                continue

            y_true = df_cat["category"]
            y_pred = df_cat["category_identified"]

            labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
            if not any(label in y_true.values for label in labels):
                continue

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues", x=labels, y=labels,
                labels=dict(x="Predicted", y="Actual", color="Count")
            )
            fig_cm.update_layout(title=f"Confusion Matrix: {cat}", height=400)
            matrix_cols[col_idx].plotly_chart(fig_cm, use_container_width=True)

            col_idx = (col_idx + 1) % 2
            if col_idx == 0 and i < len(selected_categories) - 1:
                matrix_cols = st.columns(2)

        # ROC Curves
        st.markdown("#### ROC Curve per Selected Category")
        roc_cols = st.columns(2)
        col_idx = 0
        for i, cat in enumerate(selected_categories):
            df_cat = df_filtered[
                (df_filtered["category"] == cat) |
                (df_filtered["category_identified"] == cat)
            ]
            if df_cat.empty:
                continue

            y_true = (df_cat["category"] == cat).astype(int)
            y_score = (df_cat["category_identified"] == cat).astype(int) * df_cat["pred_confidence"]

            if y_true.nunique() < 2:
                roc_cols[col_idx].warning(f"Not enough data for {cat}")
                col_idx = (col_idx + 1) % 2
                if col_idx == 0 and i < len(selected_categories) - 1:
                    roc_cols = st.columns(2)
                continue

            fpr, tpr, roc_auc = compute_roc(y_true, y_score)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
            fig_roc.update_layout(title=f"ROC Curve - {cat}", xaxis_title="FPR", yaxis_title="TPR", height=400)
            roc_cols[col_idx].plotly_chart(fig_roc, use_container_width=True)

            col_idx = (col_idx + 1) % 2
            if col_idx == 0 and i < len(selected_categories) - 1:
                roc_cols = st.columns(2)
