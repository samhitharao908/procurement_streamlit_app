import re
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

st.set_page_config(page_title="Email Bot Evaluator", layout="wide")

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def kpi_card(label, value, delta=None, color="#f0f2f6"):
    st.markdown(f"""
        <div style="
            padding: 1rem;
            background-color: {color};
            border-radius: 0.5rem;
            box-shadow: 0 0 5px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="font-size: 0.85rem; font-weight: 600; color: #6c757d;">
                {label}
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #000000;">
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

# Load data
CSV_PATH = "procurement_emails_5000_full.csv"
df = load_data(CSV_PATH)

st.title("📧 Email Bot Evaluator Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    st.markdown("### Date Range")
    date_option = st.selectbox(
        "Quick Ranges",
        options=["All Time", "Yesterday", "Last 7 Days", "Last 30 Days", "Last 3 Months", "Last 6 Months", "Last 1 Year", "Custom Range"],
        index=0
    )

    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    if date_option == "All Time":
        start_date, end_date = min_date, max_date
    elif date_option == "Yesterday":
        start_date = end_date = pd.to_datetime("today").normalize() - pd.Timedelta(days=1)
    elif date_option == "Last 7 Days":
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.Timedelta(days=7)
    elif date_option == "Last 30 Days":
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.Timedelta(days=30)
    elif date_option == "Last 3 Months":
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.DateOffset(months=3)
    elif date_option == "Last 6 Months":
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.DateOffset(months=6)
    elif date_option == "Last 1 Year":
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.DateOffset(years=1)
    elif date_option == "Custom Range":
        start_date = st.date_input("Start Date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
        end_date = st.date_input("End Date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

    all_functions = sorted(df_filtered["procurement_function"].dropna().unique())
    selected_functions = st.multiselect("Procurement Category", options=["All"] + all_functions, default=["All"])

    if "All" not in selected_functions:
        df_filtered = df_filtered[df_filtered["procurement_function"].isin(selected_functions)]

# Tabs
tab1, tab2 = st.tabs(["📊 Overview", "📋 Detailed Table"])

with tab1:
    st.subheader("📊 Email Classification Summary")
    left_col, right_col = st.columns([1, 2])

    with left_col:
        correct = int(df_filtered["is_correct"].sum())
        incorrect = int(len(df_filtered) - correct)

        pie_data = pd.DataFrame({
            "Classification": ["Correctly Classified", "Incorrectly Classified"],
            "Count": [correct, incorrect]
        })

        fig_pie = px.pie(pie_data, values="Count", names="Classification", hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with right_col:
        total_emails = len(df_filtered)
        correct_emails = correct
        total_categories = total_emails
        correct_categories = int((df_filtered["category"] == df_filtered["category_identified"]).sum())
        total_entities = int(df_filtered["number_of_entities"].sum())
        correct_entities = int(df_filtered["number_of_entities_identified"].sum())

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1: kpi_card("📬 Total Emails", total_emails)
        with r1c2: kpi_card("🏷️ Total Categories", total_categories)
        with r1c3: kpi_card("🔢 Total Entities", total_entities)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1: kpi_card("✅ Correct Emails", correct_emails)
        with r2c2: kpi_card("🎯 Correct Categories", correct_categories)
        with r2c3: kpi_card("📌 Correct Entities", correct_entities)

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1: kpi_card("📈 Email Accuracy", f"{correct_emails / total_emails:.2%}" if total_emails else "N/A")
        with r3c2: kpi_card("📊 Category Accuracy", f"{correct_categories / total_categories:.2%}" if total_categories else "N/A")
        with r3c3: kpi_card("📐 Entity Accuracy", f"{correct_entities / total_entities:.2%}" if total_entities else "N/A")

    # Confusion Matrices
    st.markdown("---")
    st.subheader("Confusion Matrix per Category")
    categories = sorted(df_filtered["category"].dropna().unique())
    matrix_cols = st.columns(2)
    col_idx = 0

    for i, cat in enumerate(categories):
        df_cat = df_filtered[df_filtered["category"] == cat]
        if df_cat.empty:
            continue

        y_true = df_cat["category"]
        y_pred = df_cat["category_identified"]
        labels = sorted(set(y_true.unique()).union(y_pred.unique()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=labels, y=labels,
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        fig_cm.update_layout(title=f"Confusion Matrix: {cat}", height=400)
        matrix_cols[col_idx].plotly_chart(fig_cm, use_container_width=True)

        col_idx = (col_idx + 1) % 2
        if col_idx == 0 and i < len(categories) - 1:
            matrix_cols = st.columns(2)

    # ROC Curves
    st.subheader("ROC Curve per Category")
    roc_cols = st.columns(2)
    col_idx = 0

    for i, cat in enumerate(categories):
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
            if col_idx == 0 and i < len(categories) - 1:
                roc_cols = st.columns(2)
            continue

        fpr, tpr, roc_auc = compute_roc(y_true, y_score)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig_roc.update_layout(title=f"ROC Curve - {cat}", xaxis_title="FPR", yaxis_title="TPR", height=400)
        roc_cols[col_idx].plotly_chart(fig_roc, use_container_width=True)

        col_idx = (col_idx + 1) % 2
        if col_idx == 0 and i < len(categories) - 1:
            roc_cols = st.columns(2)

# Tab 2: Detailed Table
with tab2:
    st.subheader("Detailed Table")
    df2 = df_filtered.copy()
    df2["sl_number"] = range(1, len(df2) + 1)
    df2["from"] = df2["from_email"]
    df2["to"] = df2["to_email"]
    table_cols = [
        "sl_number", "timestamp", "from", "to", "subject",
        "category", "category_identified", "pred_confidence",
        "number_of_entities", "number_of_entities_identified",
        "entities_identified", "cluster_id", "llm_judge_score"
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
        st.markdown(f"**Category:** `{email_row['category']}`")
        st.markdown(f"**Identified Category:** `{email_row['category_identified']}`")
        st.markdown("**Email Body:**")

        highlighted = highlight_entities(email_row["original_email"], email_row["entities_identified"])
        st.markdown(
            f"<div style='background-color: #f0f0f0; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; white-space: pre-wrap; font-family: sans-serif; font-size: 0.95rem; line-height: 1.6;'>{highlighted}</div>",
            unsafe_allow_html=True
        )


    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download Table as CSV", data=csv, file_name="detailed_emails.csv", mime="text/csv")
