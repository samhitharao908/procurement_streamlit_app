# app.py
"""
COUPA Email Bot Evaluator
"""
import re
import html
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

st.set_page_config(page_title="COUPA Email Bot Evaluator", layout="wide")

INTENT_COL_MAP = {
    "payment": ("y_payment", "prob_payment"),
    "invoice": ("y_invoice", "prob_invoice"),
    "sourcing": ("y_sourcing", "prob_sourcing"),
    "contract": ("y_contract", "prob_contract"),
    "grn": ("y_grn", "prob_grn"),
    "dispute": ("y_dispute", "prob_dispute"),
    "reminder": ("y_reminder", "prob_reminder"),
    "clarification": ("y_clarification", "prob_clarification"),
    "purchase": ("y_purchase", "prob_purchase"),
}

CSV_PATH_DEFAULT = "procurement_emails_15k_with_entities_flagged.csv"

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
            <div style="font-size: 0.85rem; font-weight: 600; color: #6c757d; word-wrap: break-word;">
                {label}
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #000000;">
                {value}
            </div>
            {f'<div style="font-size: 0.8rem; color: green;">+{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)


def highlight_entities(text, entity_dict_str):
    try:
        entity_dict = eval(entity_dict_str) if isinstance(entity_dict_str, str) else entity_dict_str
    except Exception:
        return html.escape(text)  # fallback if eval fails

    text = html.escape(text)

    green_words = entity_dict.get("green", [])
    red_words = entity_dict.get("red", [])

    color_map = {word: "green" for word in green_words}
    color_map.update({word: "red" for word in red_words})

    # Sort to replace longer phrases first
    sorted_keywords = sorted(color_map.keys(), key=len, reverse=True)

    for word in sorted_keywords:
        escaped_word = re.escape(html.escape(word))
        colored = f"<span style='color:{color_map[word]}; font-weight:bold'>{word}</span>"
        text = re.sub(escaped_word, colored, text, flags=re.IGNORECASE)

    return text


@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required_cols = [
        "timestamp", "procurement_function",
        "intent", "intent_identified",
        "number_of_entities", "number_of_entities_identified",
        "overall_accuracy", "overall_macro_f1", "llm_judge_score", "cohen_kappa",
        "from_name", "from_email", "to_name", "to_email", "subject", "body",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"The following expected columns are missing in your CSV: {missing}")
    return df

def apply_date_filter(df, date_choice, start_date=None, end_date=None):
    now = pd.Timestamp.now()
    presets = {
        "Past 1 Day": 1,
        "Past 7 Days": 7,
        "Past 30 Days": 30,
        "Past 3 Months": 90,
        "Past 6 Months": 180,
    }
    if date_choice in presets:
        ed = now
        sd = now - pd.Timedelta(days=presets[date_choice])
    else:
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["timestamp"] >= sd) & (df["timestamp"] <= ed)]

def safe_mean(series):
    try:
        return float(series.dropna().mean())
    except Exception:
        return np.nan

def compute_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def roc_plot(fpr, tpr, roc_auc, title_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC = {roc_auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title=f"ROC Curve {title_suffix}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig

st.title("📧 COUPA Email Bot Evaluator Dashboard")

df = load_data(CSV_PATH_DEFAULT)

with st.sidebar:
    st.header("Filters")
    date_choice = st.selectbox(
        "Date Range",
        ["Past 1 Day", "Past 7 Days", "Past 30 Days", "Past 3 Months", "Past 6 Months", "Custom Range"],
        index=2,
    )
    if date_choice == "Custom Range":
        custom_dates = st.date_input("Custom Range (start, end)", value=(df["timestamp"].min(), df["timestamp"].max()))
        if not isinstance(custom_dates, (list, tuple)) or len(custom_dates) != 2:
            st.error("Please select a start and end date")
            st.stop()
        start_date, end_date = custom_dates
    else:
        start_date, end_date = None, None

    all_functions = sorted(df["procurement_function"].dropna().unique())
    pf_choice = st.multiselect("Procurement Function", options=["All"] + all_functions, default=["All"])

    all_intents = list(INTENT_COL_MAP.keys())
    select_all_intents = st.checkbox("Select All Intents", value=True)
    if select_all_intents:
        selected_intents = st.multiselect("Intent to Evaluate (CM & ROC)", options=all_intents, default=all_intents, key="intents_all")
    else:
        selected_intents = st.multiselect("Intent to Evaluate (CM & ROC)", options=all_intents, key="intents_custom")


if "All" in pf_choice:
    df_filtered = apply_date_filter(df, date_choice, start_date, end_date)
else:
    df_filtered = apply_date_filter(df[df["procurement_function"].isin(pf_choice)], date_choice, start_date, end_date)

tab1, tab2 = st.tabs(["📊 Overview (Sheet 1)", "📋 Detailed Table (Sheet 2)"])

with tab1:
    top_right_col1, top_right_col2 = st.columns([2, 1])

    with top_right_col1:
        st.subheader("Entity Identification")
        total_entities = df_filtered["number_of_entities"].sum()
        correct_entities = df_filtered["number_of_entities_identified"].sum()
        incorrect_entities = max(total_entities - correct_entities, 0)
        pie_data = pd.DataFrame({"Entity Match": ["Correctly Identified", "Incorrectly Identified"], "Count": [correct_entities, incorrect_entities]})
        fig_pie = px.pie(pie_data, values="Count", names="Entity Match", hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with top_right_col2:
        st.subheader("Key KPIs")
        total_emails = len(df_filtered)
        correct_intents = (df_filtered["intent"] == df_filtered["intent_identified"]).sum()
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Total Emails", total_emails)
        with k2: kpi_card("Correctly Identified Intents", correct_intents)
        with k3: kpi_card("Total Entities", int(total_entities))
        with k4: kpi_card("Correct Entities Identified", int(correct_entities))

        filtered_intent_rows = []

        for intent_eval in selected_intents:
            y_col, p_col = INTENT_COL_MAP[intent_eval]
            if y_col in df_filtered.columns and p_col in df_filtered.columns:
                filtered_intent_rows.append(df_filtered[[y_col, p_col, "overall_accuracy", "overall_macro_f1", "llm_judge_score", "cohen_kappa"]])

        if filtered_intent_rows:
            df_scores = pd.concat(filtered_intent_rows).drop_duplicates()
        else:
            df_scores = pd.DataFrame(columns=["overall_accuracy", "overall_macro_f1", "llm_judge_score", "cohen_kappa"])

        acc_mean = safe_mean(df_scores.get("overall_accuracy", pd.Series(dtype=float)))
        f1_mean = safe_mean(df_scores.get("overall_macro_f1", pd.Series(dtype=float)))
        llm_mean = safe_mean(df_scores.get("llm_judge_score", pd.Series(dtype=float)))
        kappa_mean = safe_mean(df_scores.get("cohen_kappa", pd.Series(dtype=float)))
        
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Avg Accuracy", f"{acc_mean:.3f}" if not np.isnan(acc_mean) else "NA")
        with k2: kpi_card("Avg F1 Score", f"{f1_mean:.3f}" if not np.isnan(f1_mean) else "NA")
        with k3: kpi_card("Avg LLM Score", f"{llm_mean:.3f}" if not np.isnan(llm_mean) else "NA")
        with k4: kpi_card("Avg Cohen Kappa", f"{kappa_mean:.3f}" if not np.isnan(kappa_mean) else "NA")

    st.markdown("---")
    st.subheader("Evaluation per Intent")

    for intent_eval in selected_intents:
        y_col, p_col = INTENT_COL_MAP[intent_eval]
        if y_col in df_filtered.columns and p_col in df_filtered.columns:
            y_true = df_filtered[y_col].dropna().astype(int).clip(0, 1)
            y_score = df_filtered.loc[y_true.index, p_col].fillna(0.0)

            if len(y_true.unique()) < 2:
                st.info(f"Not enough data for `{intent_eval}` to compute metrics.")
                continue

            cm = confusion_matrix(y_true, (y_score >= 0.5).astype(int), labels=[0, 1])
            fpr, tpr, roc_auc = compute_roc(y_true, y_score)
            fig_roc = roc_plot(fpr, tpr, roc_auc, title_suffix=f" - {intent_eval.title()}")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**{intent_eval.title()} - Confusion Matrix**")
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Negative", "Positive"],
                    y=["Negative", "Positive"],
                )
                fig_cm.update_layout(title=f"Confusion Matrix - {intent_eval.title()}")
                st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{intent_eval}")

            with col2:
                st.markdown(f"**{intent_eval.title()} - ROC Curve**")
                st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{intent_eval}")

with tab2:
    st.subheader("Detailed Email Table")
    df2 = df.copy()
    df2["sl_number"] = range(1, len(df2) + 1)
    df2["date"] = pd.to_datetime(df2["timestamp"]).dt.date if "timestamp" in df2.columns else pd.NaT
    df2["from"] = df2.get("from_name", "").astype(str) + " <" + df2.get("from_email", "").astype(str) + ">"
    df2["to"] = df2.get("to_name", "").astype(str) + " <" + df2.get("to_email", "").astype(str) + ">"

    table_cols = ["sl_number", "date", "from", "to", "subject", "intent", "intent_identified"]
    all_cols = table_cols + ["body"]
    df_display = df2[all_cols].copy()

    gb = GridOptionsBuilder.from_dataframe(df_display[table_cols])
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_default_column(filter="agTextColumnFilter", editable=False, enableRowGroup=True, enablePivot=True, enableValue=True, sortable=True, resizable=True)
    for col in table_cols:
        gb.configure_column(
            col,
            wrapText=True,
            autoHeight=True,
            resizable=True,
            sortable=True,
            filter=True
        )
    grid_options = gb.build()

    grid_response = AgGrid(
        df_display[table_cols],
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        height=400,
        fit_columns_on_grid_load=True,
    )

    selected_rows = grid_response.get("selected_rows", [])
    if not isinstance(selected_rows, list):
        try:
            selected_rows = pd.DataFrame(selected_rows).to_dict("records")
        except Exception:
            selected_rows = []

    if len(selected_rows) > 0:
        selected_sl = selected_rows[0].get("sl_number")
        match = df_display[df_display["sl_number"] == selected_sl]
        if not match.empty:
            row_data = match.iloc[0]
            st.markdown("---")
            with st.expander(f"📧 Full Email: {row_data.get('subject', '')}", expanded=True):
                st.markdown(f"**From:** {row_data.get('from', '')}")
                st.markdown(f"**To:** {row_data.get('to', '')}")
                st.markdown(f"**Intent / Identified:** {row_data.get('intent', '')} / {row_data.get('intent_identified', '')}")
                st.markdown("**Body:**")
                email_body = row_data.get("body", "No email body found.")
                entity_info = df2.loc[df2["sl_number"] == selected_sl, "entities_identified"].values[0]
                highlighted = highlight_entities(email_body, entity_info)
                st.markdown(highlighted, unsafe_allow_html=True)


    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download detailed table as CSV", data=csv, file_name="detailed_emails.csv", mime="text/csv")
