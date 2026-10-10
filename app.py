import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="EdTech Sales Ops Intelligence Agent", layout="wide")
st.title("EdTech Sales Operations Intelligence Agent")
st.caption("Sales pipeline analytics · CRO performance · visa risk · commission checks · OpenRouter insights")

REQUIRED = [
    "Date", "Nationality", "Applications", "Payments", "Visa_Submissions",
    "Visa_Approvals", "CRO_ID", "Conversion_Stage", "Commission_Pending"
]
NUMERIC = ["Applications", "Payments", "Visa_Submissions", "Visa_Approvals", "Commission_Pending"]


def load_data(uploaded):
    if uploaded is None:
        default_path = Path(__file__).parent / "sales_data.xlsx"
        if default_path.exists():
            return pd.read_excel(default_path)
        return None
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def prepare_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Nationality", "CRO_ID", "Conversion_Stage"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Only exact duplicate rows are dropped automatically. Missing numeric values are
    # preserved for audit and filled with zero only in a separate analysis copy.
    exact_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    return df, exact_dupes


def data_quality(df):
    q = pd.DataFrame({
        "Metric": [
            "Rows after exact-duplicate removal",
            "Exact duplicate rows removed",
            "Missing values (all cells)",
            "Invalid/missing dates",
            "Negative numeric values",
            "Payments > Applications",
            "Visa submissions > Payments",
            "Visa approvals > Visa submissions",
            "Applications = 0 (conversion denominator risk)",
        ],
        "Count": [
            len(df),
            st.session_state.get("dupes_removed", 0),
            int(df.isna().sum().sum()),
            int(df["Date"].isna().sum()),
            int((df[NUMERIC] < 0).sum().sum()),
            int((df["Payments"] > df["Applications"]).sum()),
            int((df["Visa_Submissions"] > df["Payments"]).sum()),
            int((df["Visa_Approvals"] > df["Visa_Submissions"]).sum()),
            int((df["Applications"] == 0).sum()),
        ]
    })
    return q


def analysis_copy(df):
    out = df.copy()
    # Do not silently overwrite the audit data; missing numeric values are filled only
    # in the calculation copy, and the report explicitly flags them.
    out[NUMERIC] = out[NUMERIC].fillna(0)
    return out


def safe_rate(num, den):
    return num.div(den.where(den.ne(0)))


def build_reports(df):
    d = analysis_copy(df)
    d["Payment_Conversion"] = safe_rate(d["Payments"], d["Applications"])
    d["Visa_Submission_Rate"] = safe_rate(d["Visa_Submissions"], d["Applications"])
    d["Visa_Approval_Rate"] = safe_rate(d["Visa_Approvals"], d["Visa_Submissions"])

    nationality = d.groupby("Nationality", dropna=False).agg(
        Applications=("Applications", "sum"),
        Payments=("Payments", "sum"),
        Visa_Submissions=("Visa_Submissions", "sum"),
        Visa_Approvals=("Visa_Approvals", "sum"),
    ).reset_index()
    nationality["Payment_Conversion"] = safe_rate(nationality["Payments"], nationality["Applications"])
    nationality["Visa_Approval_Rate"] = safe_rate(nationality["Visa_Approvals"], nationality["Visa_Submissions"])
    nationality["Visa_Risk_Flag"] = nationality["Visa_Approval_Rate"].lt(0.60)
    nationality["Approval_Rate_Over_100_Flag"] = nationality["Visa_Approval_Rate"].gt(1)

    cro = d.groupby("CRO_ID", dropna=False).agg(
        Applications=("Applications", "sum"),
        Payments=("Payments", "sum"),
        Visa_Submissions=("Visa_Submissions", "sum"),
        Visa_Approvals=("Visa_Approvals", "sum"),
        Commission_Pending=("Commission_Pending", "sum"),
        Records=("CRO_ID", "size"),
    ).reset_index()
    cro["Payment_Conversion"] = safe_rate(cro["Payments"], cro["Applications"])
    cro["KPI_Status"] = cro["Payment_Conversion"].gt(0.20).map({True: "Above 20%", False: "At/below 20%"})

    daily = d.groupby("Date", dropna=False).agg(
        Applications=("Applications", "sum"),
        Payments=("Payments", "sum"),
        Visa_Submissions=("Visa_Submissions", "sum"),
        Visa_Approvals=("Visa_Approvals", "sum"),
    ).reset_index().sort_values("Date")
    daily["Payment_Conversion"] = safe_rate(daily["Payments"], daily["Applications"])
    daily["Visa_Submission_Rate"] = safe_rate(daily["Visa_Submissions"], daily["Applications"])
    daily["Visa_Approval_Rate"] = safe_rate(daily["Visa_Approvals"], daily["Visa_Submissions"])
    return d, nationality, cro, daily


def forecast_may(daily):
    valid = daily.dropna(subset=["Date"]).sort_values("Date").copy()
    if valid.empty:
        return pd.DataFrame(), {"baseline": None, "last7": None, "prev7": None, "trend": None}
    valid = valid.set_index("Date").asfreq("D")
    valid["Applications"] = valid["Applications"].fillna(0)
    last7 = float(valid["Applications"].tail(7).mean())
    prev7 = float(valid["Applications"].iloc[-14:-7].mean()) if len(valid) >= 14 else None
    trend = (last7 / prev7 - 1) if prev7 not in (None, 0) else None
    # Assignment's requested baseline: latest 7-day average. May has 31 days.
    # Seasonal uplift is not applied unless more historical months are provided.
    baseline_daily = last7
    total = baseline_daily * 31
    summary = {
        "baseline": total,
        "last7": last7,
        "prev7": prev7,
        "trend": trend,
        "seasonality_applied": False
    }
    may_dates = pd.date_range("2026-05-01", "2026-05-31", freq="D")
    forecast = pd.DataFrame({"Date": may_dates, "Forecast_Applications": [baseline_daily] * len(may_dates)})
    forecast["Forecast_Method"] = "Latest 7-day average; flat daily baseline"
    return forecast, summary


def excel_export(df, quality, nationality, cro, daily, forecast, risk, ai_summary):
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        quality.to_excel(writer, index=False, sheet_name="Data_Quality")
        daily.to_excel(writer, index=False, sheet_name="Daily_KPIs")
        nationality.to_excel(writer, index=False, sheet_name="Nationality")
        cro.to_excel(writer, index=False, sheet_name="CRO_Commissions")
        forecast.to_excel(writer, index=False, sheet_name="May_Forecast")
        risk.to_excel(writer, index=False, sheet_name="Visa_Risk")
        pd.DataFrame({"Executive_Summary": [ai_summary]}).to_excel(writer, index=False, sheet_name="Summary")
    return buff.getvalue()


def get_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def get_ai_summary(df, quality, nationality, cro, forecast_summary):
    key = get_api_key()
    if not key:
        return "OpenRouter key not found. Set OPENROUTER_API_KEY in your terminal or configure it in Streamlit Secrets. The deterministic analytics and Excel export still work."
    model = st.session_state.get("model_id", "openai/gpt-4o-mini")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90, max_retries=2)
    data = {
        "rows": int(len(df)),
        "date_min": str(df["Date"].min()),
        "date_max": str(df["Date"].max()),
        "missing_cells": int(df.isna().sum().sum()),
        "data_quality_checks": quality.to_dict(orient="records"),
        "nationality_summary": nationality.round(4).to_dict(orient="records"),
        "cro_summary": cro.round(4).to_dict(orient="records"),
        "forecast": forecast_summary,
    }
    prompt = (
        "You are an EdTech recruitment sales operations analyst. Use only the supplied metrics. "
        "Do not claim causation. Highlight data-quality caveats before making recommendations. "
        "Return concise sections: Executive Summary, Pipeline Gaps, CRO Actions, Nationality/Visa Risk, "
        "3 Actions for the Sales Team, Automation Idea. Treat any approval rate above 100% as a data issue. "
        "The May forecast is based only on a recent 7-day average; do not call it seasonally adjusted.\n\n"
        + str(data)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce evidence-based operational analysis. Never invent metrics."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "The model returned no summary."



def chatbot_tool_data(df, nationality, cro, daily, quality):
    """Expose deterministic, dataset-grounded functions to the LLM agent."""
    d = analysis_copy(df)
    apps = float(d["Applications"].sum())
    pays = float(d["Payments"].sum())
    visa_sub = float(d["Visa_Submissions"].sum())
    visa_approved = float(d["Visa_Approvals"].sum())
    pending = float(d["Commission_Pending"].sum())

    def kpi_summary():
        return {
            "records": int(len(d)),
            "date_range": [str(d["Date"].min()), str(d["Date"].max())],
            "applications": apps,
            "payments": pays,
            "payment_conversion": pays / apps if apps else None,
            "visa_submissions": visa_sub,
            "visa_submission_rate_vs_applications": visa_sub / apps if apps else None,
            "visa_approvals": visa_approved,
            "visa_approval_rate_vs_submissions": visa_approved / visa_sub if visa_sub else None,
            "pending_commissions": pending,
        }

    def nationality_metrics():
        cols = ["Nationality", "Applications", "Payments", "Payment_Conversion", "Visa_Submissions", "Visa_Approvals", "Visa_Approval_Rate", "Visa_Risk_Flag", "Approval_Rate_Over_100_Flag"]
        return nationality[cols].sort_values("Payment_Conversion", ascending=False).replace({float("nan"): None}).to_dict(orient="records")

    def cro_metrics():
        cols = ["CRO_ID", "Applications", "Payments", "Payment_Conversion", "Visa_Submissions", "Visa_Approvals", "Commission_Pending", "KPI_Status", "Records"]
        return cro[cols].sort_values("Payment_Conversion", ascending=False).to_dict(orient="records")

    def daily_metrics(start_date=None, end_date=None):
        x = daily.copy()
        if start_date:
            x = x[x["Date"] >= pd.to_datetime(start_date, errors="coerce")]
        if end_date:
            x = x[x["Date"] <= pd.to_datetime(end_date, errors="coerce")]
        x["Date"] = x["Date"].astype(str)
        return x.to_dict(orient="records")

    def quality_issues():
        violations = d[(d["Payments"] > d["Applications"]) | (d["Visa_Submissions"] > d["Payments"]) | (d["Visa_Approvals"] > d["Visa_Submissions"])]
        cols = ["Date", "Nationality", "Applications", "Payments", "Visa_Submissions", "Visa_Approvals", "CRO_ID"]
        return {
            "quality_checks": quality.to_dict(orient="records"),
            "funnel_violation_rows": violations[cols].assign(Date=violations["Date"].astype(str)).to_dict(orient="records"),
            "note": "Do not silently correct these records. Verify source definitions and CRM records with the business owner."
        }

    return {
        "get_kpi_summary": kpi_summary,
        "get_nationality_metrics": nationality_metrics,
        "get_cro_metrics": cro_metrics,
        "get_daily_metrics": daily_metrics,
        "get_data_quality_issues": quality_issues,
    }


def run_dataset_chat(question, history, df, nationality, cro, daily, quality, model):
    key = get_api_key()
    if not key:
        raise RuntimeError("OpenRouter key is missing. Configure OPENROUTER_API_KEY in Streamlit Secrets for the deployed app.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90, max_retries=2)
    funcs = chatbot_tool_data(df, nationality, cro, daily, quality)
    tool_specs = [
        {"type": "function", "function": {"name": "get_kpi_summary", "description": "Get overall dataset totals, date range, conversion and pending commissions.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "get_nationality_metrics", "description": "Get application/payment conversion and visa approval rates by nationality, including risk flags.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "get_cro_metrics", "description": "Get applications, payments, conversion, pending commission and KPI status by CRO.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "get_daily_metrics", "description": "Get daily KPI records, optionally filtered by inclusive ISO dates YYYY-MM-DD.", "parameters": {"type": "object", "properties": {"start_date": {"type": "string", "description": "Optional inclusive start date YYYY-MM-DD"}, "end_date": {"type": "string", "description": "Optional inclusive end date YYYY-MM-DD"}}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "get_data_quality_issues", "description": "Inspect missing values, duplicates and impossible funnel-order relationships such as payments exceeding applications.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    ]
    system = (
        "You are the EdTech Sales Operations Dataset Chatbot. Answer questions using the uploaded dataset only. "
        "Call the relevant tools to retrieve exact figures before answering. Never invent figures or imply correlation is causation. "
        "Definitions: payment conversion = payments / applications; visa submission rate = visa submissions / applications; "
        "visa approval rate = visa approvals / visa submissions; CRO benchmark is payment conversion >20%; visa-risk threshold is approval rate <60%. "
        "If a visa approval rate exceeds 100%, flag it as a data-quality issue, not exceptional performance. Do not silently correct source records. "
        "The file contains March 2026 observations only, so do not claim a validated seasonal forecast. Clearly distinguish calculated values from recommendations. "
        "Be concise, show formulas where useful, and use tables for comparisons."
    )
    messages = [{"role": "system", "content": system}]
    for item in history[-8:]:
        if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    for _ in range(5):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_specs,
            tool_choice="auto",
            temperature=0.1,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "I couldn't generate a response. Please try asking a more specific question."
        messages.append({"role": "assistant", "content": message.content, "tool_calls": [tc.model_dump(exclude_none=True) for tc in message.tool_calls]})
        for tc in message.tool_calls:
            name = tc.function.name
            if name not in funcs:
                result = {"error": "Unknown tool"}
            else:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = funcs[name](**args)
                except Exception as exc:
                    result = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})
    return "I reached the analysis step limit. Please ask a narrower question."


uploaded = st.file_uploader("Upload sales workbook (.xlsx) or CSV", type=["xlsx", "csv"])
if uploaded is None:
    st.info("Upload your sales file, or place a copy named sales_data.xlsx beside this script to load it by default.")
    st.stop()

try:
    raw = load_data(uploaded)
    df, dupes = prepare_data(raw)
    st.session_state["dupes_removed"] = dupes
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

st.sidebar.header("Agent settings")
st.session_state["model_id"] = st.sidebar.text_input("OpenRouter model ID", "openai/gpt-4o-mini")
st.sidebar.caption("Python computes metrics. OpenRouter drafts the narrative from those computed results.")

quality = data_quality(df)
d, nationality, cro, daily = build_reports(df)
forecast, fsum = forecast_may(daily)
risk = nationality[nationality["Visa_Risk_Flag"] | nationality["Approval_Rate_Over_100_Flag"]].copy()

apps = float(d["Applications"].sum())
payments = float(d["Payments"].sum())
submissions = float(d["Visa_Submissions"].sum())
visa_rate = submissions / apps if apps else 0
commission = float(d["Commission_Pending"].sum())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Applications", f"{apps:,.0f}")
k2.metric("Payments", f"{payments:,.0f}")
k3.metric("Payment conversion", f"{payments/apps:.1%}" if apps else "N/A")
k4.metric("Visa submission / applications", f"{visa_rate:.1%}" if apps else "N/A")
k5.metric("Pending commissions", f"{commission:,.0f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Daily KPIs", "Nationality & Visa Risk", "CRO & Commissions", "Forecast", "Data Quality & AI Summary", "Dataset Chatbot"])

with tab1:
    st.subheader("Daily pipeline")
    st.caption("The uploaded file has one record per date. If the source is meant to contain multiple CRO/nationality rows per day, daily totals may be incomplete.")
    st.dataframe(daily, use_container_width=True)
    fig, ax = plt.subplots()
    ax.plot(daily["Date"], daily["Applications"], marker="o", label="Applications")
    ax.plot(daily["Date"], daily["Payments"], marker="o", label="Payments")
    ax.set_ylabel("Count")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Nationality conversion and visa risk")
    st.dataframe(nationality.sort_values("Payment_Conversion", ascending=False), use_container_width=True)
    st.write("Visa-risk flag: approval rate below 60%. Approval rates above 100% are separately flagged as data-quality issues.")
    st.dataframe(risk, use_container_width=True)
    fig, ax = plt.subplots()
    plotnat = nationality.dropna(subset=["Payment_Conversion"]).sort_values("Payment_Conversion")
    ax.barh(plotnat["Nationality"].astype(str), plotnat["Payment_Conversion"] * 100)
    ax.axvline(20, linestyle="--", label="20% reference")
    ax.set_xlabel("Payments / Applications (%)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("CRO performance and pending commissions")
    st.dataframe(cro.sort_values("Payment_Conversion", ascending=False), use_container_width=True)
    st.metric("Total pending commissions", f"{cro['Commission_Pending'].sum():,.0f}")
    st.caption("Commission totals are summed from the supplied records; validate against the finance ledger and commission rules before payment or contract decisions.")

with tab4:
    st.subheader("May 2026 application forecast")
    if fsum.get("baseline") is not None:
        a, b, c = st.columns(3)
        a.metric("May baseline projection", f"{fsum['baseline']:,.0f}")
        b.metric("Latest 7-day average", f"{fsum['last7']:,.1f} apps/day")
        c.metric("Change vs prior 7 days", f"{fsum['trend']:.1%}" if fsum["trend"] is not None else "N/A")
    st.warning("Only March observations are present. This is a 7-day-average baseline, not a validated seasonal forecast. Add prior months to estimate seasonality robustly.")
    st.dataframe(forecast, use_container_width=True)
    fig, ax = plt.subplots()
    ax.plot(daily["Date"], daily["Applications"], label="Observed")
    ax.plot(forecast["Date"], forecast["Forecast_Applications"], label="May baseline")
    ax.set_ylabel("Applications per day")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with tab5:
    st.subheader("Data-quality audit")
    st.dataframe(quality, use_container_width=True)
    st.write("Rows with funnel-order violations")
    violations = d[
        (d["Payments"] > d["Applications"]) |
        (d["Visa_Submissions"] > d["Payments"]) |
        (d["Visa_Approvals"] > d["Visa_Submissions"])
    ]
    st.dataframe(violations, use_container_width=True)
    if st.button("Generate AI executive summary"):
        with st.spinner("Generating summary with OpenRouter..."):
            try:
                summary = get_ai_summary(d, quality, nationality, cro, fsum)
                st.session_state["ai_summary"] = summary
            except Exception as exc:
                st.error(f"OpenRouter request failed: {exc}")
    ai_summary = st.session_state.get("ai_summary", "Click Generate AI executive summary to draft evidence-based recommendations.")
    st.markdown(ai_summary)

st.divider()
ai_summary = st.session_state.get("ai_summary", "AI summary not generated.")
st.download_button(
    "Download full Excel report",
    data=excel_export(d, quality, nationality, cro, daily, forecast, risk, ai_summary),
    file_name="EdTech_Sales_Operations_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


with tab6:
    st.subheader("Ask questions about your sales dataset")
    st.caption("The chatbot uses Python analytical tools to retrieve figures from this workbook before composing an answer with OpenRouter.")
    st.markdown("Try: **Which CRO has the highest conversion?** · **Which nationalities are below the 60% visa approval threshold?** · **Show the data-quality issues.** · **What are total pending commissions?**")
    if "dataset_chat_history" not in st.session_state:
        st.session_state["dataset_chat_history"] = []
    for chat_item in st.session_state["dataset_chat_history"]:
        with st.chat_message(chat_item["role"]):
            st.markdown(chat_item["content"])
    user_question = st.chat_input("Ask a question about applications, payments, CROs, visas, or commissions…", key="dataset_chat_input")
    if user_question:
        st.session_state["dataset_chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the dataset and preparing an answer..."):
                try:
                    answer = run_dataset_chat(
                        user_question,
                        st.session_state["dataset_chat_history"][:-1],
                        d, nationality, cro, daily, quality,
                        st.session_state.get("model_id", "openai/gpt-4o-mini")
                    )
                    st.markdown(answer)
                    st.session_state["dataset_chat_history"].append({"role": "assistant", "content": answer})
                except Exception as exc:
                    st.error(f"Chatbot request failed: {exc}")
    if st.button("Clear chatbot conversation"):
        st.session_state["dataset_chat_history"] = []
        st.rerun()
