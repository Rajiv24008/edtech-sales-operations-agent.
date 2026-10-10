# EdTech Sales Operations Intelligence Agent

A Streamlit dashboard for sales-pipeline analysis, CRO performance, pending commissions, visa-risk flags, a transparent May application baseline forecast, data-quality checks, and optional OpenRouter-generated executive recommendations.

## 1. Install

Open Anaconda Prompt and change directory to this project folder:

```bat
cd "C:\path\to\edtech_sales_ops_agent"
pip install -r requirements.txt
```

## 2. Set your OpenRouter key

Set the key in the same Anaconda Prompt window. Replace the placeholder locally; do not paste the key into chat or source code.

```bat
set "OPENROUTER_API_KEY=YOUR_NEW_OPENROUTER_KEY"
```

The dashboard's deterministic analysis and Excel export work without the key. The AI executive summary requires it.

## 3. Launch

```bat
streamlit run app.py
```

Upload `Gedu_Manager Sales Analyst_Assignment data.xlsx` in the app.

## Metrics and assumptions

- Payment conversion = Payments / Applications.
- Visa submission rate = Visa_Submissions / Applications.
- Visa approval rate = Visa_Approvals / Visa_Submissions.
- Visa risk flag = approval rate below 60%; rates above 100% are also flagged as data-quality errors.
- CRO KPI benchmark = payment conversion above 20%.
- Pending commissions are summed from the provided records; reconcile with finance before paying.
- May forecast = latest 7-day average daily applications × 31. This is a baseline, not a seasonal forecast. A single month is not enough to establish robust annual seasonality.
- Exact duplicate rows are removed. Missing values are shown in the audit; numeric blanks are filled with zero only in the analysis copy and are not silently overwritten in the cleaned-data export.
- The source currently contains one row per date. Confirm whether the intended grain is daily total or one row per CRO/nationality per day before interpreting daily totals.
