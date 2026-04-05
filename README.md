# SaaS Revenue Analysis

End-to-end SaaS analysis portfolio project using a synthetic subscription dataset, SQL-based metric modeling, and interactive visual analytics.

## Key Finding 

`partner_referral` has the lowest CAC of any acquisition channel and a blended LTV:CAC ratio that looks healthy. However, when filtered to Enterprise customers only, its monthly churn is 3x higher than every other channel, compressing average customer lifetime from 52 months to 16.5 and destroying 68% of expected LTV. This is hidden in top-level channel reporting because Enterprise is 13% of signups but 48% of MRR.

## What This Demonstrates

- MRR waterfall: classifying monthly movement into New / Expansion / 
  Contraction / Churn / Reactivation using SQL window functions
- Cohort retention: MRR retention matrix built with recursive CTEs, 
  visualized as a heatmap
- Channel segmentation: retention curves split by acquisition channel 
  and plan tier to surface tier-specific anomalies
- LTV/CAC modeling: lifetime value by channel x plan segment with 
  CAC payback analysis

**Tech Stack:** Python, SQLite, Plotly Dash, pandas

## Repository Structure

```text
saas-dashboard/
|- data/
|  |- saas_dataset_generator.py
|  |- saas_data.db
|  |- mrr_waterfall.csv
|  |- cohort_retention.csv
|  |- ltv_cac.csv
|  `- raw/
|- notebooks/
|  `- saas_dashboard.ipynb
|- dashboard/
|  `- saas_dashboard.py
|- docs/
|  `- index.html
|  `- *.html
|- outputs/
|  `- *.png
|- docs/
|- README.md
|- requirements.txt
```

## How To Run 

From `saas-dashboard/`:

```bash
pip install -r requirements.txt
python dashboard/saas_dashboard.py
```

Open in browser:

- http://127.0.0.1:8050/

Run notebook version:

```bash
jupyter notebook notebooks/saas_dashboard.ipynb
```

## Live Site
feliciachen01.github.io/saas-revenue-analysis