# SaaS Revenue Analysis

End-to-end SaaS analytics portfolio project using a synthetic subscription dataset, SQL-based metric modeling, and interactive visual analytics.

## Key Finding 

`partner_referral` looks normal in aggregate retention, but Enterprise-tier analysis exposes the anomaly: much higher churn than peer channels, driving significantly lower Enterprise LTV.

**Tech Stack:** Python, SQLite, Plotly Dash, pandas

## Project Summary

This project demonstrates how to analyze a SaaS business from raw subscription data through dashboard-ready outputs. It includes:

- synthetic SaaS dataset generation (`data/saas_data.db`)
- monthly MRR movement modeling
- cohort retention analysis
- retention segmentation by acquisition channel
- LTV/CAC analysis with anomaly highlighting
- an interactive Dash dashboard (`dashboard/saas_dashboard.py`)

## What This Demonstrates

- SQL-first analytics workflows for product and revenue metrics
- practical SaaS metric decomposition: `New`, `Expansion`, `Contraction`, `Churn`, `Reactivation`
- cohort MRR retention modeling and trend interpretation
- channel-level segmentation to reveal hidden anomalies
- tradeoff analysis with LTV:CAC and payback period views
- reproducible dashboard artifact generation for portfolio presentation

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
|- outputs/
|  `- *.html chart exports
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

## Data + Outputs

- canonical data inputs live in `data/`
- transient/intermediate exports live in `data/raw/`
- rendered charts are exported to `outputs/`
