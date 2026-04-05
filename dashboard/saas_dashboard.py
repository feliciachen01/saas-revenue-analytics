import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "saas_data.db"

ALL_PLANS = ["Starter", "Growth", "Enterprise"]
ALL_CHANNELS = sorted([
    "content_marketing", "direct", "organic_search",
    "paid_search", "partner_referral", "social_media",
])

CHANNEL_COLORS = {
    "organic_search":    "#2c3e50",
    "paid_search":       "#2980b9",
    "partner_referral":  "#e74c3c",
    "content_marketing": "#27ae60",
    "direct":            "#8e44ad",
    "social_media":      "#f39c12",
}
PLAN_COLORS = {
    "Starter":    "#5b8dd9",
    "Growth":     "#f0a500",
    "Enterprise": "#e05c5c",
}
CAC_BY_CHANNEL = {
    "organic_search": 120, "paid_search": 340, "social_media": 280,
    "content_marketing": 150, "partner_referral": 95, "direct": 60,
}

# 36 months in dataset (2023-01 to 2025-12), used for the date-range slider
ALL_MONTHS = [
    f"{y}-{m:02d}-01"
    for y in [2023, 2024, 2025]
    for m in range(1, 13)
]

MIN_COHORTS_CH = 8  # minimum cohorts needed to plot a channel-retention point
CHANNEL_RETENTION_CACHE = {}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mrr_movements():
    csv_path = DATA_DIR / "mrr_waterfall.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_cohort():
    csv_path = DATA_DIR / "cohort_retention.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.set_index("cohort")
        return df
    return pd.DataFrame()


def load_ltv():
    csv_path = DATA_DIR / "ltv_cac.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _channel_retention_query(plan_filters=None, mode="mrr"):
    """Return {channel: {offset: (avg_pct, n_cohorts)}} from DB."""
    if mode not in {"mrr", "logo"}:
        raise ValueError(f"Unsupported retention mode: {mode}")

    plan_filters = tuple(sorted(plan_filters or ()))
    cache_key = (plan_filters, mode)
    if cache_key in CHANNEL_RETENTION_CACHE:
        return CHANNEL_RETENTION_CACHE[cache_key]

    quoted_plans = ", ".join(f"'{plan}'" for plan in plan_filters)
    plan_clause = f"AND s.plan IN ({quoted_plans})" if quoted_plans else ""
    sql = f"""
    WITH RECURSIVE months AS (
        SELECT '2023-01-01' AS month_start
        UNION ALL
        SELECT date(month_start, '+1 month') FROM months
        WHERE month_start < '2025-12-01'
    ),
    user_cohort AS (
        SELECT s.user_id, u.acquisition_channel AS channel,
               strftime('%Y-%m-01', MIN(s.started_at)) AS cohort_date
        FROM fact_subscriptions s
        JOIN dim_users u ON s.user_id = u.user_id
        WHERE 1=1 {plan_clause}
        GROUP BY s.user_id
    ),
    sub_months AS (
        SELECT s.user_id, m.month_start, s.mrr
        FROM fact_subscriptions s
        JOIN months m
          ON m.month_start >= strftime('%Y-%m-01', s.started_at)
         AND (s.ended_at IS NULL
              OR m.month_start < strftime('%Y-%m-01', s.ended_at, '+1 month'))
        WHERE 1=1 {plan_clause}
    ),
    customer_month AS (
        SELECT user_id, month_start, MAX(mrr) AS mrr
        FROM sub_months GROUP BY user_id, month_start
    )
    SELECT uc.channel,
           (CAST(strftime('%Y', cm.month_start) AS INTEGER)
            - CAST(strftime('%Y', uc.cohort_date) AS INTEGER)) * 12
           + (CAST(strftime('%m', cm.month_start) AS INTEGER)
             - CAST(strftime('%m', uc.cohort_date) AS INTEGER)) AS month_offset,
           SUM(cm.mrr) AS active_mrr,
           COUNT(DISTINCT cm.user_id) AS active_users,
           strftime('%Y-%m', uc.cohort_date) AS cohort
    FROM customer_month cm
    JOIN user_cohort uc ON cm.user_id = uc.user_id
    GROUP BY uc.channel, cohort, month_offset
    ORDER BY uc.channel, cohort, month_offset;
    """
    try:
        conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(sql).fetchall()
    conn.close()

    raw = defaultdict(lambda: defaultdict(dict))
    for channel, offset, active_mrr, active_users, cohort in rows:
        metric_value = active_mrr if mode == "mrr" else active_users
        raw[channel][cohort][offset] = metric_value

    result = {}
    for channel, cohort_data in raw.items():
        ret_by_cohort = {}
        for cohort, offsets in cohort_data.items():
            base = offsets.get(0, 0)
            if base == 0:
                continue
            ret_by_cohort[cohort] = {o: 100.0 * v / base for o, v in offsets.items()}

        if not ret_by_cohort:
            continue

        max_offset = max(o for r in ret_by_cohort.values() for o in r)
        curve = {}
        for o in range(max_offset + 1):
            vals = [r[o] for r in ret_by_cohort.values() if o in r]
            if len(vals) >= MIN_COHORTS_CH:
                curve[o] = (round(np.mean(vals), 1), len(vals))

        result[channel] = curve

    CHANNEL_RETENTION_CACHE[cache_key] = result
    return result


print("Loading CSV data...")
mrr_movement_df = load_mrr_movements()
cohort_df = load_cohort()
ltv_df = load_ltv()
print("Data ready.\n")


# ── Helper: KPI card ──────────────────────────────────────────────────────────

def kpi_card(label, value, highlight=False, tooltip=None):
    return html.Div([
        html.P(label, title=tooltip, style={
            "margin": "0 0 4px", "fontSize": "10px", "color": "#7f8c8d",
            "textTransform": "uppercase", "fontWeight": "700", "letterSpacing": "0.5px",
            "textDecoration": "underline dotted" if tooltip else "none",
            "cursor": "help" if tooltip else "default",
        }),
        html.P(value, style={
            "margin": 0, "fontSize": "20px", "fontWeight": "700",
            "color": "#c0392b" if highlight else "#2c3e50",
        }),
    ], style={
        "background": "#fff8f8" if highlight else "#f8f9fa",
        "border": "1px solid #f5c6cb" if highlight else "1px solid #ecf0f1",
        "borderRadius": "8px", "padding": "12px 16px",
        "flex": "1", "minWidth": "120px",
    })


# ── Chart builders ────────────────────────────────────────────────────────────

def add_horizontal_line(fig, y, color, dash, width=1.5, annotation_text=None,
                        annotation_position="top right", annotation_font=None):
    """Plotly-version-safe horizontal reference line."""
    if hasattr(fig, "add_hline"):
        fig.add_hline(
            y=y,
            line_dash=dash,
            line_color=color,
            line_width=width,
            annotation_text=annotation_text,
            annotation_position=annotation_position,
            annotation_font=annotation_font,
        )
        return

    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=y,
        y1=y,
        line=dict(color=color, dash=dash, width=width),
    )
    if annotation_text:
        x = 0.98 if annotation_position == "top right" else 0.02
        xanchor = "right" if annotation_position == "top right" else "left"
        fig.add_annotation(
            xref="paper",
            x=x,
            y=y,
            text=annotation_text,
            showarrow=False,
            xanchor=xanchor,
            yanchor="bottom",
            font=annotation_font or dict(size=10, color="#666"),
            bgcolor="rgba(255,255,255,0.85)",
        )


def build_mrr_movement_fig(start_month, end_month):
    if mrr_movement_df.empty:
        return None, html.P(
            "mrr_waterfall.csv not found — run mrr_waterfall.py first.",
            style={"color": "#e74c3c"},
        )

    df = mrr_movement_df[
        (mrr_movement_df["month"] >= start_month)
        & (mrr_movement_df["month"] <= end_month)
    ].copy()
    if df.empty:
        return None, html.P("No data for selected date range.")

    labels = [m[:7] for m in df["month"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="New",         x=labels, y=df["new"],         marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Expansion",   x=labels, y=df["expansion"],   marker_color="#3498db"))
    fig.add_trace(go.Bar(name="Reactivation",x=labels, y=df["reactivation"],marker_color="#9b59b6"))
    fig.add_trace(go.Bar(name="Contraction", x=labels, y=df["contraction"], marker_color="#e67e22"))
    fig.add_trace(go.Bar(name="Churn",       x=labels, y=df["churn"],       marker_color="#e74c3c"))
    fig.add_trace(go.Scatter(
        name="Ending MRR", x=labels, y=df["ending_mrr"],
        mode="lines+markers", yaxis="y2",
        line=dict(color="#2c3e50", width=2.5), marker=dict(size=5),
    ))
    fig.update_layout(
        title=f"MRR Movement Trend: {labels[0]} - {labels[-1]}",
        barmode="relative",
        yaxis=dict(title="MRR Movement ($)", tickformat="$,.0f"),
        yaxis2=dict(title="Ending MRR ($)", overlaying="y", side="right",
                    tickformat="$,.0f", showgrid=False),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", yanchor="bottom"),
        template="plotly_white",
        height=500,
        margin=dict(l=80, r=90, t=80, b=50),
    )
    return fig, None


def build_cohort_figs(start_month, end_month):
    if cohort_df.empty:
        return None, None, html.P(
            "cohort_retention.csv not found — run cohort_retention.py first.",
            style={"color": "#e74c3c"},
        )

    start_ym = start_month[:7]
    end_ym = end_month[:7]
    df = cohort_df[(cohort_df.index >= start_ym) & (cohort_df.index <= end_ym)].copy()
    if df.empty:
        return None, None, html.P("No cohorts in selected date range.")

    cohorts = list(df.index)
    month_cols = [c for c in df.columns if c.startswith("M")]

    z_arr = df[month_cols].values.astype(float)
    z_arr[z_arr == 0] = np.nan
    text_vals = [
        [f"{v:.0f}%" if not pd.isna(v) else "" for v in row]
        for row in z_arr
    ]

    try:
        heatmap_trace = go.Heatmap(
            z=z_arr, x=month_cols, y=cohorts,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 8},
            colorscale=[
                [0.0,  "#d73027"], [0.3, "#fc8d59"], [0.5, "#fee08b"],
                [0.7,  "#d9ef8b"], [0.85, "#91cf60"], [1.0, "#1a9850"],
            ],
            zmin=0, zmax=120,
            colorbar=dict(title="MRR<br>Retention %", ticksuffix="%"),
            hoverongaps=False,
            hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
        )
        heatmap = go.Figure(data=heatmap_trace)
    except Exception:
        heatmap = go.Figure(data=go.Heatmap(
            z=z_arr, x=month_cols, y=cohorts,
            colorscale=[
                [0.0,  "#d73027"], [0.3, "#fc8d59"], [0.5, "#fee08b"],
                [0.7,  "#d9ef8b"], [0.85, "#91cf60"], [1.0, "#1a9850"],
            ],
            zmin=0, zmax=120,
            colorbar=dict(title="MRR<br>Retention %", ticksuffix="%"),
            hoverongaps=False,
            hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
        ))
        for y_i, cohort in enumerate(cohorts):
            for x_i, month in enumerate(month_cols):
                label = text_vals[y_i][x_i]
                if label:
                    heatmap.add_annotation(
                        x=month,
                        y=cohort,
                        text=label,
                        showarrow=False,
                        font=dict(size=8, color="#2c3e50"),
                    )
    heatmap.update_layout(
        title=f"MRR Cohort Retention Heatmap ({start_ym} – {end_ym})",
        xaxis=dict(
            title=dict(text="Months Since Signup", standoff=2),
            dtick=3,
            side="top",
        ),
        yaxis=dict(title="Signup Cohort", autorange="reversed"),
        template="plotly_white",
        height=max(380, len(cohorts) * 22 + 120),
        margin=dict(l=80, r=100, t=80, b=40),
    )

    # Average retention curve
    MIN_C = 10
    valid_x, avg_y, p25_y, p75_y = [], [], [], []
    for col in month_cols:
        vals = df[col].replace(0, np.nan).dropna()
        if len(vals) >= MIN_C:
            valid_x.append(col)
            avg_y.append(float(vals.mean()))
            p25_y.append(float(np.percentile(vals, 25)))
            p75_y.append(float(np.percentile(vals, 75)))

    curve = go.Figure()
    if valid_x:
        curve.add_trace(go.Scatter(x=valid_x, y=p75_y, mode="lines",
                                   line=dict(width=0), showlegend=False))
        curve.add_trace(go.Scatter(x=valid_x, y=p25_y, mode="lines",
                                   line=dict(width=0), fill="tonexty",
                                   fillcolor="rgba(52,152,219,0.2)", name="P25–P75"))
        curve.add_trace(go.Scatter(
            x=valid_x, y=avg_y, mode="lines+markers", name="Average",
            line=dict(color="#2c3e50", width=2.5), marker=dict(size=5),
        ))
    curve.update_layout(
        title="Average MRR Retention Curve",
        xaxis=dict(title="Months Since Signup"),
        yaxis=dict(title="MRR Retention %", ticksuffix="%"),
        template="plotly_white",
        height=360,
        margin=dict(l=80, r=40, t=60, b=50),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    return heatmap, curve, None


def build_channel_retention_fig(plans, channels, mode, title, subtitle):
    if not plans or not channels:
        return None, html.P("Select at least one channel.")

    ret_data = _channel_retention_query(plan_filters=plans, mode=mode)
    filtered = {ch: curve for ch, curve in ret_data.items() if ch in set(channels)}
    if not filtered:
        return None, html.P("No data for selected filters.")

    fig = go.Figure()
    for channel in sorted(filtered):
        curve = filtered[channel]
        if not curve:
            continue
        offsets = sorted(curve)
        vals = [curve[o][0] for o in offsets]
        ns = [curve[o][1] for o in offsets]
        xl = [f"M{o}" for o in offsets]
        highlight = channel == "partner_referral"

        fig.add_trace(go.Scatter(
            x=xl, y=vals, mode="lines+markers", name=channel,
            line=dict(color=CHANNEL_COLORS.get(channel, "#999"), width=3.5 if highlight else 2),
            marker=dict(size=6 if highlight else 4),
            opacity=1.0 if highlight else 0.75,
            customdata=list(zip([channel] * len(offsets), ns)),
            hovertemplate="%{customdata[0]}<br>Month %{x}<br>"
                          "Retention: %{y:.1f}%<br>n=%{customdata[1]}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
        xaxis=dict(title="Months Since Signup", dtick=3),
        yaxis=dict(title="Retention %", ticksuffix="%", range=[0, 115]),
        template="plotly_white",
        height=460,
        margin=dict(l=80, r=40, t=90, b=70),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig, None


def build_ltvcac_figs(plans, channels):
    if ltv_df.empty:
        return None, None, html.P(
            "ltv_cac.csv not found — run ltv_cac.py first.",
            style={"color": "#e74c3c"},
        )
    if not plans or not channels:
        return None, None, html.P("Select at least one plan and channel.")

    df = ltv_df[ltv_df["plan"].isin(plans) & ltv_df["channel"].isin(channels)].copy()
    if df.empty:
        return None, None, html.P("No data for selected filters.")

    # Bubble chart
    bubble = go.Figure()
    sym = {"Starter": "circle", "Growth": "diamond", "Enterprise": "square"}
    for plan in ["Starter", "Growth", "Enterprise"]:
        segs = df[df["plan"] == plan].to_dict("records")
        if not segs:
            continue
        bubble.add_trace(go.Scatter(
            x=[s["payback_months"] for s in segs],
            y=[s["ltv_cac_ratio"]  for s in segs],
            mode="markers", name=plan,
            marker=dict(
                size=[max(8, s["total_initial_mrr"] ** 0.5 / 2) for s in segs],
                color=[CHANNEL_COLORS.get(s["channel"], "#999") for s in segs],
                opacity=0.75, line=dict(width=1.5, color="white"),
                symbol=sym[plan],
            ),
            text=[
                f"{s['channel']}<br>{s['plan']}<br>"
                f"LTV: ${s['ltv']:,.0f}<br>CAC: ${s['cac']}<br>"
                f"LTV:CAC: {s['ltv_cac_ratio']}x<br>"
                f"Payback: {s['payback_months']}mo<br>Churn: {s['monthly_churn_pct']}%/mo"
                for s in segs
            ],
            hoverinfo="text",
        ))

    pr_ent = df[(df["channel"] == "partner_referral") & (df["plan"] == "Enterprise")]
    if not pr_ent.empty:
        r = pr_ent.iloc[0]
        bubble.add_annotation(
            x=r["payback_months"], y=r["ltv_cac_ratio"],
            text=(f"partner_referral Enterprise<br>"
                  f"LTV:CAC {r['ltv_cac_ratio']}x | Churn {r['monthly_churn_pct']}%/mo<br>"
                  f"<i>3× normal Enterprise churn</i>"),
            showarrow=True, arrowhead=2, ax=80, ay=-60,
            bordercolor="#e74c3c", borderwidth=1.5,
            bgcolor="rgba(231,76,60,0.08)", font=dict(size=10),
        )

    add_horizontal_line(
        bubble,
        y=3.0,
        dash="dot",
        color="#bdc3c7",
        annotation_text="LTV:CAC = 3x (healthy threshold)",
        annotation_font=dict(size=9, color="#95a5a6"),
    )
    bubble.update_layout(
        title=("LTV:CAC Analysis by Channel and Plan Tier<br>"
               "<sup>Bubble size = initial MRR volume · "
               "Shape: circle=Starter, diamond=Growth, square=Enterprise</sup>"),
        xaxis=dict(title="CAC Payback Period (months)"),
        yaxis=dict(title="LTV:CAC Ratio", ticksuffix="x"),
        template="plotly_white",
        height=460,
        margin=dict(l=80, r=80, t=90, b=60),
        legend=dict(title="Plan", orientation="h", y=1.02, x=0.5, xanchor="center"),
    )

    # Grouped bar chart
    ch_order = sorted(df["channel"].unique())
    bar = go.Figure()
    for plan in ["Starter", "Growth", "Enterprise"]:
        plan_map = {s["channel"]: s for s in df[df["plan"] == plan].to_dict("records")}
        bar.add_trace(go.Bar(
            name=plan, x=ch_order,
            y=[plan_map[ch]["ltv"] if ch in plan_map else 0 for ch in ch_order],
            marker_color=PLAN_COLORS[plan],
            hovertemplate=f"<b>%{{x}}</b><br>{plan}<br>LTV: $%{{y:,.0f}}<extra></extra>",
        ))

    ent_no_pr = df[(df["plan"] == "Enterprise") & (df["channel"] != "partner_referral")]
    if not ent_no_pr.empty:
        avg_ent = ent_no_pr["ltv"].mean()
        add_horizontal_line(
            bar,
            y=avg_ent,
            dash="dash",
            color="#888",
            width=1.5,
            annotation_text=f"Avg Enterprise LTV (excl. partner_referral): ${avg_ent:,.0f}",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#555"),
        )
        if not pr_ent.empty and "partner_referral" in ch_order:
            r = pr_ent.iloc[0]
            bar.add_annotation(
                x=list(ch_order).index("partner_referral"), y=r["ltv"],
                text=f"${r['ltv']:,.0f}<br><i>vs ${avg_ent:,.0f} avg</i>",
                showarrow=True, arrowhead=2, ax=0, ay=-50,
                font=dict(size=10, color="#c0392b"),
                bordercolor="#e74c3c", borderwidth=1.5,
                bgcolor="rgba(231,76,60,0.08)",
            )

    bar.update_layout(
        barmode="group",
        title="LTV by Acquisition Channel and Plan Tier",
        xaxis=dict(title="Acquisition Channel", tickangle=-20),
        yaxis=dict(title="LTV ($)", tickprefix="$", separatethousands=True),
        template="plotly_white",
        height=420,
        margin=dict(l=80, r=60, t=80, b=100),
        legend=dict(title="Plan", orientation="h", y=1.02, x=0.5, xanchor="center"),
    )

    return bubble, bar, None


# ── App layout ────────────────────────────────────────────────────────────────

app = Dash(__name__, title="SaaS Analytics Dashboard")

LABEL_STYLE = {
    "fontSize": "11px", "fontWeight": "700", "color": "#34495e",
    "textTransform": "uppercase", "letterSpacing": "0.4px",
    "display": "block", "marginBottom": "8px",
}

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.H1("SaaS Analytics Dashboard", style={
            "margin": 0, "fontSize": "20px", "fontWeight": "700", "color": "#2c3e50",
        }),
        html.Span(
            "Synthetic SaaS · MRR Movements · Cohort Retention · LTV/CAC · Anomaly Detection",
            style={"fontSize": "12px", "color": "#7f8c8d", "marginLeft": "16px"},
        ),
    ], style={
        "padding": "12px 24px", "borderBottom": "1px solid #ecf0f1",
        "background": "#fff", "display": "flex", "alignItems": "center",
    }),

    html.Div([

        # ── Sidebar ───────────────────────────────────────────────────────────
        html.Div([
            html.Div("Filters", style={
                "fontSize": "10px", "fontWeight": "700", "color": "#95a5a6",
                "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "18px",
            }),

            html.Div([
                html.Label("Date Range", style=LABEL_STYLE),
                dcc.RangeSlider(
                    id="date-range",
                    min=0, max=35, step=1,
                    value=[0, 35],
                    marks={i: {"label": ALL_MONTHS[i][:7], "style": {"fontSize": "9px"}}
                           for i in [0, 6, 12, 18, 24, 30, 35]},
                    allowCross=False,
                    tooltip={"always_visible": False},
                ),
                html.Div(id="date-display", style={
                    "fontSize": "11px", "color": "#7f8c8d",
                    "textAlign": "center", "marginTop": "4px", "marginBottom": "4px",
                }),
            ], id="date-filter-block"),

            html.Hr(id="hr-after-date", style={
                "border": "none", "borderTop": "1px solid #ecf0f1", "margin": "16px 0"
            }),

            html.Div([
                html.Label("Plan", style=LABEL_STYLE),
                dcc.Checklist(
                    id="plan-filter",
                    options=[{"label": f"  {p}", "value": p} for p in ALL_PLANS],
                    value=ALL_PLANS,
                    labelStyle={
                        "display": "block", "fontSize": "13px",
                        "marginBottom": "5px", "cursor": "pointer",
                    },
                ),
            ], id="plan-filter-block"),

            html.Hr(id="hr-after-plan", style={
                "border": "none", "borderTop": "1px solid #ecf0f1", "margin": "16px 0"
            }),

            html.Div([
                html.Label("Channel", style=LABEL_STYLE),
                dcc.Checklist(
                    id="channel-filter",
                    options=[{"label": f"  {c}", "value": c} for c in ALL_CHANNELS],
                    value=ALL_CHANNELS,
                    labelStyle={
                        "display": "block", "fontSize": "12px",
                        "marginBottom": "4px", "cursor": "pointer",
                    },
                ),
            ], id="channel-filter-block"),

            html.Hr(id="hr-after-channel", style={
                "border": "none", "borderTop": "1px solid #ecf0f1", "margin": "16px 0"
            }),

            html.P(id="filter-help", children=[
                html.Strong("Date range"), " → MRR Movements, Cohort Retention", html.Br(),
                html.Strong("Plan"), " → LTV/CAC", html.Br(),
                html.Strong("Channel"), " → Retention by Channel, LTV/CAC",
            ], style={"fontSize": "11px", "color": "#95a5a6", "lineHeight": "1.9", "margin": 0}),

        ], style={
            "width": "210px", "flexShrink": "0",
            "padding": "20px 14px",
            "background": "#f8f9fa",
            "borderRight": "1px solid #ecf0f1",
            "overflowY": "auto",
        }),

        # ── Tab area ──────────────────────────────────────────────────────────
        html.Div([
            dcc.Tabs(id="tabs", value="mrr", children=[
                dcc.Tab(label="MRR Movements",        value="mrr"),
                dcc.Tab(label="Cohort Retention",     value="cohort"),
                dcc.Tab(label="Retention by Channel", value="channel"),
                dcc.Tab(label="LTV / CAC",            value="ltvcac"),
            ]),
            html.Div(id="tab-content", style={
                "padding": "16px 20px", "overflowY": "auto", "flex": "1",
            }),
        ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),

], style={
    "fontFamily": "'Segoe UI', system-ui, -apple-system, sans-serif",
    "background": "#fff",
    "display": "flex", "flexDirection": "column", "height": "100vh",
})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("date-display", "children"),
    Input("date-range", "value"),
)
def update_date_display(date_range):
    return f"{ALL_MONTHS[date_range[0]][:7]} – {ALL_MONTHS[date_range[1]][:7]}"


@app.callback(
    Output("date-filter-block", "style"),
    Output("plan-filter-block", "style"),
    Output("channel-filter-block", "style"),
    Output("hr-after-date", "style"),
    Output("hr-after-plan", "style"),
    Output("hr-after-channel", "style"),
    Output("filter-help", "style"),
    Input("tabs", "value"),
)
def toggle_filter_visibility(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    hr = {"border": "none", "borderTop": "1px solid #ecf0f1", "margin": "16px 0"}
    hr_hide = {"display": "none"}
    help_hide = {"display": "none"}

    if tab in {"mrr", "cohort"}:
        return show, hide, hide, hr_hide, hr_hide, hr_hide, help_hide

    if tab == "channel":
        return hide, hide, show, hr_hide, hr_hide, hr_hide, help_hide

    if tab == "ltvcac":
        return hide, show, show, hr_hide, hr, hr_hide, help_hide

    return show, show, show, hr, hr, hr, help_hide


@app.callback(
    Output("tab-content", "children"),
    Input("tabs",          "value"),
    Input("date-range",    "value"),
    Input("plan-filter",   "value"),
    Input("channel-filter","value"),
)
def render_tab(tab, date_range, plans, channels):
    start = ALL_MONTHS[date_range[0]]
    end   = ALL_MONTHS[date_range[1]]
    plans    = plans    or []
    channels = channels or []

    # ── MRR Movements ────────────────────────────────────────────────────────
    if tab == "mrr":
        try:
            fig, err = build_mrr_movement_fig(start, end)
            if err is not None:
                return err
            df = mrr_movement_df[
                (mrr_movement_df["month"] >= start) & (mrr_movement_df["month"] <= end)
            ]
            last = df.iloc[-1]
            ending_mrr = float(last["ending_mrr"])
            total_net_new = float(df["net_new_mrr"].sum())
            total_new = float(df["new"].sum())
            total_churn = float(df["churn"].sum())
            kpis = html.Div([
                kpi_card(
                    "Ending MRR",
                    f"${ending_mrr:,.0f}",
                    tooltip=(
                        "Ending MRR for the last selected month. "
                        "Formula per month: beginning_mrr + new + expansion + reactivation + "
                        "contraction + churn."
                    ),
                ),
                kpi_card(
                    "Net New MRR",
                    f"${total_net_new:,.0f}",
                    highlight=total_net_new < 0,
                    tooltip=(
                        "Sum of monthly net new MRR over the selected range. "
                        "Monthly net new = new + expansion + reactivation + contraction + churn."
                    ),
                ),
                kpi_card(
                    "Total New MRR",
                    f"${total_new:,.0f}",
                    tooltip="Sum of new-business MRR from new subscriptions in the selected range.",
                ),
                kpi_card(
                    "Total Churn MRR",
                    f"${total_churn:,.0f}",
                    highlight=True,
                    tooltip=(
                        "Sum of churned MRR in the selected range (stored as negative values)."
                    ),
                ),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "14px", "flexWrap": "wrap"})
            return html.Div([kpis, dcc.Graph(figure=fig, config={"displayModeBar": False})])
        except Exception as ex:
            return html.P(
                f"Unable to render MRR Movements tab: {ex}",
                style={"color": "#e74c3c"},
            )

    # ── Cohort Retention ──────────────────────────────────────────────────────
    elif tab == "cohort":
        try:
            heatmap, curve, err = build_cohort_figs(start, end)
            if err is not None:
                return err
            return html.Div([
                dcc.Graph(figure=heatmap, config={"displayModeBar": False}),
                dcc.Graph(figure=curve,   config={"displayModeBar": False}),
            ])
        except Exception as ex:
            return html.P(
                f"Unable to render Cohort Retention tab: {ex}",
                style={"color": "#e74c3c"},
            )

    # ── Retention by Channel ──────────────────────────────────────────────────
    elif tab == "channel":
        try:
            if not channels:
                return html.P("Select at least one channel.")

            fig_all, err_all = build_channel_retention_fig(
                plans=ALL_PLANS,
                channels=channels,
                mode="logo",
                title="Retention by Acquisition Channel (All Plans - Logo Retention)",
                subtitle="partner_referral blends in at aggregate level",
            )
            if err_all is not None:
                return err_all

            fig_ent, err_ent = build_channel_retention_fig(
                plans=["Enterprise"],
                channels=channels,
                mode="mrr",
                title="Retention by Acquisition Channel (Enterprise - MRR Retention)",
                subtitle="partner_referral anomaly is visible in Enterprise",
            )
            if err_ent is not None:
                return err_ent

            insight = html.Div(
                "Seeded anomaly: partner_referral Enterprise churn is 3x normal "
                "(5.4%/mo vs 1.8%/mo). The aggregate all-plan view can hide this.",
                style={
                    "background": "rgba(231,76,60,0.07)",
                    "border": "1px solid #e74c3c",
                    "borderRadius": "6px", "padding": "10px 14px",
                    "fontSize": "13px", "color": "#c0392b", "marginBottom": "12px",
                },
            )

            return html.Div([
                insight,
                dcc.Graph(figure=fig_all, config={"displayModeBar": False}),
                dcc.Graph(figure=fig_ent, config={"displayModeBar": False}),
            ])
        except Exception as ex:
            return html.P(
                f"Unable to render Retention by Channel tab: {ex}",
                style={"color": "#e74c3c"},
            )

    # ── LTV / CAC ─────────────────────────────────────────────────────────────
    elif tab == "ltvcac":
        try:
            bubble, bar, err = build_ltvcac_figs(plans, channels)
            if err is not None:
                return err
            df = ltv_df[ltv_df["plan"].isin(plans) & ltv_df["channel"].isin(channels)]
            pr_ent = df[(df["channel"] == "partner_referral") & (df["plan"] == "Enterprise")]
            kpis = html.Div([
                kpi_card("Segments shown", str(len(df))),
                kpi_card("Avg LTV:CAC", f"{df['ltv_cac_ratio'].mean():.1f}x"),
                kpi_card("Avg Payback", f"{df['payback_months'].mean():.1f} mo"),
                *([] if pr_ent.empty else [
                    kpi_card("partner_referral Ent LTV",
                             f"${pr_ent.iloc[0]['ltv']:,.0f}", highlight=True),
                ]),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "14px", "flexWrap": "wrap"})
            return html.Div([
                kpis,
                dcc.Graph(figure=bubble, config={"displayModeBar": False}),
                dcc.Graph(figure=bar,    config={"displayModeBar": False}),
            ])
        except Exception as ex:
            return html.P(
                f"Unable to render LTV/CAC tab: {ex}",
                style={"color": "#e74c3c"},
            )

    return html.P("Unknown tab.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
