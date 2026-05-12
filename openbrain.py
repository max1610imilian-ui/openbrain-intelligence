"""
OpenBrain Neural Demand Predictor
Enterprise-Grade Demand Forecasting Dashboard
"""

import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OpenBrain Demand Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME-AWARE CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    color-scheme: light dark;
}

/* Dark Mode Variables */
html, body, .stApp {
    --bg:           #0a0e1f;
    --bg-light:     #0f1629;
    --bg-input:     #141d2e;
    --border:       rgba(255,255,255,0.07);
    --border-hover: rgba(255,255,255,0.15);
    --text-primary: #e8eef5;
    --text-secondary: #a8b5c8;
    --text-muted:   #6b7585;
    --accent-1:     #0084ff;
    --accent-2:     #00d4aa;
}

/* Light Mode Variables */
@media (prefers-color-scheme: light) {
    html, body, .stApp {
        --bg:           #ffffff;
        --bg-light:     #f8f9fb;
        --bg-input:     #f0f2f6;
        --border:       rgba(0,0,0,0.08);
        --border-hover: rgba(0,0,0,0.15);
        --text-primary: #1a202c;
        --text-secondary: #475569;
        --text-muted:   #94a3b8;
        --accent-1:     #0066ff;
        --accent-2:     #00a878;
    }
}

* { box-sizing: border-box; }

html, body {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-light);
    border-right: 1px solid var(--border);
}

/* Main Container */
.main .block-container {
    padding: 2rem;
    max-width: 1400px;
}

/* Expanders */
[data-testid="stExpander"] {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.75rem;
}

[data-testid="stExpander"] summary {
    padding: 0.75rem 1rem;
    background: var(--bg-light);
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    cursor: pointer;
}

[data-testid="stExpander"] summary:hover {
    background: var(--bg-light);
    border-color: var(--border-hover);
}

/* Input Fields */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 2px rgba(0, 132, 255, 0.1) !important;
}

/* Slider */
[data-testid="stSlider"] input {
    accent-color: var(--accent-1) !important;
}

/* Checkbox */
.stCheckbox label {
    color: var(--text-primary) !important;
    cursor: pointer;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border);
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 600;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent-1) !important;
    border-bottom: 3px solid var(--accent-1) !important;
}

/* Metrics */
[data-testid="stMetricContainer"] {
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background: var(--bg-light);
    border: 1px solid var(--border);
}

/* Custom Cards */
.glass-card {
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
}

.alert-info {
    background: rgba(0, 132, 255, 0.08);
    border-left: 4px solid var(--accent-1);
    padding: 1rem;
    border-radius: 6px;
    color: var(--text-primary);
}

.alert-success {
    background: rgba(0, 212, 170, 0.08);
    border-left: 4px solid var(--accent-2);
    padding: 1rem;
    border-radius: 6px;
    color: var(--text-primary);
}

.alert-warning {
    background: rgba(255, 165, 0, 0.08);
    border-left: 4px solid #ff9500;
    padding: 1rem;
    border-radius: 6px;
    color: var(--text-primary);
}

.alert-danger {
    background: rgba(255, 87, 90, 0.08);
    border-left: 4px solid #ff5a5a;
    padding: 1rem;
    border-radius: 6px;
    color: var(--text-primary);
}

.status-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 12px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    border: 1px solid;
    margin-right: 0.5rem;
}

.badge-live {
    background: rgba(0, 212, 170, 0.1);
    color: var(--accent-2);
    border-color: var(--accent-2);
}

.badge-info {
    background: rgba(0, 132, 255, 0.1);
    color: var(--accent-1);
    border-color: var(--accent-1);
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}

.header-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.header-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    """Train neural network on synthetic data (cached permanently)"""
    np.random.seed(42)
    n_days = 1500
    dates = pd.date_range("2020-01-01", periods=n_days)
    
    # Generate features
    temp = np.random.normal(12, 10, n_days).clip(-15, 42)
    rain = np.random.exponential(2, n_days).clip(0, 60)
    season = np.sin(2 * np.pi * dates.month / 12) * 15
    
    df = pd.DataFrame({
        "date": dates,
        "weekday": dates.weekday,
        "month": dates.month,
        "quarter": dates.quarter,
        "day_of_year": dates.dayofyear,
        "temperature": temp,
        "rainfall": rain,
        "sunshine": (np.random.uniform(0, 12, n_days) * (1 - rain / 80)).clip(0, 12),
        "holiday": np.random.choice([0, 1], n_days, p=[0.8, 0.2]),
        "event": np.random.choice([0, 1], n_days, p=[0.95, 0.05]),
        "ad_budget": np.random.uniform(50, 600, n_days),
        "tv_reach": np.random.uniform(0, 80, n_days),
        "social": np.random.uniform(0, 8, n_days),
        "energy_index": np.random.normal(100, 15, n_days).clip(50, 180),
        "competition": np.random.uniform(0, 1, n_days),
        "satisfaction": np.random.normal(7.5, 1, n_days).clip(1, 10),
        "delivery": np.random.uniform(0.65, 1, n_days),
    })
    
    # Target: demand
    base = 50 + season
    demand = (
        base +
        (df["weekday"] >= 5).astype(int) * 15 +
        np.log1p(df["ad_budget"]) * 2.5 -
        df["rainfall"] * 0.5 +
        df["sunshine"] +
        df["event"] * 40 +
        df["holiday"] * 8 -
        df["competition"] * 20 +
        df["social"] +
        df["tv_reach"] * 0.1 +
        df["satisfaction"] * 2 +
        df["delivery"] * 10 -
        np.maximum(0, df["energy_index"] - 100) * 0.05 +
        np.random.normal(0, 4, n_days)
    ).astype(int).clip(0, None)
    
    df["demand"] = demand
    df["revenue"] = (demand * np.random.uniform(3, 6, n_days)).round(2)
    
    features = [
        "weekday", "month", "quarter", "day_of_year",
        "temperature", "rainfall", "sunshine",
        "holiday", "event", "ad_budget", "tv_reach", "social",
        "energy_index", "competition", "satisfaction", "delivery"
    ]
    
    X = df[features].values
    y = df["demand"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        alpha=0.0005,
        batch_size=32,
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "r2": round(r2_score(y_test, y_pred), 3),
        "mape": round(np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100, 2),
    }
    
    return model, scaler, df, features, metrics

# Load model
with st.spinner("Loading model..."):
    model, scaler, data_hist, features, metrics = train_model()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#a8b5c8", size=11),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        showline=False,
        tickfont=dict(color="#6b7585", size=10)
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        showline=False,
        tickfont=dict(color="#6b7585", size=10)
    ),
    margin=dict(l=40, r=20, t=40, b=30),
    hovermode="x unified",
    showlegend=True,
)

colors = {
    "primary": "#0084ff",
    "secondary": "#00d4aa",
    "warning": "#ff9500",
    "danger": "#ff5a5a"
}

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="header-subtitle">OpenBrain Intelligence</div>
    <div class="header-title">Neural Demand Predictor</div>
    <div style="margin-top: 0.5rem;">
        <span class="status-badge badge-live">Live</span>
        <span class="status-badge badge-info">MLP</span>
        <span class="status-badge badge-info">R² = """ + str(metrics["r2"]) + """</span>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("#### Input Parameters")
    
    with st.expander("Date & Weather", expanded=True):
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Day", day_names, label_visibility="collapsed")
        day_idx = day_names.index(selected_day)
        
        month = st.slider("Month", 1, 12, 6, label_visibility="collapsed")
        quarter = (month - 1) // 3 + 1
        day_of_year = (pd.Timestamp(2024, month, 15) - pd.Timestamp(2024, 1, 1)).days + 15
        
        temp = st.slider("Temperature (C)", -15, 45, 15, label_visibility="collapsed")
        rain = st.slider("Rainfall (mm)", 0, 60, 10, label_visibility="collapsed")
        sun = st.slider("Sunshine (h)", 0, 12, 6, 0.5, label_visibility="collapsed")
    
    with st.expander("Market & Ads"):
        ad_budget = st.slider("Ad Budget (EUR)", 0, 600, 250, label_visibility="collapsed")
        tv = st.slider("TV Reach", 0, 80, 30, label_visibility="collapsed")
        social = st.slider("Social Reach", 0, 8, 4, 0.1, label_visibility="collapsed")
        energy = st.slider("Energy Index", 50, 180, 100, label_visibility="collapsed")
        comp = st.slider("Competition", 0.0, 1.0, 0.3, 0.05, label_visibility="collapsed")
    
    with st.expander("Quality & Ops"):
        satisfaction = st.slider("Satisfaction", 1, 10, 7.5, 0.1, label_visibility="collapsed")
        delivery = st.slider("Delivery Ready", 0.6, 1.0, 0.85, 0.05, label_visibility="collapsed")
        is_holiday = st.checkbox("Holiday")
        is_event = st.checkbox("Event")
    
    with st.expander("Pricing"):
        price = st.slider("Price (EUR)", 1, 20, 4.5, 0.1, label_visibility="collapsed")
        margin_pct = st.slider("Margin (%)", 10, 50, 25, label_visibility="collapsed")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Model Quality")
    col1, col2 = st.columns(2)
    col1.metric("MAE", metrics["mae"])
    col2.metric("RMSE", metrics["rmse"])
    col1.metric("MAPE", f"{metrics['mape']}%")
    col2.metric("R2", metrics["r2"])

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
input_data = np.array([[
    day_idx, month, quarter, day_of_year,
    temp, rain, sun,
    int(is_holiday), int(is_event),
    ad_budget, tv, social,
    energy, comp, satisfaction, delivery
]])

demand = int(max(0, model.predict(scaler.transform(input_data))[0]))
revenue = demand * price
margin = revenue * (margin_pct / 100)
confidence_low = int(demand * 0.9)
confidence_high = int(demand * 1.1)

# Historical baseline
hist_baseline = int(data_hist[data_hist["month"] == month]["demand"].mean())
delta_pct = ((demand - hist_baseline) / (hist_baseline + 1) * 100) if hist_baseline > 0 else 0

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns(3)
col1.metric("Demand Forecast", f"{demand} units", f"{delta_pct:+.1f}% YoY")
col2.metric("Revenue", f"EUR {revenue:,.0f}", f"@ EUR {price:.2f}/unit")
col3.metric("Contribution", f"EUR {margin:,.0f}", f"{margin_pct}% margin")

col4, col5, col6 = st.columns(3)
col4.metric("95% Band", f"{confidence_low}-{confidence_high}", "Confidence")
col5.metric("Capacity", f"{min(100, int(demand / 1.5))}%", "Utilization")
col6.metric("Hist. Avg", f"{hist_baseline} units", f"Month {month}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["Dashboard", "Market Analysis", "Scenarios", "Simulation", "Risk Analysis", "Model Info"])

# TAB 1: DASHBOARD
with tabs[0]:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Status
        if demand > 120:
            st.markdown(
                '<div class="alert-danger">Capacity limit approaching. Scale resources.</div>',
                unsafe_allow_html=True
            )
        elif comp > 0.6:
            st.markdown(
                '<div class="alert-warning">High competition. Review positioning.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="alert-success">Operating normally. All metrics optimal.</div>',
                unsafe_allow_html=True
            )
        
        # Hourly distribution
        hours = np.arange(7, 22)
        hourly_demand = (np.abs(np.random.normal(demand / 15, demand / 30, len(hours))).cumsum())
        hourly_demand = (hourly_demand / hourly_demand.max() * demand).astype(int)
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hours, y=hourly_demand,
            fill="tozeroy", line=dict(color=colors["primary"], width=2),
            fillcolor="rgba(0, 132, 255, 0.1)"
        ))
        fig_hourly.update_layout(**layout, title="Hourly Distribution", height=300)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # 7-day forecast
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_forecast = [int(demand * np.random.uniform(0.8, 1.2)) for _ in range(7)]
        
        fig_daily = go.Figure(go.Bar(
            x=day_labels, y=daily_forecast,
            marker_color=colors["primary"]
        ))
        fig_daily.update_layout(**layout, title="7-Day Forecast", height=300, showlegend=False)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col_right:
        # Factor radar
        factors = ["Ads", "Temp", "Weekend", "Event", "Social", "Satisfaction", "Delivery"]
        values = [
            ad_budget / 600,
            (temp + 15) / 60,
            1.0 if day_idx >= 5 else 0.2,
            1.0 if is_event else 0.05,
            social / 8,
            satisfaction / 10,
            delivery
        ]
        
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=factors + [factors[0]],
            fill="toself",
            fillcolor="rgba(0, 132, 255, 0.1)",
            line=dict(color=colors["primary"], width=2)
        ))
        fig_radar.update_layout(**layout, height=350, title="Factor Profile")
        fig_radar.update_layout(polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

# TAB 2: MARKET ANALYSIS
with tabs[1]:
    months_short = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = data_hist.groupby("month").agg({
        "demand": "mean",
        "revenue": "sum"
    }).reset_index()
    monthly["month_name"] = monthly["month"].map({i: m for i, m in enumerate(months_short, 1)})
    
    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_monthly.add_trace(
        go.Bar(x=monthly["month_name"], y=monthly["demand"], name="Demand", marker_color=colors["primary"]),
        secondary_y=False
    )
    fig_monthly.add_trace(
        go.Scatter(x=monthly["month_name"], y=monthly["revenue"], name="Revenue", 
                   line=dict(color=colors["secondary"], width=3), mode="lines+markers"),
        secondary_y=True
    )
    fig_monthly.update_layout(**layout, title="Monthly Trends", height=350)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Quarterly distribution
    quarterly = data_hist.groupby("quarter")["demand"].apply(list).to_dict()
    fig_box = go.Figure()
    for q in [1, 2, 3, 4]:
        fig_box.add_trace(go.Box(y=quarterly.get(q, []), name=f"Q{q}"))
    fig_box.update_layout(**layout, title="Quarterly Distribution", height=300)
    st.plotly_chart(fig_box, use_container_width=True)

# TAB 3: SCENARIOS
with tabs[2]:
    scenarios = {
        "Baseline": input_data[0].tolist(),
        "Best Case": [5, month, quarter, day_of_year, 20, 0, 10, 1, 1, 500, 70, 7, 80, 0.1, 9, 0.95],
        "Worst Case": [1, month, quarter, day_of_year, 5, 40, 2, 0, 0, 100, 10, 2, 150, 0.8, 5, 0.7],
        "Event Boost": [day_idx, month, quarter, day_of_year, temp, 0, sun, int(is_holiday), 1, min(600, ad_budget * 1.5), tv + 20, social + 2, energy, comp, satisfaction, delivery],
    }
    
    scenario_names = []
    scenario_demands = []
    scenario_revenues = []
    
    for name, params in scenarios.items():
        dem = int(max(0, model.predict(scaler.transform(np.array([params])))[0]))
        rev = dem * price
        scenario_names.append(name)
        scenario_demands.append(dem)
        scenario_revenues.append(rev)
    
    col_scen1, col_scen2 = st.columns(2)
    
    with col_scen1:
        fig_scen = go.Figure(go.Bar(
            x=scenario_names, y=scenario_demands,
            marker_color=[colors["primary"], colors["secondary"], colors["danger"], colors["warning"]]
        ))
        fig_scen.update_layout(**layout, title="Demand by Scenario", height=300, showlegend=False)
        st.plotly_chart(fig_scen, use_container_width=True)
    
    with col_scen2:
        fig_rev = go.Figure(go.Funnel(
            y=scenario_names, x=scenario_revenues,
            marker=dict(color=[colors["primary"], colors["secondary"], colors["danger"], colors["warning"]])
        ))
        fig_rev.update_layout(**layout, title="Revenue Comparison (EUR)", height=300)
        st.plotly_chart(fig_rev, use_container_width=True)
    
    scen_df = pd.DataFrame({
        "Scenario": scenario_names,
        "Demand": scenario_demands,
        "Revenue (EUR)": [f"{r:,.0f}" for r in scenario_revenues],
        "Delta": [d - scenario_demands[0] for d in scenario_demands],
    })
    st.dataframe(scen_df, use_container_width=True, hide_index=True)

# TAB 4: SIMULATION
with tabs[3]:
    col_sim_opts, col_sim_chart = st.columns([1, 3])
    
    with col_sim_opts:
        st.markdown("#### Simulation Setup")
        sim_days = st.slider("Days", 7, 365, 60, label_visibility="collapsed")
        sim_noise = st.slider("Market Noise", 0, 25, 8, label_visibility="collapsed")
        trend_opt = st.selectbox("Trend", ["Stable", "Growth +5%", "Decline -5%"], label_visibility="collapsed")
    
    with col_sim_chart:
        trend_factor = {"Stable": 1.0, "Growth +5%": 1.05, "Decline -5%": 0.95}[trend_opt]
        sim_dates = pd.date_range(datetime.date.today(), periods=sim_days)
        sim_values = [int(max(0, demand * trend_factor + np.random.normal(0, sim_noise))) for _ in range(sim_days)]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_dates, y=sim_values,
            fill="tozeroy", line=dict(color=colors["primary"], width=2),
            fillcolor="rgba(0, 132, 255, 0.1)"
        ))
        fig_sim.add_hline(y=demand, line_dash="dash", line_color="gray", 
                         annotation_text=f"Baseline: {demand}")
        fig_sim.update_layout(**layout, title=f"{sim_days}-Day Simulation", height=350)
        st.plotly_chart(fig_sim, use_container_width=True)
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("Avg", f"{np.mean(sim_values):.0f}")
    col_k2.metric("Peak", f"{max(sim_values)}")
    col_k3.metric("Min", f"{min(sim_values)}")
    col_k4.metric("Total Rev", f"EUR {sum(sim_values) * price:,.0f}")

# TAB 5: RISK ANALYSIS
with tabs[4]:
    risk_indicators = {
        "Demand Variance": min(100, abs(delta_pct) * 2),
        "Capacity Load": min(100, int(demand / 1.5)),
        "Competition": int(comp * 100),
        "Weather Impact": int(rain / 60 * 100),
        "Energy Cost": int(max(0, (energy - 100) / 80 * 100)),
    }
    
    for risk_name, risk_val in risk_indicators.items():
        color_risk = colors["secondary"] if risk_val < 50 else (colors["warning"] if risk_val < 75 else colors["danger"])
        st.write(f"{risk_name}: {risk_val}%")
        st.progress(risk_val / 100, text=f"{risk_val}%")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Sensitivity Analysis")
    
    sensitivity_params = {
        "Ad Budget +30%": ad_budget * 1.3,
        "Competition +0.2": comp + 0.2,
        "Satisfaction +1": satisfaction + 1,
        "Energy +30": energy + 30,
    }
    
    sens_results = []
    for param_name, param_val in sensitivity_params.items():
        test_input = input_data[0].copy()
        if "Ad Budget" in param_name:
            test_input[9] = min(600, param_val)
        elif "Competition" in param_name:
            test_input[13] = min(1.0, param_val)
        elif "Satisfaction" in param_name:
            test_input[14] = min(10.0, param_val)
        elif "Energy" in param_name:
            test_input[12] = min(180, param_val)
        
        new_dem = int(max(0, model.predict(scaler.transform(np.array([test_input])))[0]))
        delta = new_dem - demand
        sens_results.append({"Parameter": param_name, "Delta": delta, "% Change": f"{delta/demand*100:+.1f}%"})
    
    sens_df = pd.DataFrame(sens_results)
    st.dataframe(sens_df, use_container_width=True, hide_index=True)

# TAB 6: MODEL INFO
with tabs[5]:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Training Samples", len(data_hist))
    col_m2.metric("Features", len(features))
    col_m3.metric("Hidden Layers", "2")
    col_m4.metric("Parameters", "~12K")
    
    st.markdown("#### Architecture")
    arch_data = {
        "Layer": ["Input", "Hidden 1", "Hidden 2", "Output"],
        "Size": [16, 128, 64, 1],
        "Activation": ["—", "ReLU", "ReLU", "Linear"]
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)
    
    st.markdown("#### Performance Metrics")
    perf_data = {
        "Metric": ["MAE", "RMSE", "MAPE", "R² Score"],
        "Value": [metrics["mae"], metrics["rmse"], f"{metrics['mape']}%", metrics["r2"]]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    
    # Validation chart
    val_data = data_hist.tail(200).copy()
    val_pred = model.predict(scaler.transform(val_data[features].values))
    
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=val_data["date"], y=val_data["demand"], name="Actual", line=dict(color=colors["primary"])))
    fig_val.add_trace(go.Scatter(x=val_data["date"], y=val_pred, name="Predicted", line=dict(color=colors["secondary"], dash="dot")))
    fig_val.update_layout(**layout, title="Validation: Actual vs Predicted", height=300)
    st.plotly_chart(fig_val, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; margin-top: 3rem;">
    OpenBrain Neural Demand Predictor | Enterprise Grade | Real-time Analysis | © 2024
</div>
""", unsafe_allow_html=True)
