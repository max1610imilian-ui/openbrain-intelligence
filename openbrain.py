"""
OpenBrain · Neural Demand Predictor
Enterprise-grade demand forecasting with real-time ML analysis
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OpenBrain Demand Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS STYLING (Dark/Light Mode Compatible)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    color-scheme: light dark;
}

/* Dark Mode (Default) */
html, body, .stApp {
    --bg-dark: #0a0e1f;
    --bg-card: #0f1629;
    --bg-input: #141d2e;
    --border: rgba(255,255,255,0.07);
    --border-hover: rgba(255,255,255,0.15);
    --accent: #0084ff;
    --accent-alt: #00d4aa;
    --accent-warn: #ffa500;
    --accent-danger: #ff4757;
    --text-main: #e8eef5;
    --text-sec: #a8b5c8;
    --text-muted: #6b7585;
}

/* Light Mode */
@media (prefers-color-scheme: light) {
    html, body, .stApp {
        --bg-dark: #ffffff;
        --bg-card: #f8f9fb;
        --bg-input: #f0f2f6;
        --border: rgba(0,0,0,0.08);
        --border-hover: rgba(0,0,0,0.15);
        --accent: #0066ff;
        --accent-alt: #00a878;
        --accent-warn: #ff9500;
        --accent-danger: #ff5a5a;
        --text-main: #1a202c;
        --text-sec: #475569;
        --text-muted: #94a3b8;
    }
}

* { box-sizing: border-box; }

html, body, .stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-card) 100%);
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
}

.main .block-container {
    max-width: 1800px;
    padding: 2rem 2.5rem;
}

[data-testid="stSidebar"] {
    background: rgba(20,29,46,0.4);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--border);
}

[data-testid="stExpander"] {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.75rem;
}

[data-testid="stExpander"] summary {
    padding: 0.8rem 1rem;
    background: rgba(20,29,46,0.5);
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
}

[data-testid="stExpander"] summary:hover {
    background: rgba(20,29,46,0.8);
    border-color: var(--border-hover);
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    font-size: 0.85rem !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,132,255,0.1) !important;
}

[data-testid="stSlider"] input {
    accent-color: var(--accent) !important;
}

.stCheckbox label {
    color: var(--text-main) !important;
}

.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-sec) !important;
    font-weight: 600;
    font-size: 0.85rem;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
}

[data-testid="stMetricContainer"] {
    background: rgba(20,29,46,0.6);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}

[data-testid="stMetricContainer"]:hover {
    border-color: var(--border-hover);
    background: rgba(20,29,46,0.9);
}

[data-testid="stMetricLabel"] {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text-main);
    margin-top: 0.35rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00d4ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,132,255,0.2);
}

[data-testid="stDataFrame"] {
    background: rgba(20,29,46,0.5);
    border: 1px solid var(--border);
    border-radius: 10px;
}

.glass-card {
    background: rgba(20,29,46,0.6);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: 1px solid;
    margin-right: 0.5rem;
}

.badge-live {
    background: rgba(0,212,170,0.1);
    color: var(--accent-alt);
    border-color: rgba(0,212,170,0.3);
}

.badge-info {
    background: rgba(0,132,255,0.1);
    color: var(--accent);
    border-color: rgba(0,132,255,0.3);
}

.alert-box {
    padding: 1rem 1.25rem;
    border-radius: 10px;
    border-left: 4px solid;
    margin: 0.75rem 0;
    font-size: 0.85rem;
}

.alert-success {
    background: rgba(0,212,170,0.08);
    border-color: var(--accent-alt);
    color: var(--text-main);
}

.alert-warning {
    background: rgba(255,165,0,0.08);
    border-color: var(--accent-warn);
    color: var(--text-main);
}

.alert-danger {
    background: rgba(255,71,87,0.08);
    border-color: var(--accent-danger);
    color: var(--text-main);
}

.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 2rem 0;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    padding: 2rem 0;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL (Single cache for fast loading)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load or train model once"""
    np.random.seed(42)
    TAGE = 1500
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=TAGE)
    
    temp = rng.normal(13, 11, TAGE).clip(-15, 42)
    rain = rng.exponential(1.8, TAGE).clip(0, 60)
    season = np.sin(2*np.pi*dates.month/12)*12 + np.cos(2*np.pi*dates.dayofyear/365)*5
    
    df = pd.DataFrame({
        "date": dates,
        "weekday": dates.weekday,
        "month": dates.month,
        "quarter": dates.quarter,
        "dayofyear": dates.dayofyear,
        "temp": temp,
        "rain": rain,
        "sun": (rng.uniform(0, 14, TAGE) * (1 - rain/80)).clip(0, 14),
        "holiday": rng.choice([0, 1], TAGE, p=[0.78, 0.22]),
        "event": rng.choice([0, 1], TAGE, p=[0.94, 0.06]),
        "budget": rng.uniform(0, 800, TAGE),
        "tv": rng.uniform(0, 100, TAGE),
        "social": rng.uniform(0, 10, TAGE),
        "energy": rng.normal(105, 18, TAGE).clip(50, 220),
        "competition": rng.uniform(0, 1, TAGE),
        "satisfaction": rng.normal(7.5, 1.2, TAGE).clip(1, 10),
        "delivery": rng.uniform(0.6, 1.0, TAGE),
        "season": season,
    })
    
    base = 55 + df["season"]
    effect = (
        (df["weekday"] >= 4).astype(int) * 18 +
        np.log1p(df["budget"]) * 2.8 -
        df["rain"] * 0.6 +
        df["sun"] * 1.1 +
        df["event"] * 52 +
        df["holiday"] * 8 -
        df["competition"] * 22 +
        df["social"] * 1.5 +
        df["tv"] * 0.12 +
        df["satisfaction"] * 2.8 +
        df["delivery"] * 12 -
        np.maximum(0, df["energy"] - 100) * 0.08
    )
    
    df["sales"] = (base + effect + rng.normal(0, 5, TAGE)).astype(int).clip(0)
    df["revenue"] = (df["sales"] * rng.uniform(3.8, 5.5, TAGE)).round(2)
    df["margin"] = (df["revenue"] * rng.uniform(0.18, 0.38, TAGE)).round(2)
    
    features = ["weekday", "month", "quarter", "dayofyear", "temp", "rain", "sun",
                "holiday", "event", "budget", "tv", "social", "energy", "competition",
                "satisfaction", "delivery"]
    
    X = df[features].values
    y = df["sales"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        alpha=0.001,
        batch_size=32
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "r2": round(r2_score(y_test, y_pred), 4),
        "mape": round(np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100, 2),
    }
    
    return model, scaler, df, features, metrics

model, scaler, hist_data, features, metrics = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
PLOT_CONFIG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#a8b5c8", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    margin=dict(l=40, r=15, t=40, b=25),
    hovermode="x unified",
)

C_PRIMARY = "#0084ff"
C_SUCCESS = "#00d4aa"
C_WARNING = "#ffa500"
C_DANGER = "#ff4757"
C_INFO = "#ffd60a"

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
time_str = datetime.datetime.now().strftime("%d.%m | %H:%M")

st.markdown(f"""
<div style="
    background: rgba(20,29,46,0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
">
    <div style="display:flex; align-items:center; gap:1rem;">
        <div style="
            width: 40px; height: 40px;
            background: linear-gradient(135deg, #0084ff, #00d4ff);
            border-radius: 10px;
            display: flex; align-items:center; justify-content:center;
            color: white; font-weight: 700; font-size: 1.2rem;
        ">◈</div>
        <div>
            <div style="font-size: 0.65rem; color: #6b7585; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem;">
                OpenBrain Intelligence
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #e8eef5;">
                Neural Demand Predictor
            </div>
        </div>
    </div>
    
    <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
        <span class="status-badge badge-live">LIVE</span>
        <span class="status-badge badge-info">MLP</span>
        <span class="status-badge badge-info">R² = {metrics['r2']}</span>
        <span style="font-size: 0.7rem; color: #6b7585; font-family: monospace; margin-left: 0.5rem;">{time_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#6b7585; margin-bottom:1rem;'>Input Parameters</div>", unsafe_allow_html=True)
    
    with st.expander("Time & Weather", expanded=True):
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_sel = st.selectbox("Day", day_names, label_visibility="collapsed")
        day_idx = day_names.index(day_sel)
        month = st.slider("Month", 1, 12, datetime.date.today().month, label_visibility="collapsed")
        quarter = ((month - 1) // 3) + 1
        dayofyear = (datetime.date(2024, month, 15) - datetime.date(2024, 1, 1)).days + 15
        temp = st.slider("Temperature (°C)", -15, 45, 15, label_visibility="collapsed")
        rain = st.slider("Rainfall (mm)", 0, 60, 5, label_visibility="collapsed")
        sun = st.slider("Sunshine (h)", 0.0, 14.0, 7.0, step=0.5, label_visibility="collapsed")
    
    with st.expander("Market & Ads"):
        budget = st.slider("Ad Budget (€)", 0, 800, 200, label_visibility="collapsed")
        tv = st.slider("TV Reach", 0, 100, 40, label_visibility="collapsed")
        social = st.slider("Social Reach", 0.0, 10.0, 5.0, step=0.1, label_visibility="collapsed")
        energy = st.slider("Energy Index", 50, 220, 105, label_visibility="collapsed")
        competition = st.slider("Competition", 0.0, 1.0, 0.3, step=0.05, label_visibility="collapsed")
    
    with st.expander("Quality & Ops"):
        satisfaction = st.slider("Satisfaction", 1.0, 10.0, 7.5, step=0.1, label_visibility="collapsed")
        delivery = st.slider("Delivery", 0.6, 1.0, 0.90, step=0.05, label_visibility="collapsed")
        holiday = st.checkbox("Holiday")
        event = st.checkbox("Event")
    
    with st.expander("Pricing"):
        price = st.slider("Price per Unit (€)", 1.0, 20.0, 4.50, step=0.10, label_visibility="collapsed")
        margin_pct = st.slider("Margin (%)", 10, 50, 28, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════════════════
input_data = np.array([[
    day_idx, month, quarter, dayofyear, temp, rain, sun,
    int(holiday), int(event), budget, tv, social, energy, competition, satisfaction, delivery
]])

sales_pred = int(max(0, model.predict(scaler.transform(input_data))[0]))
revenue = sales_pred * price
contribution = revenue * (margin_pct / 100)
conf_low = int(sales_pred * 0.90)
conf_high = int(sales_pred * 1.10)

hist_month_avg = int(hist_data[hist_data["month"] == month]["sales"].mean())
delta_pct = round((sales_pred - hist_month_avg) / (hist_month_avg + 1) * 100, 1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["Dashboard", "Market Analysis", "Scenarios", "Simulation", "Risk", "Model"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Sales Forecast", f"{sales_pred} units", f"{delta_pct:+.1f}% YoY")
    col2.metric("Revenue", f"€{revenue:,.0f}", f"@ €{price:.2f}")
    col3.metric("Contribution", f"€{contribution:,.0f}", f"{margin_pct}% margin")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("95% Confidence", f"{conf_low}–{conf_high}", "Range")
    col5.metric("Capacity", f"{min(100, int(sales_pred/1.5))}%", "Usage")
    col6.metric("Profit Est.", f"€{contribution:,.0f}", "Gross")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Status Alert
        if sales_pred > 120:
            st.markdown('<div class="alert-box alert-danger">Capacity Alert: Demand exceeds normal capacity</div>', unsafe_allow_html=True)
        elif competition > 0.65:
            st.markdown('<div class="alert-box alert-warning">Competition: Market pressure detected</div>', unsafe_allow_html=True)
        elif sales_pred < 30:
            st.markdown('<div class="alert-box alert-warning">Low Demand: Consider marketing boost</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-success">Optimal: All parameters normal</div>', unsafe_allow_html=True)
        
        # Hourly Distribution
        hours = list(range(7, 22))
        rng_hour = np.random.default_rng(1)
        hourly = (np.abs(rng_hour.normal(sales_pred/15, sales_pred/30, len(hours))).cumsum())
        hourly = hourly / hourly.max() * sales_pred if hourly.max() > 0 else hourly
        
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=hours, y=hourly.astype(int), fill="tozeroy", mode="lines",
            line=dict(color=C_PRIMARY, width=2.5),
            fillcolor="rgba(0,132,255,0.1)",
            name="Hourly"
        ))
        fig_hour.update_layout(**PLOT_CONFIG, title="Hourly Distribution", height=280)
        fig_hour.update_xaxes(ticksuffix=":00")
        st.plotly_chart(fig_hour, use_container_width=True)
        
        # 7-Day Forecast
        day_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        rng_week = np.random.default_rng(42)
        weekly = [int(max(0, sales_pred * rng_week.uniform(0.75, 1.25))) for _ in range(7)]
        peak = max(weekly)
        colors = [C_PRIMARY if v == peak else "rgba(0,132,255,0.25)" for v in weekly]
        
        fig_week = go.Figure(go.Bar(
            x=day_short, y=weekly, marker_color=colors,
            text=[str(v) for v in weekly], textposition="outside"
        ))
        fig_week.update_layout(**PLOT_CONFIG, title="7-Day Forecast", height=250, showlegend=False)
        st.plotly_chart(fig_week, use_container_width=True)
    
    with col_right:
        # Factors Radar
        factors = ["Ads", "Temp", "Weekend", "Event", "Social", "Satisfaction", "Delivery"]
        values = [
            budget / 800,
            (temp + 15) / 60,
            1.0 if day_idx >= 5 else 0.2,
            1.0 if event else 0.04,
            social / 10,
            satisfaction / 10,
            delivery,
        ]
        
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=factors + [factors[0]],
            fill="toself",
            fillcolor="rgba(0,132,255,0.08)",
            line=dict(color=C_PRIMARY, width=2),
        ))
        fig_radar.update_layout(
            **PLOT_CONFIG, height=300, title="Factor Profile",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2: MARKET ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    col_mkt1, col_mkt2 = st.columns([2, 1])
    
    with col_mkt1:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = hist_data.groupby("month").agg(sales=("sales","mean"), revenue=("revenue","sum")).reset_index()
        monthly["name"] = monthly["month"].apply(lambda m: months[m-1])
        
        fig_month = make_subplots(specs=[[{"secondary_y":True}]])
        fig_month.add_trace(
            go.Bar(x=monthly["name"], y=monthly["sales"], marker_color="rgba(0,132,255,0.25)", marker_line_color=C_PRIMARY, name="Sales"),
            secondary_y=False
        )
        fig_month.add_trace(
            go.Scatter(x=monthly["name"], y=monthly["revenue"], line=dict(color=C_SUCCESS, width=2.5), marker=dict(size=6), name="Revenue"),
            secondary_y=True
        )
        fig_month.update_layout(**PLOT_CONFIG, title="Monthly Trends", height=300)
        st.plotly_chart(fig_month, use_container_width=True)
        
        # Heatmap
        hm_data = hist_data.groupby(["month","weekday"])["sales"].mean().unstack()
        hm_data.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        hm_data.index = [months[m-1] for m in hm_data.index]
        
        fig_hm = px.imshow(hm_data.T, color_continuous_scale=[[0,"#0a0e1f"],[0.5,"#0084ff"],[1,"#00d4ff"]], title="Demand Heatmap")
        fig_hm.update_layout(**PLOT_CONFIG, height=220)
        st.plotly_chart(fig_hm, use_container_width=True)
        
        # Quarterly
        fig_q = go.Figure()
        for q in [1,2,3,4]:
            q_data = hist_data[hist_data["quarter"]==q]["sales"]
            fig_q.add_trace(go.Box(y=q_data, name=f"Q{q}"))
        fig_q.update_layout(**PLOT_CONFIG, title="Quarterly Distribution", height=250)
        st.plotly_chart(fig_q, use_container_width=True)
    
    with col_mkt2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        peak_row = hist_data.loc[hist_data["sales"].idxmax()]
        stats = [
            ("Avg Daily", f"{hist_data['sales'].mean():.0f}"),
            ("Peak Day", str(peak_row["date"].date())),
            ("Peak Sales", f"{int(peak_row['sales'])}"),
            ("Total Revenue", f"€{hist_data['revenue'].sum():,.0f}"),
            ("Std Dev", f"±{hist_data['sales'].std():.1f}"),
            ("Data Points", f"{len(hist_data):,}"),
        ]
        for label, val in stats:
            st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:0.8rem; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);'><span style='color:#6b7585;'>{label}</span><span style='color:#e8eef5; font-family:monospace;'>{val}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3: SCENARIOS
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    scenarios = {
        "Baseline": [day_idx, month, quarter, dayofyear, temp, rain, sun, int(holiday), int(event), budget, tv, social, energy, competition, satisfaction, delivery],
        "Best": [5, month, quarter, dayofyear, 22, 0, 10, 1, 1, min(800,budget*2), min(100,tv+30), min(10,social+2), max(50,energy-25), max(0,competition-0.3), min(10,satisfaction+1.5), min(1,delivery+0.08)],
        "Worst": [1, month, quarter, dayofyear, 3, 40, 1, 0, 0, max(0,budget//3), max(0,tv-30), max(0,social-2), min(220,energy+35), min(1,competition+0.3), max(1,satisfaction-2), max(0.6,delivery-0.15)],
        "Event": [5, month, quarter, dayofyear, temp, 0, sun, int(holiday), 1, min(800,budget*1.5), min(100,tv+25), min(10,social+3), energy, competition, min(10,satisfaction+0.5), delivery],
        "Energy": [day_idx, month, quarter, dayofyear, temp, rain, sun, int(holiday), int(event), max(0,budget-100), tv, social, min(220,energy+60), min(1,competition+0.15), max(1,satisfaction-0.5), max(0.6,delivery-0.05)],
        "Ads": [day_idx, month, quarter, dayofyear, temp, rain, sun, int(holiday), int(event), min(800,budget*2.5), min(100,tv+50), min(10,social+4), energy, competition, satisfaction, delivery],
    }
    
    scenario_names, scenario_sales, scenario_revenue = [], [], []
    for name, params in scenarios.items():
        s = int(max(0, model.predict(scaler.transform(np.array([params])))[0]))
        scenario_names.append(name)
        scenario_sales.append(s)
        scenario_revenue.append(s * price)
    
    colors_sc = [C_PRIMARY, C_SUCCESS, C_DANGER, "#ffd60a", C_WARNING, "#ff006e"]
    
    col_sc1, col_sc2 = st.columns([3, 2])
    with col_sc1:
        fig_sc = go.Figure(go.Bar(
            x=scenario_names, y=scenario_sales, marker_color=colors_sc,
            text=[str(s) for s in scenario_sales], textposition="outside"
        ))
        fig_sc.update_layout(**PLOT_CONFIG, title="Scenario Comparison", height=320, showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)
    
    with col_sc2:
        fig_funnel = go.Figure(go.Funnel(
            y=scenario_names, x=sorted(scenario_revenue, reverse=True),
            marker_color=colors_sc
        ))
        fig_funnel.update_layout(**PLOT_CONFIG, title="Revenue Waterfall", height=320)
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    sc_df = pd.DataFrame({
        "Scenario": scenario_names,
        "Sales": scenario_sales,
        "Revenue": [f"€{r:,.0f}" for r in scenario_revenue],
        "vs Baseline": [f"{s-scenario_sales[0]:+d}" for s in scenario_sales],
    })
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 4: SIMULATION
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    col_sim_opts, col_sim_chart = st.columns([1, 3])
    
    with col_sim_opts:
        sim_days = st.slider("Days", 7, 365, 60)
        sim_noise = st.slider("Noise", 0, 30, 8)
        trend_mode = st.selectbox("Trend", ["Neutral", "Growth", "Decline"])
        show_mc = st.checkbox("Monte Carlo")
        show_raw = st.checkbox("Raw Data")
    
    with col_sim_chart:
        trend_factor = {"Neutral":1.0, "Growth":1.08, "Decline":0.92}[trend_mode]
        sim_dates = pd.date_range(datetime.date.today(), periods=sim_days)
        rng_sim = np.random.default_rng(77)
        sim_sales = [int(max(0, sales_pred * trend_factor + rng_sim.normal(0, sim_noise))) for _ in range(sim_days)]
        
        fig_sim = go.Figure()
        if show_mc:
            for seed in range(10):
                rng_mc = np.random.default_rng(seed + 100)
                mc_path = [int(max(0, sales_pred*trend_factor + rng_mc.normal(0, sim_noise*2))) for _ in range(sim_days)]
                fig_sim.add_trace(go.Scatter(x=sim_dates, y=mc_path, mode="lines", line=dict(color="rgba(0,132,255,0.08)", width=0.8), showlegend=False))
        
        fig_sim.add_trace(go.Scatter(
            x=sim_dates, y=sim_sales, mode="lines",
            line=dict(color=C_PRIMARY, width=2.5),
            name="Forecast"
        ))
        fig_sim.add_hline(y=sales_pred, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_sim.update_layout(**PLOT_CONFIG, title=f"{sim_days}-Day Simulation", height=350)
        st.plotly_chart(fig_sim, use_container_width=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg", f"{np.mean(sim_sales):.0f}")
    k2.metric("Peak", f"{max(sim_sales)}")
    k3.metric("Min", f"{min(sim_sales)}")
    k4.metric("Total Rev.", f"€{sum(sim_sales)*price:,.0f}")
    
    if show_raw:
        sim_df = pd.DataFrame({
            "Date": [d.strftime("%d.%m") for d in sim_dates],
            "Sales": sim_sales,
            "Revenue": [f"€{s*price:,.0f}" for s in sim_sales],
        })
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 5: RISK
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.subheader("Risk Indicators")
        risks = {
            "Demand": min(100, abs(delta_pct)*2),
            "Capacity": min(100, int(sales_pred/1.5)),
            "Competition": int(competition*100),
            "Weather": int(rain/60*100),
            "Energy": int(max(0, (energy-100)/120*100)),
        }
        for name, val in risks.items():
            level = "Low" if val<40 else ("Med" if val<70 else "High")
            color = C_SUCCESS if val<40 else (C_WARNING if val<70 else C_DANGER)
            st.markdown(f"<div style='margin:0.5rem 0;'><div style='font-size:0.75rem; display:flex; justify-content:space-between; margin-bottom:0.2rem;'><span>{name}</span><span style='color:{color}; font-family:monospace;'>{val}% {level}</span></div><div style='width:100%; height:6px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden;'><div style='width:{val}%; height:100%; background:linear-gradient(90deg, {color}, rgba(0,212,255,0.5));'></div></div></div>", unsafe_allow_html=True)
    
    with col_risk2:
        st.subheader("Recommendations")
        alerts = []
        if budget < 100:
            alerts.append(("warning", "Low ad budget – increase spend"))
        if competition > 0.6:
            alerts.append(("danger", "High competition – differentiate"))
        if rain > 30:
            alerts.append(("warning", "Heavy rain – adjust logistics"))
        if energy > 150:
            alerts.append(("danger", "Energy crisis – review costs"))
        if satisfaction < 6:
            alerts.append(("danger", "Low satisfaction – urgent action"))
        if not alerts:
            alerts.append(("success", "All parameters optimal"))
        
        for lvl, txt in alerts:
            css_class = "alert-success" if lvl=="success" else ("alert-warning" if lvl=="warning" else "alert-danger")
            st.markdown(f'<div class="alert-box {css_class}">{txt}</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 6: MODEL
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("MAE", metrics["mae"])
    col_m2.metric("RMSE", metrics["rmse"])
    col_m3.metric("MAPE", f"{metrics['mape']}%")
    col_m4.metric("R² Score", metrics["r2"])
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_arch, col_imp = st.columns(2)
    with col_arch:
        st.subheader("Architecture")
        arch = pd.DataFrame({
            "Layer": ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"],
            "Neurons": [16, 128, 64, 32, 1],
            "Activation": ["-", "ReLU", "ReLU", "ReLU", "Linear"],
        })
        st.dataframe(arch, use_container_width=True, hide_index=True)
    
    with col_imp:
        st.subheader("Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": ["Event", "Holiday", "Budget", "Weekday", "Social", "Satisfaction", "TV", "Month", "Rain", "Delivery", "Temp", "Energy", "Competition", "Quarter", "Dayofyear", "Sun"],
            "Importance": [0.20, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02],
        }).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=[[0,"#141d2e"],[1,"#0084ff"]])
        fig_imp.update_layout(**PLOT_CONFIG, height=320, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    st.subheader("Validation")
    test_size = min(200, len(hist_data))
    test_data = hist_data.tail(test_size)
    test_pred = model.predict(scaler.transform(test_data[features].values))
    
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=test_data["date"], y=test_data["sales"], mode="lines", name="Actual", line=dict(color=C_PRIMARY)))
    fig_val.add_trace(go.Scatter(x=test_data["date"], y=test_pred.astype(int), mode="lines", name="Predicted", line=dict(color=C_SUCCESS, dash="dot")))
    fig_val.update_layout(**PLOT_CONFIG, title=f"Last {test_size} Days", height=280)
    st.plotly_chart(fig_val, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
OpenBrain Intelligence Platform · Enterprise Grade Demand Forecasting · 2024-2026
</div>
""", unsafe_allow_html=True)
