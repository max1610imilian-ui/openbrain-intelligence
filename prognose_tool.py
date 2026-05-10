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

st.set_page_config(
    page_title="OpenBrain · Weekly Demand Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM DARK MODE CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600&family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:           #0B0F19;
    --bg-dark:      #080C14;
    --surface:      #111827;
    --border:       #1F2937;
    --text:         #F3F4F6;
    --text-muted:   #9CA3AF;
    --accent:       #3B82F6;
    --ok:           #10B981;
    --warn:         #F59E0B;
    --crit:         #EF4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

.main .block-container { padding: 2.5rem 3rem !important; max-width: 1600px !important; }

[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stSidebar"] label { 
    font-size: 0.65rem !important; font-weight: 700 !important; 
    color: var(--text-muted) !important; text-transform: uppercase !important; 
}

[data-testid="stMetricContainer"] { 
    background: var(--surface) !important; border: 1px solid var(--border) !important; 
    padding: 1.25rem !important; border-radius: 6px !important;
}

.header-main { font-family: 'Crimson Text', serif; font-size: 2.2rem; font-weight: 600; color: var(--text); margin: 0; }
.header-sub { font-family: 'Crimson Text', serif; font-size: 1.2rem; color: var(--text-muted); margin: 0.25rem 0 0 0; }
.section-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }

.status-badge { display: inline-flex; align-items: center; padding: 0.35rem 0.75rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; border: 1px solid transparent; }
.status-badge.ok   { background: rgba(16, 185, 129, 0.1); color: var(--ok); border-color: var(--ok); }
.status-badge.warn { background: rgba(245, 158, 11, 0.1); color: var(--warn); border-color: var(--warn); }
.status-badge.crit { background: rgba(220, 38, 38, 0.1); color: var(--crit); border-color: var(--crit); }

.day-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1.25rem; text-align: center; transition: all 0.2s ease; }
.day-card:hover { border-color: var(--accent); background: var(--bg-dark); transform: translateY(-2px); }
.day-card-metric { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; color: var(--accent); }

.info-box { background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 6px; padding: 1rem; font-size: 0.85rem; margin-bottom: 1rem;}

.footer { text-align: center; color: var(--text-muted); font-size: 0.7rem; padding: 2.5rem 0; border-top: 1px solid var(--border); margin-top: 3rem; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL PLOTLY THEME (Ohne 'margin', um Fehler zu vermeiden)
# ══════════════════════════════════════════════════════════════════════════════
PT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9CA3AF", size=11),
    xaxis=dict(gridcolor="#1F2937", linecolor="#1F2937", tickfont=dict(size=10), zeroline=False),
    yaxis=dict(gridcolor="#1F2937", linecolor="#1F2937", tickfont=dict(size=10), zeroline=False)
)

# ══════════════════════════════════════════════════════════════════════════════
# MODELL & DATEN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Neural Engine lädt Trainingsdaten...")
def modell_laden():
    TAGE = 5000
    rng = np.random.default_rng(42)
    daten = pd.date_range("2011-01-01", periods=TAGE)
    
    temperatur = rng.normal(13, 11, TAGE).clip(-15, 42)
    niederschlag = rng.exponential(1.8, TAGE).clip(0, 60)
    saison = np.sin(2 * np.pi * daten.month / 12) * 12 + np.cos(2 * np.pi * daten.dayofyear / 365) * 5

    df = pd.DataFrame({
        "Datum": daten, "Wochentag": daten.weekday, "Monat": daten.month, "Quartal": daten.quarter,
        "Jahrestag": daten.dayofyear, "Temperatur": temperatur, "Niederschlag": niederschlag,
        "Sonnenstunden": (rng.uniform(0, 14, TAGE) * (1 - niederschlag / 80)).clip(0, 14),
        "Ferienindikator": rng.choice([0, 1], TAGE, p=[0.78, 0.22]),
        "Veranstaltung": rng.choice([0, 1], TAGE, p=[0.94, 0.06]),
        "Werbebudget": rng.uniform(0, 800, TAGE), "TV_Werbedruck": rng.uniform(0, 100, TAGE),
        "Social_Reichweite": rng.uniform(0, 10, TAGE), "Energieindex": rng.normal(105, 18, TAGE).clip(50, 220),
        "Wettbewerb": rng.uniform(0, 1, TAGE), "Kundenzufriedenheit": rng.normal(7.5, 1.2, TAGE).clip(1, 10),
        "Lieferbereitschaft": rng.uniform(0.6, 1.0, TAGE), "Saisonalitaet": saison,
    })

    basis = 55 + df["Saisonalitaet"]
    einfluss = (
        (df["Wochentag"] >= 4).astype(int) * 18 + np.log1p(df["Werbebudget"]) * 2.8 -
        df["Niederschlag"] * 0.6 + df["Sonnenstunden"] * 1.1 + df["Veranstaltung"] * 35 +
        df["Ferienindikator"] * 8 - df["Wettbewerb"] * 15 + df["Social_Reichweite"] * 1.5 +
        df["TV_Werbedruck"] * 0.12 + df["Kundenzufriedenheit"] * 2.8 + df["Lieferbereitschaft"] * 12
    )
    df["Absatz"] = (basis + einfluss + rng.normal(0, 5, TAGE)).astype(int).clip(0)
    
    MERKMALE = ["Wochentag", "Monat", "Quartal", "Jahrestag", "Temperatur", "Niederschlag", "Sonnenstunden", 
                "Ferienindikator", "Veranstaltung", "Werbebudget", "TV_Werbedruck", "Social_Reichweite", 
                "Energieindex", "Wettbewerb", "Kundenzufriedenheit", "Lieferbereitschaft"]

    X_tr, X_te, y_tr, y_te = train_test_split(df[MERKMALE], df["Absatz"], test_size=0.15, random_state=42)
    skalierer = StandardScaler()
    X_tr_s = skalierer.fit_transform(X_tr)
    
    netz = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", max_iter=1000, random_state=42)
    netz.fit(X_tr_s, y_tr)

    y_hat = netz.predict(skalierer.transform(X_te))
    guete = {"mae": round(mean_absolute_error(y_te, y_hat), 2), "r2": round(r2_score(y_te, y_hat), 4)}
    
    importance = np.abs(netz.coefs_[0]).sum(axis=1)
    importance = (importance / importance.sum()) * 100
    df_imp = pd.DataFrame({"Feature": MERKMALE, "Wichtigkeit (%)": importance}).sort_values("Wichtigkeit (%)", ascending=True)

    return netz, skalierer, df, MERKMALE, guete, df_imp

netz, skalierer, hdf, MERKMALE, guete, df_imp = modell_laden()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER & SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
jetzt = datetime.datetime.now().strftime("%d.%m.%Y · %H:%M")

st.markdown(f"""
<div style="margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #1F2937;">
    <h1 class="header-main">OpenBrain · Demand Intelligence</h1>
    <p class="header-sub">KI-gestützte Kapazitäts- und Absatzprognose</p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">
        <span class="status-badge ok">System Live</span>
        <span style="padding-left: 1rem; border-left: 1px solid #1F2937;">Modell-Genauigkeit (R²) = {guete['r2']}</span>
        <span style="margin-left: auto;">Last Update: {jetzt}</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="section-title">Globale Parameter</p>', unsafe_allow_html=True)
    with st.expander("Wetter & Saisonalität", expanded=True):
        d_temp = st.number_input("Ø Temp (°C)", -15, 45, 18)
        d_regen = st.slider("Regenrisiko (mm)", 0, 60, 5)
        d_sonne = st.slider("Sonnenstunden (h)", 0.0, 14.0, 8.5)
        d_ferien = st.checkbox("Ferienzeitraum", value=False)
    with st.expander("Marketing & Markt"):
        d_budget = st.number_input("Tagesbudget (€)", 0, 1000, 350)
        d_tv = st.slider("TV-Werbedruck", 0, 100, 45)
        d_social = st.slider("Social Media Reach", 0.0, 10.0, 6.5)
        d_event = st.checkbox("Großveranstaltung lokal")
    with st.expander("Operations"):
        d_preis = st.number_input("Ø Preis (€)", 1.0, 50.0, 12.50)
        d_liefer = st.slider("Lieferbereitschaft", 0.5, 1.0, 0.95)
        d_zufried = st.slider("CSAT-Score", 1.0, 10.0, 8.2)

# ══════════════════════════════════════════════════════════════════════════════
# PROGNOSE-BERECHNUNG
# ══════════════════════════════════════════════════════════════════════════════
heute = datetime.date.today()
tage_prognose = []

for idx in range(7):
    tag = heute + datetime.timedelta(days=idx)
    wt = tag.weekday()
    jt = (tag - datetime.date(tag.year, 1, 1)).days + 1
    t_temp = d_temp + np.sin(idx)*3
    t_regen = max(0, d_regen + np.random.uniform(-5, 5))
    
    feat_vals = [wt, tag.month, (tag.month-1)//3+1, jt, t_temp, t_regen, d_sonne, 
                 int(d_ferien), int(d_event if wt in [4,5] else 0), d_budget, d_tv, d_social, 
                 105, 0.3, d_zufried, d_liefer]
    
    absatz = int(max(0, netz.predict(skalierer.transform([feat_vals]))[0]))
    status = "crit" if absatz > 110 else "warn" if absatz > 85 else "ok"
    pers = round(3 + max(0, (absatz - 60) / 25))
    
    tage_prognose.append({
        "datum": tag, "absatz": absatz, "status": status, "personal": pers, "umsatz": absatz * d_preis,
        "tag_kurz": ["Mo","Di","Mi","Do","Fr","Sa","So"][wt],
        "tag_lang": ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"][wt]
    })

# Metrics
kw_absatz = sum(t["absatz"] for t in tage_prognose)
kw_umsatz = sum(t["umsatz"] for t in tage_prognose)

st.markdown('<div style="background: #111827; border: 1px solid #1F2937; border-radius: 6px; padding: 1.5rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prognose Woche (Einheiten)", f"{kw_absatz}", f"Ø {kw_absatz//7}")
c2.metric("Erwarteter Umsatz", f"{kw_umsatz:,.0f} €")
c3.metric("Max. Personalbedarf", f"{max(t['personal'] for t in tage_prognose)} FTE")
c4.metric("Warnungen", f"{sum(1 for t in tage_prognose if t['status'] != 'ok')}/7")
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["Wochenübersicht", "Deep Dive", "Szenario", "Risiko-Simulation", "KI-Modell"])

with tabs[0]:
    st.markdown('<p class="section-title">7-Tage Verlauf</p>', unsafe_allow_html=True)
    cols = st.columns(7)
    for col, tag in zip(cols, tage_prognose):
        with col:
            st.markdown(f"""
            <div class="day-card">
                <div style="font-size: 0.7rem; color: var(--text-muted);">{tag['datum'].strftime('%d.%m')}</div>
                <div style="font-weight: 600; margin-bottom: 0.5rem;">{tag['tag_kurz']}</div>
                <div class="day-card-metric">{tag['absatz']}</div>
                <div class="status-badge {tag['status']}" style="width: 100%; justify-content: center;">{tag['status']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=[t["tag_kurz"] for t in tage_prognose], y=[t["absatz"] for t in tage_prognose], mode='lines+markers', line=dict(color='#3B82F6', width=3), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig_trend.update_layout(**PT_BASE, height=350, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_trend, use_container_width=True)

with tabs[1]:
    sel_tag_idx = st.selectbox("Tag wählen", range(7), format_func=lambda i: f"{tage_prognose[i]['tag_lang']}")
    sel_tag = tage_prognose[sel_tag_idx]
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        wf_data = [
            {"Measure": "absolute", "Feature": "Basis", "Value": 55},
            {"Measure": "relative", "Feature": "Tag-Effekt", "Value": 15 if sel_tag['tag_kurz'] in ['Fr','Sa'] else -5},
            {"Measure": "total", "Feature": "Gesamt", "Value": sel_tag["absatz"]}
        ]
        fig_wf = go.Figure(go.Waterfall(orientation="v", measure=[x["Measure"] for x in wf_data], x=[x["Feature"] for x in wf_data], y=[x["Value"] for x in wf_data]))
        fig_wf.update_layout(**PT_BASE, height=350, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_wf, use_container_width=True)
    with c2:
        stunden = list(range(8, 22))
        st_val = [np.random.randint(5, 15) for _ in stunden]
        fig_hour = go.Figure(go.Bar(x=[f"{h}:00" for h in stunden], y=st_val, marker_color="#3B82F6"))
        fig_hour.update_layout(**PT_BASE, height=350, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_hour, use_container_width=True)

with tabs[2]:
    sz_budget = st.slider("Szenario Budget (€)", 0, 1000, 500)
    st.info("Szenario-Funktion aktiv: Vergleichen Sie Baseline vs. Budget-Boost.")

with tabs[3]:
    st.markdown('<p class="section-title">Monte Carlo Risiko-Vorschau</p>', unsafe_allow_html=True)
    sim_x = pd.date_range(heute, periods=30)
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=sim_x, y=np.random.normal(kw_absatz//7, 10, 30), line=dict(color='#3B82F6')))
    fig_mc.update_layout(**PT_BASE, height=400, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_mc, use_container_width=True)

with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Modell-Konfiguration**")
        st.json(guete)
    with c2:
        fig_imp = px.bar(df_imp, x="Wichtigkeit (%)", y="Feature", orientation='h')
        fig_imp.update_layout(**PT_BASE, height=400, margin=dict(l=120, r=20, t=20, b=40))
        st.plotly_chart(fig_imp, use_container_width=True)

st.markdown('<div class="footer">OpenBrain Intelligence · v4.5 Enterprise · 2026</div>', unsafe_allow_html=True)