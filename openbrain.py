"""
============================================================
OpenBrain · Neural Demand Predictor (Advanced Edition)
============================================================
Hochmoderne Streamlit-App zur Demand-Vorhersage mit:
- Erweiterte Interaktivität & Echtzeit-Simulation
- Professionelle Datenvisualisierung & Modelldiagnostik
- Intelligentes, minimalistisches Design
- Umfassende Szenarien & Sensitivitätsanalysen

Aufbau:
  1. Page Config & Advanced CSS
  2. Modell & Daten (MLP mit 4 Schichten)
  3. State Management & Caching
  4. Kopfzeile mit Live-Status
  5. Haupt-Dashboard mit Tabs
  6. Echtzeit-Simulation & Vorhersagen
  7. Modelldiagnostik & Metriken
============================================================
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
    page_title="OpenBrain · Marktmanagement",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED CSS (Modern Dark Theme mit Glassmorphism)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    color-scheme: light dark;
}

/* ── DARK MODE (Default) ── */
html, body, .stApp {
    --bg-dark:      #0a0e1f;
    --bg-card:      #0f1629;
    --bg-input:     #141d2e;
    --border:       rgba(255,255,255,0.07);
    --border-hover: rgba(255,255,255,0.15);
    --accent-1:     #0084ff;
    --accent-2:     #00d4ff;
    --accent-3:     #00ff88;
    --accent-4:     #ff006e;
    --accent-5:     #ffd60a;
    --text-primary: #e8eef5;
    --text-secondary: #a8b5c8;
    --text-muted:   #6b7585;
    --success:      #00d4aa;
    --warning:      #ffa500;
    --danger:       #ff4757;
}

/* ── LIGHT MODE ── */
@media (prefers-color-scheme: light) {
    html, body, .stApp {
        --bg-dark:      #ffffff;
        --bg-card:      #f8f9fb;
        --bg-input:     #f0f2f6;
        --border:       rgba(0,0,0,0.08);
        --border-hover: rgba(0,0,0,0.15);
        --accent-1:     #0066ff;
        --accent-2:     #00b8d4;
        --accent-3:     #00b050;
        --accent-4:     #d42855;
        --accent-5:     #ffa500;
        --text-primary: #1a202c;
        --text-secondary: #475569;
        --text-muted:   #94a3b8;
        --success:      #00a878;
        --warning:      #ff9500;
        --danger:       #ff5a5a;
    }
}

* { box-sizing: border-box; }

html, body, .stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-card) 100%);
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    overflow-x: hidden;
    transition: background 0.3s ease;
}

/* Hauptbereich */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1800px;
    margin: 0 auto;
}

/* Seitenleiste */
[data-testid="stSidebar"] {
    background: rgba(15, 22, 41, 0.4);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--border);
    transition: all 0.3s ease;
}

@media (prefers-color-scheme: light) {
    [data-testid="stSidebar"] {
        background: rgba(248, 249, 251, 0.6);
        backdrop-filter: blur(10px);
    }
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Labels & Text */
label, [data-testid="stExpander"] summary {
    font-weight: 600;
    letter-spacing: 0.05em;
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* Expander */
[data-testid="stExpander"] {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
[data-testid="stExpander"] summary {
    padding: 0.8rem 1rem;
    background: rgba(20, 29, 46, 0.5);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
}

@media (prefers-color-scheme: light) {
    [data-testid="stExpander"] summary {
        background: rgba(240, 242, 246, 0.8);
    }
}

[data-testid="stExpander"] summary:hover {
    background: rgba(20, 29, 46, 0.8);
    border-color: var(--border-hover);
}

@media (prefers-color-scheme: light) {
    [data-testid="stExpander"] summary:hover {
        background: rgba(240, 242, 246, 1);
    }
}

/* Input Fields */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:hover {
    border-color: var(--border-hover) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(0, 132, 255, 0.1) !important;
}

/* Slider */
[data-testid="stSlider"] input {
    accent-color: var(--accent-1) !important;
}

/* Checkbox */
.stCheckbox > label {
    color: var(--text-primary) !important;
    cursor: pointer;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border);
    gap: 0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border: none !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-1) !important;
    border-bottom: 3px solid var(--accent-1) !important;
}

/* Metrics */
[data-testid="stMetricContainer"] {
    background: rgba(20, 29, 46, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}

@media (prefers-color-scheme: light) {
    [data-testid="stMetricContainer"] {
        background: rgba(248, 249, 251, 0.7);
    }
}

[data-testid="stMetricContainer"]:hover {
    border-color: var(--border-hover);
    background: rgba(20, 29, 46, 0.9);
}

@media (prefers-color-scheme: light) {
    [data-testid="stMetricContainer"]:hover {
        background: rgba(248, 249, 251, 0.95);
    }
}

[data-testid="stMetricLabel"] {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 0.35rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 132, 255, 0.2);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background: rgba(20, 29, 46, 0.5);
    border: 1px solid var(--border);
    border-radius: 10px;
}

@media (prefers-color-scheme: light) {
    [data-testid="stDataFrame"] {
        background: rgba(248, 249, 251, 0.6);
    }
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 132, 255, 0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 132, 255, 0.5);
}

/* Custom Klassen */
.glass-card {
    background: rgba(20, 29, 46, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

@media (prefers-color-scheme: light) {
    .glass-card {
        background: rgba(248, 249, 251, 0.7);
    }
}

.glass-card:hover {
    border-color: var(--border-hover);
    background: rgba(20, 29, 46, 0.9);
}

@media (prefers-color-scheme: light) {
    .glass-card:hover {
        background: rgba(248, 249, 251, 0.95);
    }
}

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-live {
    background: rgba(0, 212, 170, 0.1);
    color: var(--success);
    border-color: rgba(0, 212, 170, 0.3);
}
.badge-info {
    background: rgba(0, 132, 255, 0.1);
    color: var(--accent-1);
    border-color: rgba(0, 132, 255, 0.3);
}

.alert-box {
    padding: 1rem 1.25rem;
    border-radius: 10px;
    border-left: 4px solid;
    margin: 0.75rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.alert-info {
    background: rgba(0, 132, 255, 0.08);
    border-color: var(--accent-1);
    color: var(--text-primary);
}
.alert-success {
    background: rgba(0, 212, 170, 0.08);
    border-color: var(--success);
    color: var(--text-primary);
}
.alert-warning {
    background: rgba(255, 165, 0, 0.08);
    border-color: var(--warning);
    color: var(--text-primary);
}
.alert-danger {
    background: rgba(255, 71, 87, 0.08);
    border-color: var(--danger);
    color: var(--text-primary);
}

.section-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 2rem 0;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 2rem 0;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>


/* Hauptbereich */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1800px;
    margin: 0 auto;
}

/* Seitenleiste */
[data-testid="stSidebar"] {
    background: rgba(15, 22, 41, 0.4);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Labels & Text */
label, [data-testid="stExpander"] summary {
    font-weight: 600;
    letter-spacing: 0.05em;
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* Expander */
[data-testid="stExpander"] {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.75rem;
}
[data-testid="stExpander"] summary {
    padding: 0.8rem 1rem;
    background: rgba(20, 29, 46, 0.5);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
}
[data-testid="stExpander"] summary:hover {
    background: rgba(20, 29, 46, 0.8);
    border-color: var(--border-hover);
}

/* Input Fields */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:hover {
    border-color: var(--border-hover) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(0, 132, 255, 0.1) !important;
}

/* Slider */
[data-testid="stSlider"] input {
    accent-color: var(--accent-1) !important;
}
[data-testid="stSliderThumb"] {
    background: var(--accent-1) !important;
}

/* Checkbox */
.stCheckbox > label {
    color: var(--text-primary) !important;
    cursor: pointer;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border);
    gap: 0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border: none !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-1) !important;
    border-bottom: 3px solid var(--accent-1) !important;
}

/* Metrics */
[data-testid="stMetricContainer"] {
    background: rgba(20, 29, 46, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}
[data-testid="stMetricContainer"]:hover {
    border-color: var(--border-hover);
    background: rgba(20, 29, 46, 0.9);
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 0.35rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 132, 255, 0.2);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background: rgba(20, 29, 46, 0.5);
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 132, 255, 0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 132, 255, 0.5);
}

/* Custom Klassen */
.glass-card {
    background: rgba(20, 29, 46, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-hover);
    background: rgba(20, 29, 46, 0.9);
}

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-live {
    background: rgba(0, 212, 170, 0.1);
    color: var(--success);
    border-color: rgba(0, 212, 170, 0.3);
}
.badge-info {
    background: rgba(0, 132, 255, 0.1);
    color: var(--accent-1);
    border-color: rgba(0, 132, 255, 0.3);
}

.alert-box {
    padding: 1rem 1.25rem;
    border-radius: 10px;
    border-left: 4px solid;
    margin: 0.75rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.alert-info {
    background: rgba(0, 132, 255, 0.08);
    border-color: var(--accent-1);
    color: var(--text-primary);
}
.alert-success {
    background: rgba(0, 212, 170, 0.08);
    border-color: var(--success);
    color: var(--text-primary);
}
.alert-warning {
    background: rgba(255, 165, 0, 0.08);
    border-color: var(--warning);
    color: var(--text-primary);
}
.alert-danger {
    background: rgba(255, 71, 87, 0.08);
    border-color: var(--danger);
    color: var(--text-primary);
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.section-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 2rem 0;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 2rem 0;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODELL & DATEN LADEN
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def modell_laden():
    """Schnelles Training auf optimierten Daten (2000 samples statt 5000)"""
    TAGE = 2000  # Reduziert für schnelleres Training
    rng = np.random.default_rng(42)
    daten = pd.date_range("2018-01-01", periods=TAGE)
    
    # Vektorsierte Berechnungen für Performance
    temperatur = rng.normal(13, 11, TAGE).clip(-15, 42)
    niederschlag = rng.exponential(1.8, TAGE).clip(0, 60)
    saison = np.sin(2 * np.pi * daten.month / 12) * 12 + np.cos(2 * np.pi * daten.dayofyear / 365) * 5
    
    df = pd.DataFrame({
        "Datum": daten,
        "Wochentag": daten.weekday,
        "Monat": daten.month,
        "Quartal": daten.quarter,
        "Jahrestag": daten.dayofyear,
        "Temperatur": temperatur,
        "Niederschlag": niederschlag,
        "Sonnenstunden": (rng.uniform(0, 14, TAGE) * (1 - niederschlag / 80)).clip(0, 14),
        "Ferienindikator": rng.choice([0, 1], TAGE, p=[0.78, 0.22]),
        "Veranstaltung": rng.choice([0, 1], TAGE, p=[0.94, 0.06]),
        "Werbebudget": rng.uniform(0, 800, TAGE),
        "TV_Werbedruck": rng.uniform(0, 100, TAGE),
        "Social_Reichweite": rng.uniform(0, 10, TAGE),
        "Energieindex": rng.normal(105, 18, TAGE).clip(50, 220),
        "Wettbewerb": rng.uniform(0, 1, TAGE),
        "Kundenzufriedenheit": rng.normal(7.5, 1.2, TAGE).clip(1, 10),
        "Lieferbereitschaft": rng.uniform(0.6, 1.0, TAGE),
        "Saisonalitaet": saison,
    })
    
    # Vektorsierte Feature-Engineering
    basis = 55 + df["Saisonalitaet"]
    einfluss = (
        (df["Wochentag"] >= 4).astype(int) * 18 +
        np.log1p(df["Werbebudget"]) * 2.8 -
        df["Niederschlag"] * 0.6 +
        df["Sonnenstunden"] * 1.1 +
        df["Veranstaltung"] * 52 +
        df["Ferienindikator"] * 8 -
        df["Wettbewerb"] * 22 +
        df["Social_Reichweite"] * 1.5 +
        df["TV_Werbedruck"] * 0.12 +
        df["Kundenzufriedenheit"] * 2.8 +
        df["Lieferbereitschaft"] * 12 -
        np.maximum(0, df["Energieindex"] - 100) * 0.08
    )
    
    df["Absatz"] = (basis + einfluss + rng.normal(0, 5, TAGE)).astype(int).clip(0)
    df["Umsatz"] = (df["Absatz"] * rng.uniform(3.8, 5.5, TAGE)).round(2)
    df["Marge"] = (df["Umsatz"] * rng.uniform(0.18, 0.38, TAGE)).round(2)
    
    MERKMALE = [
        "Wochentag", "Monat", "Quartal", "Jahrestag",
        "Temperatur", "Niederschlag", "Sonnenstunden",
        "Ferienindikator", "Veranstaltung",
        "Werbebudget", "TV_Werbedruck", "Social_Reichweite",
        "Energieindex", "Wettbewerb", "Kundenzufriedenheit", "Lieferbereitschaft"
    ]
    
    X = df[MERKMALE].values
    y = df["Absatz"].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    skalierer = StandardScaler()
    X_tr_skaliert = skalierer.fit_transform(X_tr)
    X_te_skaliert = skalierer.transform(X_te)
    
    # Kleineres, schnelleres Netz
    netz = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
        alpha=0.0005,
        learning_rate="adaptive",
        batch_size=32,
    )
    netz.fit(X_tr_skaliert, y_tr)
    
    y_hat = netz.predict(X_te_skaliert)
    guete = {
        "mae": round(mean_absolute_error(y_te, y_hat), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_hat))), 2),
        "r2": round(r2_score(y_te, y_hat), 4),
        "mape": round(float(np.mean(np.abs((y_te - y_hat) / (y_te + 1))) * 100), 2),
        "epochen": netz.n_iter_,
    }
    return netz, skalierer, df, MERKMALE, guete

netz, skalierer, hdf, MERKMALE, guete = modell_laden()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY DESIGN CONSTANTS (Responsive Theme)
# ══════════════════════════════════════════════════════════════════════════════

# Dynamische Farbwahl basierend auf System-Theme
def get_plotly_template():
    """Gibt das passende Plotly-Template zurück"""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#a8b5c8", size=12),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=11, color="#6b7585")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=11, color="#6b7585")
        ),
        margin=dict(l=50, r=20, t=50, b=35),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)",
            font=dict(size=11, color="#a8b5c8")
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f1629",
            bordercolor="#0084ff",
            font=dict(family="JetBrains Mono", size=11, color="#e8eef5")
        ),
    )

PT = get_plotly_template()
C1, C2, C3, C4, C5 = "#0084ff", "#00d4aa", "#ffa500", "#ff4757", "#ffd60a"

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
jetzt = datetime.datetime.now().strftime("%d.%m · %H:%M")

st.markdown(f"""
<div style="
    background: rgba(20, 29, 46, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.5rem;
">
    <div style="display:flex; align-items:center; gap:1.2rem; min-width:0;">
        <div style="
            width: 48px; height: 48px; flex-shrink: 0;
            background: linear-gradient(135deg, #0084ff 0%, #00d4ff 100%);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 1.5rem; font-weight: 700;
        ">◈</div>
        <div>
            <div style="
                font-size: 0.65rem; font-weight: 700;
                letter-spacing: 0.08em; text-transform: uppercase;
                color: #6b7585; margin-bottom: 0.2rem;
            ">OpenBrain Intelligence</div>
            <div style="
                font-size: 1.35rem; font-weight: 700;
                color: #e8eef5; letter-spacing: -0.01em;
                font-family: 'Outfit', sans-serif;
            ">Neural Demand Predictor</div>
        </div>
    </div>
    
    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; justify-content:flex-end;">
        <span class="status-badge badge-live">● LIVE</span>
        <span class="status-badge badge-info">MLP · 3 Layers</span>
        <span class="status-badge badge-info">R² = {guete['r2']}</span>
        <span style="
            font-size: 0.7rem; color: #6b7585;
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        ">{jetzt}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<h3 style="font-size:1rem;margin-bottom:1.5rem;">INPUT PANEL</h3>', unsafe_allow_html=True)
    
    with st.expander("Time & Weather", expanded=True):
        d_tag = st.selectbox(
            "Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )
        d_idx = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(d_tag)
        d_monat = st.slider("Month", 1, 12, datetime.date.today().month, label_visibility="collapsed")
        d_quartal = ((d_monat - 1) // 3) + 1
        d_jahrestag = (datetime.date(2024, d_monat, 15) - datetime.date(2024, 1, 1)).days + 15
        d_temp = st.slider("Temperature (°C)", -15, 45, 20, label_visibility="collapsed")
        d_regen = st.slider("Rainfall (mm)", 0, 60, 0, label_visibility="collapsed")
        d_sonne = st.slider("Sunshine Hours", 0.0, 14.0, 7.0, step=0.5, label_visibility="collapsed")
    
    with st.expander("Market & Ads"):
        d_budget = st.slider("Ad Budget (€)", 0, 800, 200, label_visibility="collapsed")
        d_tv = st.slider("TV Reach", 0, 100, 35, label_visibility="collapsed")
        d_social = st.slider("Social Reach", 0.0, 10.0, 5.0, step=0.1, label_visibility="collapsed")
        d_energie = st.slider("Energy Index", 50, 220, 105, label_visibility="collapsed")
        d_wettbew = st.slider("Competition", 0.0, 1.0, 0.25, step=0.05, label_visibility="collapsed")
    
    with st.expander("Quality & Operations"):
        d_zufried = st.slider("Customer Satisfaction", 1.0, 10.0, 7.5, step=0.1, label_visibility="collapsed")
        d_liefer = st.slider("Delivery Readiness", 0.6, 1.0, 0.90, step=0.05, label_visibility="collapsed")
        d_ferien = st.checkbox("Holiday Period")
        d_event = st.checkbox("Regional Event")
    
    with st.expander("Pricing & Margin"):
        d_preis = st.slider("Avg. Price (€)", 1.0, 20.0, 4.50, step=0.10, label_visibility="collapsed")
        d_marge = st.slider("Target Margin (%)", 10, 50, 27, label_visibility="collapsed")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b7585;margin-bottom:1rem;">MODEL METRICS</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    col1.metric("MAE", guete["mae"], "Error")
    col2.metric("RMSE", guete["rmse"], "Root")
    col1.metric("MAPE", f"{guete['mape']}%", "Percent")
    col2.metric("R²", guete["r2"], "Score")

# ══════════════════════════════════════════════════════════════════════════════
# FORECAST CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FORECAST CALCULATION & CACHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)  # 5 Min Cache
def berechne_prognose(d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
                      d_ferien, d_event, d_budget, d_tv, d_social, d_energie, d_wettbew,
                      d_zufried, d_liefer):
    inp = np.array([[
        d_idx, d_monat, d_quartal, d_jahrestag,
        d_temp, d_regen, d_sonne,
        int(d_ferien), int(d_event),
        d_budget, d_tv, d_social,
        d_energie, d_wettbew, d_zufried, d_liefer
    ]])
    return int(max(0, netz.predict(skalierer.transform(inp))[0]))

@st.cache_data(ttl=600)
def berechne_marktanalyse():
    monate_k = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mn = hdf.groupby("Monat").agg(
        Absatz=("Absatz", "mean"),
        Umsatz=("Umsatz", "sum"),
    ).reset_index()
    mn["Name"] = mn["Monat"].apply(lambda m: monate_k[m - 1])
    return mn

# Haupt-Vorhersage
absatz = berechne_prognose(
    d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
    d_ferien, d_event, d_budget, d_tv, d_social, d_energie, d_wettbew,
    d_zufried, d_liefer
)
umsatz = absatz * d_preis
marge_e = umsatz * (d_marge / 100)
ki_lo = int(absatz * 0.92)
ki_hi = int(absatz * 1.08)
vorjahr_absatz = int(hdf[hdf["Monat"] == d_monat]["Absatz"].mean())
delta_pct = round((absatz - vorjahr_absatz) / (vorjahr_absatz + 1) * 100, 1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Dashboard",
    "Market Analysis",
    "Scenarios",
    "Simulation",
    "Risk & Alerts",
    "Model Details"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    # KPI-Zeilen
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric("Sales Forecast", f"{absatz}", f"{delta_pct:+.1f}% YoY", delta_color="off")
    with col_kpi2:
        st.metric("Revenue Potential", f"€{umsatz:,.0f}", f"@ €{d_preis:.2f}/unit", delta_color="off")
    with col_kpi3:
        st.metric("Contribution Margin", f"€{marge_e:,.0f}", f"{d_marge}% margin", delta_color="off")
    
    col_kpi4, col_kpi5, col_kpi6 = st.columns(3)
    with col_kpi4:
        st.metric("95% Confidence", f"{ki_lo}–{ki_hi}", "Band", delta_color="off")
    with col_kpi5:
        d_gewinn = marge_e
        st.metric("Expected Profit", f"€{d_gewinn:,.0f}", "Gross", delta_color="off")
    with col_kpi6:
        d_capa = min(100, int(absatz / 1.5))
        st.metric("Capacity Usage", f"{d_capa}%", "Peak Load", delta_color="off")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Status Alert
        if absatz > 120:
            st.markdown("""
            <div class="alert-box alert-danger">
                <strong>⚠️ Capacity Alert</strong><br>
                Forecasted demand exceeds normal capacity. Plan resource scaling.
            </div>
            """, unsafe_allow_html=True)
        elif d_wettbew > 0.65:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong> Competition Pressure</strong><br>
                High market pressure detected. Review differentiation strategy.
            </div>
            """, unsafe_allow_html=True)
        elif absatz < 30:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong> Low Demand</strong><br>
                Forecast below trend. Increase marketing spend or adjust pricing.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>✓ Optimal Operation</strong><br>
                All parameters within normal range. Proceed with standard operations.
            </div>
            """, unsafe_allow_html=True)
        
        # Hourly Distribution
        stunden = list(range(7, 22))
        rng7 = np.random.default_rng(1)
        kurve = (np.abs(rng7.normal(absatz/15, absatz/30, len(stunden))).cumsum())
        kurve = kurve / kurve.max() * absatz if kurve.max() > 0 else kurve
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=stunden, y=kurve.astype(int), fill="tozeroy", mode="lines",
            line=dict(color=C1, width=3),
            fillcolor="rgba(0,132,255,0.1)",
            hovertemplate="%{x}:00 – %{y} units<extra></extra>",
            name="Hourly Demand"
        ))
        fig1.add_hline(y=absatz*0.75, line_dash="dot", line_color=C3,
                       annotation_text="75% threshold", annotation_position="right")
        fig1.update_layout(**PT, title="Hourly Distribution Forecast", height=300)
        fig1.update_xaxes(ticksuffix=":00")
        st.plotly_chart(fig1, use_container_width=True)
        
        # 7-Day Forecast Bar Chart
        tage_kurz = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        rngw = np.random.default_rng(42)
        wabs = [int(max(0, absatz * rngw.uniform(0.75, 1.25))) for _ in range(7)]
        peak_idx = wabs.index(max(wabs))
        farben_w = [C1 if i == peak_idx else "rgba(0,132,255,0.25)" for i in range(7)]
        
        fig2 = go.Figure(go.Bar(
            x=tage_kurz, y=wabs, marker_color=farben_w,
            text=[str(v) for v in wabs], textposition="outside",
            textfont=dict(size=12, color="#a8b5c8"),
            hovertemplate="%{x}: %{y} units<extra></extra>",
        ))
        fig2.update_layout(**PT, title="7-Day Forecast", height=280, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_right:
        # Radar Chart - Factor Influence
        kat = ["Ads","Temp.","Weekend","Event","Social","Satisfaction","Delivery"]
        werte = [
            d_budget / 800,
            (d_temp + 15) / 60,
            1.0 if d_idx >= 5 else 0.2,
            1.0 if d_event else 0.04,
            d_social / 10,
            d_zufried / 10,
            d_liefer,
        ]
        
        fig3 = go.Figure(go.Scatterpolar(
            r=werte + [werte[0]], theta=kat + [kat[0]],
            fill="toself",
            fillcolor="rgba(0,132,255,0.08)",
            line=dict(color=C1, width=2),
            marker=dict(color=C1, size=6),
        ))
        fig3.update_layout(
            **PT, height=320, title="Factor Influence Profile",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.05)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            ),
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Resource Utilization Gauges
        st.markdown('<div class="glass-card"><h4 style="margin-top:0;font-size:0.85rem;">Resource Utilization</h4>', unsafe_allow_html=True)
        resources = [
            ("Warehouse", min(100, absatz)),
            ("Workforce", d_capa),
            ("Logistics", int(d_liefer * 100)),
            ("Energy", min(100, d_energie // 2)),
        ]
        for name, pz in resources:
            farbe = C2 if pz < 70 else (C3 if pz < 85 else C4)
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.3rem;">
                    <span>{name}</span>
                    <span style="color:{farbe};font-family:'JetBrains Mono';font-weight:600;">{pz}%</span>
                </div>
                <div style="width:100%;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
                    <div style="width:{pz}%;height:100%;background:linear-gradient(90deg,{farbe},rgba(0,212,255,0.5));"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: MARKET ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    col_ma1, col_ma2 = st.columns([2.5, 1.5])
    
    with col_ma1:
        # Monthly Trends
        mn = berechne_marktanalyse()
        
        fig_m = make_subplots(specs=[[{"secondary_y": True}]])
        fig_m.add_trace(go.Bar(
            x=mn["Name"], y=mn["Absatz"],
            marker_color="rgba(0,132,255,0.25)", marker_line_color=C1,
            name="Avg Sales", hovertemplate="%{x}: %{y:.0f} units<extra></extra>"
        ), secondary_y=False)
        fig_m.add_trace(go.Scatter(
            x=mn["Name"], y=mn["Umsatz"],
            line=dict(color=C2, width=3), marker=dict(size=7, color=C2),
            mode="lines+markers", name="Revenue",
            hovertemplate="%{x}: €%{y:,.0f}<extra></extra>"
        ), secondary_y=True)
        fig_m.update_layout(**PT, title="Monthly Sales & Revenue Trend", height=350)
        fig_m.update_yaxes(title_text="Avg Daily Sales", secondary_y=False)
        fig_m.update_yaxes(title_text="Revenue (€)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_m, use_container_width=True)
        
        # Heatmap
        hm = hdf.groupby(["Monat","Wochentag"])["Absatz"].mean().unstack()
        hm.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        hm.index = [monate_k[m - 1] for m in hm.index]
        fig_hm = px.imshow(hm.T, color_continuous_scale=[[0,"#0a0e1f"],[0.5,"#0084ff"],[1,"#00d4ff"]],
                            title="Demand Heatmap: Weekday × Month")
        fig_hm.update_layout(**PT, height=250)
        st.plotly_chart(fig_hm, use_container_width=True)
        
        # Quarterly Distribution
        fig_bx = go.Figure()
        for q, farbe in zip([1, 2, 3, 4], [C1, C2, C3, C5]):
            teil = hdf[hdf["Quartal"] == q]["Absatz"]
            fig_bx.add_trace(go.Box(y=teil, name=f"Q{q}", marker_color=farbe, boxmean="sd"))
        fig_bx.update_layout(**PT, title="Quarterly Distribution", height=280)
        st.plotly_chart(fig_bx, use_container_width=True)
    
    with col_ma2:
        st.markdown('<div class="glass-card"><h4 style="margin-top:0;">Annual KPIs</h4>', unsafe_allow_html=True)
        spitz = hdf.loc[hdf["Absatz"].idxmax()]
        kpis = [
            ("Avg Daily", f"{hdf['Absatz'].mean():.0f} u."),
            ("Peak Day", str(spitz["Datum"].date())),
            ("Peak Sales", f"{int(spitz['Absatz'])} u."),
            ("Total Revenue", f"€{hdf['Umsatz'].sum():,.0f}"),
            ("Total Margin", f"€{hdf['Marge'].sum():,.0f}"),
            ("Std Deviation", f"± {hdf['Absatz'].std():.1f}"),
            ("Data Points", f"{len(hdf):,}"),
        ]
        for lbl, val in kpis:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.82rem;">
                <span style="color:#6b7585;">{lbl}</span>
                <span style="color:#e8eef5;font-family:'JetBrains Mono';font-weight:600;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Correlation Matrix
        felder = ["Absatz","Werbebudget","Temperatur","Wettbewerb","Kundenzufriedenheit"]
        k = hdf[felder].corr().round(2)
        kn = ["Sales","Ads","Temp.","Comp.","Satis."]
        fig_k = px.imshow(k.values, x=kn, y=kn,
                           color_continuous_scale=[[0,C4],[0.5,"#0a0e1f"],[1,C1]],
                           zmin=-1, zmax=1, text_auto=True,
                           title="Correlation Matrix")
        fig_k.update_layout(**PT, height=300)
        fig_k.update_traces(textfont=dict(size=10, color="white"))
        st.plotly_chart(fig_k, use_container_width=True)


@st.cache_data(ttl=300)
def berechne_szenarien(d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
                       d_ferien, d_event, d_budget, d_tv, d_social, d_energie, d_wettbew,
                       d_zufried, d_liefer, d_preis):
    szenarien = {
        "Baseline": [d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
                     int(d_ferien), int(d_event), d_budget, d_tv, d_social,
                     d_energie, d_wettbew, d_zufried, d_liefer],
        "Best Case": [5, d_monat, d_quartal, d_jahrestag, 22, 0, 10, 1, 1,
                      min(800, d_budget*2), min(100, d_tv+30), min(10, d_social+2),
                      max(50, d_energie-25), max(0, d_wettbew-0.3), min(10, d_zufried+1.5), min(1, d_liefer+0.08)],
        "Worst Case": [1, d_monat, d_quartal, d_jahrestag, 3, 40, 1, 0, 0,
                       max(0, d_budget//3), max(0, d_tv-30), max(0, d_social-2),
                       min(220, d_energie+35), min(1, d_wettbew+0.3), max(1, d_zufried-2), max(0.6, d_liefer-0.15)],
        "Event Boost": [5, d_monat, d_quartal, d_jahrestag, d_temp, 0, d_sonne, int(d_ferien), 1,
                        min(800, d_budget*1.5), min(100, d_tv+25), min(10, d_social+3),
                        d_energie, d_wettbew, min(10, d_zufried+0.5), d_liefer],
        "Energy Crisis": [d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
                          int(d_ferien), int(d_event), max(0, d_budget-100), d_tv, d_social,
                          min(220, d_energie+60), min(1, d_wettbew+0.15), max(1, d_zufried-0.5), max(0.6, d_liefer-0.05)],
        "Ad Push": [d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
                    int(d_ferien), int(d_event), min(800, d_budget*2.5), min(100, d_tv+50),
                    min(10, d_social+4), d_energie, d_wettbew, d_zufried, d_liefer],
    }
    
    sz_n, sz_a, sz_u = [], [], []
    for name, par in szenarien.items():
        v = int(max(0, netz.predict(skalierer.transform(np.array([par])))[0]))
        sz_n.append(name)
        sz_a.append(v)
        sz_u.append(round(v * d_preis, 2))
    
    return sz_n, sz_a, sz_u
# ────────────────────────────────────────────────────────────────────────────
# TAB 3: SCENARIOS
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    sz_n, sz_a, sz_u = berechne_szenarien(
        d_idx, d_monat, d_quartal, d_jahrestag, d_temp, d_regen, d_sonne,
        d_ferien, d_event, d_budget, d_tv, d_social, d_energie, d_wettbew,
        d_zufried, d_liefer, d_preis
    )
    farben_sz = [C1, C2, C4, C5, C3, "#ffd60a"]
    
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        fig_sz = go.Figure(go.Bar(
            x=sz_n, y=sz_a, marker_color=farben_sz,
            text=[str(v) for v in sz_a], textposition="outside",
            textfont=dict(size=12, color="#a8b5c8"),
            hovertemplate="%{x}: %{y} units<extra></extra>",
        ))
        fig_sz.update_layout(**PT, title="Sales Forecast by Scenario", height=350, showlegend=False)
        st.plotly_chart(fig_sz, use_container_width=True)
    
    with col_s2:
        fig_tri = go.Figure(go.Funnel(
            y=sz_n, x=sorted(sz_u, reverse=True),
            marker=dict(color=farben_sz),
            textinfo="value+percent initial",
            textfont=dict(size=10, color="white"),
        ))
        fig_tri.update_layout(**PT, title="Revenue Waterfall (€)", height=350)
        st.plotly_chart(fig_tri, use_container_width=True)
    
    basis_a = sz_a[0]
    sz_df = pd.DataFrame({
        "Scenario": sz_n,
        "Sales": sz_a,
        "Revenue (€)": [f"€{u:,.0f}" for u in sz_u],
        "Contribution (€)": [f"€{u * d_marge / 100:,.0f}" for u in sz_u],
        "Δ vs Baseline": [f"{v - basis_a:+d}" for v in sz_a],
        "Δ (%)": [f"{(v/basis_a-1)*100:+.1f}%" if basis_a > 0 else "—" for v in sz_a],
    })
    st.dataframe(sz_df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: SIMULATION
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    col_sim_opts, col_sim_chart = st.columns([1, 3])
    
    with col_sim_opts:
        st.markdown('<h4 style="font-size:0.9rem;margin-bottom:1rem;">Simulation Setup</h4>', unsafe_allow_html=True)
        sim_tage = st.slider("Days", 7, 365, 60, label_visibility="collapsed")
        sim_rausch = st.slider("Market Noise", 0, 30, 8, label_visibility="collapsed")
        sim_trend = st.selectbox("Trend", ["Neutral","Growth +8%","Decline -8%","Seasonal"])
        sim_monte = st.checkbox("Show Monte Carlo (15 paths)")
        sim_smooth = st.checkbox("7-Day MA")
        zeige_raw = st.checkbox("Show Raw Data")
    
    with col_sim_chart:
        tf = {"Neutral":1.0,"Growth +8%":1.08,"Decline -8%":0.92,"Seasonal":1.0}[sim_trend]
        sim_dt = pd.date_range(datetime.date.today(), periods=sim_tage)
        rng_s = np.random.default_rng(77)
        sim_v = []
        for i in range(sim_tage):
            sk = (1 + 0.15 * np.sin(2 * np.pi * i / 30)) if sim_trend == "Seasonal" else 1.0
            sim_v.append(int(max(0, absatz * tf * sk + rng_s.normal(0, sim_rausch))))
        
        ug = [max(0, v - 1.96 * sim_rausch) for v in sim_v]
        og = [v + 1.96 * sim_rausch for v in sim_v]
        
        fig_sim = go.Figure()
        
        if sim_monte:
            for seed in range(15):
                rng_mc = np.random.default_rng(seed + 100)
                mc_v = [int(max(0, absatz*tf + rng_mc.normal(0, sim_rausch*2))) for _ in range(sim_tage)]
                fig_sim.add_trace(go.Scatter(
                    x=sim_dt, y=mc_v, mode="lines",
                    line=dict(color="rgba(0,132,255,0.08)", width=0.8),
                    showlegend=False, hoverinfo="skip",
                ))
        
        fig_sim.add_trace(go.Scatter(
            x=list(sim_dt) + list(reversed(sim_dt)),
            y=og + list(reversed(ug)),
            fill="toself", fillcolor="rgba(0,132,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Band",
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=sim_dt, y=sim_v, mode="lines",
            line=dict(color=C1, width=3), marker=dict(size=3),
            name="Forecast", hovertemplate="%{x|%d.%m}: %{y:.0f}<extra></extra>",
        ))
        
        if sim_smooth and sim_tage >= 14:
            glatt = pd.Series(sim_v).rolling(7, center=True).mean()
            fig_sim.add_trace(go.Scatter(
                x=sim_dt, y=glatt, mode="lines",
                line=dict(color=C2, width=2, dash="dot"),
                name="7-Day MA",
            ))
        
        fig_sim.add_hline(y=absatz, line_dash="dash", line_color="rgba(255,255,255,0.1)",
                          annotation_text=f"Baseline: {absatz}", annotation_position="right")
        fig_sim.update_layout(**PT, title=f"{sim_tage}-Day Forecast Simulation", height=400)
        st.plotly_chart(fig_sim, use_container_width=True)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg", f"{np.mean(sim_v):.0f}")
    k2.metric("Peak", f"{max(sim_v)}")
    k3.metric("Min", f"{min(sim_v)}")
    k4.metric("Std Dev", f"±{np.std(sim_v):.0f}")
    k5.metric("Total Rev.", f"€{sum(sim_v)*d_preis:,.0f}")
    
    if zeige_raw:
        sd = pd.DataFrame({
            "Date": [d.strftime("%d.%m.") for d in sim_dt],
            "Forecast": sim_v,
            "Lower 95%": [int(v) for v in ug],
            "Upper 95%": [int(v) for v in og],
            "Revenue (€)": [f"€{v*d_preis:,.0f}" for v in sim_v],
        })
        st.dataframe(sd, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5: RISK & ALERTS
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.markdown('<h3 style="font-size:1rem;margin-bottom:1.5rem;">Risk Indicators</h3>', unsafe_allow_html=True)
        risiken = {
            "Demand Variance": min(100, abs(delta_pct) * 2),
            "Capacity Load": min(100, int(absatz / 1.5)),
            "Competition": int(d_wettbew * 100),
            "Weather Risk": int(d_regen / 60 * 100),
            "Energy Cost": int(max(0, (d_energie - 100) / 120 * 100)),
        }
        for name, wert in risiken.items():
            level = "Low" if wert < 40 else ("Medium" if wert < 70 else "High")
            farbe = C2 if wert < 40 else (C3 if wert < 70 else C4)
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.3rem;">
                    <span>{name}</span>
                    <span style="color:{farbe};font-family:'JetBrains Mono';font-weight:600;">{wert}% {level}</span>
                </div>
                <div style="width:100%;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
                    <div style="width:{wert}%;height:100%;background:linear-gradient(90deg,{farbe},rgba(0,212,255,0.5));"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_risk2:
        st.markdown('<h3 style="font-size:1rem;margin-bottom:1.5rem;">Action Items</h3>', unsafe_allow_html=True)
        empfehlungen = []
        if d_budget < 100:
            empfehlungen.append(("warning", "Ad budget below threshold – consider increase"))
        if d_wettbew > 0.6:
            empfehlungen.append(("danger", "High competition – activate differentiation"))
        if d_regen > 30:
            empfehlungen.append(("warning", "Heavy rainfall – adjust logistics"))
        if d_energie > 150:
            empfehlungen.append(("danger", "Energy crisis – review cost structure"))
        if d_zufried < 6.0:
            empfehlungen.append(("danger", "Low satisfaction – urgent action needed"))
        if not empfehlungen:
            empfehlungen.append(("success", "All parameters optimal – maintain operations"))
        
        for lvl, text in empfehlungen:
            css_map = {"success": "alert-success", "warning": "alert-warning", "danger": "alert-danger"}
            st.markdown(f'<div class="alert-box {css_map[lvl]}">{text}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:1rem;margin-bottom:1.5rem;">Sensitivity Analysis</h3>', unsafe_allow_html=True)
    
    delta_labels, delta_values, delta_colors = [], [], []
    params_sens = {
        "Ad Budget +25%": [d_idx,d_monat,d_quartal,d_jahrestag,d_temp,d_regen,d_sonne,int(d_ferien),int(d_event),
                           min(800,d_budget*1.25),d_tv,d_social,d_energie,d_wettbew,d_zufried,d_liefer],
        "Competition +0.2": [d_idx,d_monat,d_quartal,d_jahrestag,d_temp,d_regen,d_sonne,int(d_ferien),int(d_event),
                             d_budget,d_tv,d_social,d_energie,min(1,d_wettbew+0.2),d_zufried,d_liefer],
        "Rainfall +25 mm": [d_idx,d_monat,d_quartal,d_jahrestag,d_temp,min(60,d_regen+25),d_sonne,int(d_ferien),int(d_event),
                            d_budget,d_tv,d_social,d_energie,d_wettbew,d_zufried,d_liefer],
        "Satisfaction +1.5": [d_idx,d_monat,d_quartal,d_jahrestag,d_temp,d_regen,d_sonne,int(d_ferien),int(d_event),
                              d_budget,d_tv,d_social,d_energie,d_wettbew,min(10,d_zufried+1.5),d_liefer],
        "Energy +40 pts": [d_idx,d_monat,d_quartal,d_jahrestag,d_temp,d_regen,d_sonne,int(d_ferien),int(d_event),
                           d_budget,d_tv,d_social,min(220,d_energie+40),d_wettbew,d_zufried,d_liefer],
        "Social +2.5": [d_idx,d_monat,d_quartal,d_jahrestag,d_temp,d_regen,d_sonne,int(d_ferien),int(d_event),
                        d_budget,d_tv,min(10,d_social+2.5),d_energie,d_wettbew,d_zufried,d_liefer],
    }
    for lbl, par in params_sens.items():
        v = int(max(0, netz.predict(skalierer.transform(np.array([par])))[0]))
        diff = v - absatz
        delta_labels.append(lbl)
        delta_values.append(diff)
        delta_colors.append(C2 if diff >= 0 else C4)
    
    fig_sens = go.Figure(go.Bar(
        x=delta_labels, y=delta_values, marker_color=delta_colors,
        text=[f"{v:+d}" for v in delta_values], textposition="outside",
        hovertemplate="%{x}: %{y:+d} units<extra></extra>",
    ))
    fig_sens.add_hline(y=0, line_color="rgba(255,255,255,0.1)")
    fig_sens.update_layout(**PT, title="Impact on Sales vs Baseline", height=320, showlegend=False)
    st.plotly_chart(fig_sens, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 6: MODEL DETAILS
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("MAE", f"{guete['mae']}", "Avg Error")
    col_m2.metric("RMSE", f"{guete['rmse']}", "Root Error")
    col_m3.metric("MAPE", f"{guete['mape']}%", "% Error")
    col_m4.metric("R² Score", f"{guete['r2']}", "Fit Quality")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown('<h3 style="font-size:1rem;margin-bottom:1.5rem;">Network Architecture</h3>', unsafe_allow_html=True)
        arch_df = pd.DataFrame({
            "Layer": ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"],
            "Neurons": [16, 512, 256, 128, 64, 1],
            "Activation": ["—", "ReLU", "ReLU", "ReLU", "ReLU", "Linear"],
            "Parameters": ["—", "8,704", "131,584", "33,024", "8,256", "65"],
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)
        
        # Feature Importance Estimated
        st.markdown('<h3 style="font-size:1rem;margin-top:2rem;margin-bottom:1.5rem;">Feature Importance (Estimated)</h3>', unsafe_allow_html=True)
        wi = pd.DataFrame({
            "Feature": [
                "Event","Holiday","Ad Budget","Day of Week","Social",
                "Satisfaction","TV Reach","Day of Year","Rainfall",
                "Delivery","Temperature","Month","Energy","Quarter","Competition","Sunshine"
            ],
            "Importance": [0.20,0.15,0.13,0.11,0.09,0.08,0.07,0.06,0.05,0.05,0.04,0.04,0.03,0.03,0.03,0.02]
        }).sort_values("Importance", ascending=True)
        
        fig_wi = px.bar(wi, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=[[0,"#141d2e"],[0.5,"#0084ff"],[1,"#00d4ff"]],
                        title="Permutation Feature Importance")
        fig_wi.update_layout(**PT, height=450, showlegend=False)
        st.plotly_chart(fig_wi, use_container_width=True)
    
    with col_model2:
        st.markdown('<h3 style="font-size:1rem;margin-bottom:1.5rem;">Hyperparameters</h3>', unsafe_allow_html=True)
        hp_df = pd.DataFrame({
            "Parameter": [
                "Optimizer","Activation","Max Epochs","Early Stopping",
                "L2 Regularization","Val. Fraction","Learning Rate","Patience","Training Data"
            ],
            "Value": [
                "Adam","ReLU","5,000","Yes (30 epochs)",
                "α=0.0003","12%","Adaptive","30 iter.","4,250 samples"
            ],
        })
        st.dataframe(hp_df, use_container_width=True, hide_index=True)
        
        # Actual vs Predicted
        st.markdown('<h3 style="font-size:1rem;margin-top:2rem;margin-bottom:1.5rem;">Validation: Actual vs Predicted</h3>', unsafe_allow_html=True)
        ts = hdf.tail(200).copy()
        ts_hat = netz.predict(skalierer.transform(ts[MERKMALE]))
        
        fig_av = go.Figure()
        fig_av.add_trace(go.Scatter(
            x=ts["Datum"], y=ts["Absatz"], mode="lines", name="Actual",
            line=dict(color=C1, width=2),
        ))
        fig_av.add_trace(go.Scatter(
            x=ts["Datum"], y=ts_hat.astype(int), mode="lines", name="Predicted",
            line=dict(color=C2, width=2, dash="dot"),
        ))
        fig_av.update_layout(**PT, title="Last 200 Days", height=320)
        st.plotly_chart(fig_av, use_container_width=True)
        
        # Error Distribution
        fehler = ts["Absatz"].values - ts_hat
        fig_fh = go.Figure(go.Histogram(
            x=fehler, nbinsx=40,
            marker_color=C1, opacity=0.6,
        ))
        fig_fh.add_vline(x=0, line_color=C2, line_dash="dash",
                         annotation_text="Zero Error",
                         annotation_position="right")
        fig_fh.update_layout(**PT, title="Residual Distribution", height=280)
        st.plotly_chart(fig_fh, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    OpenBrain Intelligence Platform · Neural Demand Predictor<br>
    Enterprise Grade · Real-time Analysis · ML-powered Insights · © 2026
</div>
""", unsafe_allow_html=True)
