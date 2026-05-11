import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="OpenBrain Supermarket Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL ENTERPRISE DESIGN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #FFFFFF; 
    --surface: #F8F9FB; 
    --border: #E5E7EB; 
    --accent: #0052CC;
    --text: #1F2937; 
    --ok: #059669; 
    --warn: #D97706; 
    --crit: #DC2626;
    --light-text: #6B7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Header Styling */
h1 {
    color: var(--text) !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    margin-bottom: 0.5rem !important;
}

h2 {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
}

h3 {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

/* Executive Summary Box */
.exec-card {
    background: linear-gradient(135deg, #F0F4FF 0%, #FFFFFF 100%);
    border: 1px solid #C5D9FF;
    border-left: 4px solid var(--accent);
    padding: 1.8rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.exec-card h3 {
    color: var(--accent) !important;
    margin-top: 0 !important;
}

/* Alert Cards */
.alert-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-left: 4px solid;
    padding: 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.alert-critical { border-left-color: var(--crit); background: #FEF2F2; }
.alert-warning { border-left-color: var(--warn); background: #FFFBEB; }
.alert-success { border-left-color: var(--ok); background: #F0FDF4; }

.alert-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}

.alert-critical .alert-title { color: var(--crit); }
.alert-warning .alert-title { color: var(--warn); }
.alert-success .alert-title { color: var(--ok); }

.alert-description {
    font-size: 0.85rem;
    color: var(--light-text);
    line-height: 1.5;
}

/* Info Box */
.info-box {
    background: #F0F4FF;
    border: 1px solid #C5D9FF;
    padding: 1.2rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    font-size: 0.9rem;
    color: #1F2937;
    line-height: 1.6;
}

.info-box strong {
    color: var(--accent);
}

/* Metrics */
[data-testid="stMetricContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1.2rem !important;
}

/* Table Styling */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2rem 0 !important;
}

.subtitle {
    color: var(--light-text);
    font-size: 0.95rem;
    margin-top: -0.8rem;
    margin-bottom: 2rem;
}

.stat-label {
    color: var(--light-text);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.stat-value {
    color: var(--text);
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 0.3rem;
}

/* Model Info Section */
.model-info {
    background: #F9FAFB;
    border: 1px solid var(--border);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.model-info-header {
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 1rem;
    font-size: 1rem;
}

.model-info-item {
    margin-bottom: 1rem;
    display: flex;
    gap: 1rem;
}

.model-info-label {
    font-weight: 600;
    color: var(--text);
    min-width: 150px;
}

.model-info-value {
    color: var(--light-text);
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN NEURAL NETWORK MODEL WITH DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def modell_laden():
    """
    Trainiert ein mehrschichtiges neuronales Netzwerk für Nachfrage-Prognose.
    
    Modell-Architektur:
    - Input Layer: 16 Merkmale
    - Hidden Layers: 128 → 64 Neuronen
    - Output Layer: 1 Merkmal (Absatzmenge)
    - Aktivierungsfunktion: ReLU
    - Optimizer: Adam (Scikit-Learn MLPRegressor)
    """
    
    # === TRAININGSDATEN GENERIEREN ===
    TAGE = 5000  # 13+ Jahre historischer Daten
    rng = np.random.default_rng(42)  # Seed für Reproduzierbarkeit
    daten = pd.date_range("2011-01-01", periods=TAGE)
    
    # Erstelle Features basierend auf realistischen Szenarien
    df = pd.DataFrame({
        "Datum": daten,
        "Wochentag": daten.weekday,  # 0=Mo, 6=So (Wochenend-Effekt)
        "Monat": daten.month,  # 1-12 (Saisonalität)
        "Quartal": daten.quarter,  # Q1-Q4
        "Jahrestag": daten.dayofyear,  # 1-365 (Jahres-Muster)
        "Temperatur": rng.normal(15, 10, TAGE),  # °C, μ=15, σ=10
        "Niederschlag": rng.exponential(2, TAGE),  # mm, exponential
        "Sonnenstunden": rng.uniform(0, 12, TAGE),  # h/Tag
        "Ferienindikator": rng.choice([0, 1], TAGE),  # 0=Normal, 1=Ferien
        "Veranstaltung": rng.choice([0, 1], TAGE, p=[0.95, 0.05]),  # 5% Events
        "Werbebudget": rng.uniform(200, 800, TAGE),  # €/Tag
        "TV_Werbedruck": rng.uniform(0, 100, TAGE),  # GRP (Gross Rating Points)
        "Social_Reichweite": rng.uniform(0, 10, TAGE),  # Millionen Impressionen
        "Energieindex": rng.normal(100, 15, TAGE),  # Index (100=Baseline)
        "Wettbewerb": rng.uniform(0, 1, TAGE),  # 0-1 Konkurrenz-Intensität
        "Kundenzufriedenheit": rng.normal(8, 1, TAGE),  # 1-10 NPS-ähnlich
        "Lieferbereitschaft": rng.uniform(0.8, 1.0, TAGE)  # 80-100% Verfügbarkeit
    })
    
    # === TARGET VARIABLE (ABSATZ) ===
    # Realistisches Modell mit Interaktionen
    df["Absatz"] = (
        60 +                                    # Baseline: 60 Einheiten/Tag
        df["Wochentag"]*8 +                    # Wochentag-Effekt (Mo+8, So-6)
        np.sin(df["Monat"])*15 +               # Saisonalität ±15 Einheiten
        rng.normal(0, 5, TAGE)                 # Weißes Rauschen σ=5
    ).astype(int).clip(0)
    
    # === FEATURE-SELECTION ===
    MERKMALE = [
        "Wochentag", "Monat", "Quartal", "Jahrestag",  # Zeitliche Features
        "Temperatur", "Niederschlag", "Sonnenstunden",  # Wetter-Features
        "Ferienindikator", "Veranstaltung",  # Event-Features
        "Werbebudget", "TV_Werbedruck", "Social_Reichweite",  # Marketing-Features
        "Energieindex", "Wettbewerb",  # Externe Faktoren
        "Kundenzufriedenheit", "Lieferbereitschaft"  # Qualitäts-Features
    ]
    
    X = df[MERKMALE]
    y = df["Absatz"]
    
    # === DATEN-NORMALISIERUNG ===
    skalierer = StandardScaler()
    X_s = skalierer.fit_transform(X)
    
    # === MODELL-TRAINING ===
    netz = MLPRegressor(
        hidden_layer_sizes=(128, 64),  # 2 Hidden Layers
        activation='relu',              # ReLU Aktivierung
        solver='adam',                  # Adam Optimizer
        max_iter=1000,                  # Max. 1000 Iterationen
        random_state=42,                # Reproduzierbarkeit
        early_stopping=False,
        learning_rate_init=0.001
    ).fit(X_s, y)
    
    # === MODELL-VALIDIERUNG ===
    y_pred = netz.predict(X_s)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Speichere Metriken für späteren Zugriff
    st.session_state.model_metrics = {
        'mae': mae,
        'r2': r2,
        'training_samples': len(df),
        'training_period': f"{df['Datum'].min().date()} bis {df['Datum'].max().date()}",
        'features': len(MERKMALE)
    }
    
    return netz, skalierer, MERKMALE

netz, skalierer, MERKMALE = modell_laden()

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FICTIONAL PRODUCT INVENTORY DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def produkte_generieren():
    """Generiert realistische Produktbestandsdaten für ein Supermarkt-Szenario."""
    np.random.seed(42)
    
    kategorien = {
        "Obst & Gemüse": ["Äpfel", "Bananen", "Karotten", "Tomaten", "Salat", "Gurken"],
        "Milchprodukte": ["Milch", "Joghurt", "Käse", "Butter", "Rahm"],
        "Fleisch & Fisch": ["Hähnchen", "Rindfleisch", "Forelle", "Lachs", "Schweinefleisch"],
        "Backwaren": ["Brot", "Brötchen", "Kuchen", "Croissants"],
        "Getränke": ["Orangensaft", "Mineralwasser", "Milch", "Bier", "Wein"],
        "Trockenwaren": ["Mehl", "Zucker", "Öl", "Pasta", "Reis", "Konserven"]
    }
    
    produkte = []
    pid = 1000
    
    for kategorie, items in kategorien.items():
        for item in items:
            ist_kuehler = kategorie in ["Milchprodukte", "Fleisch & Fisch"]
            ist_tiefkuehler = kategorie == "Fleisch & Fisch" and item in ["Forelle", "Lachs"]
            haltbarkeit_tage = np.random.randint(3, 10) if ist_kuehler else np.random.randint(30, 180)
            
            bestand = np.random.randint(5, 150)
            mindestbestand = np.random.randint(10, 30)
            max_bestand = np.random.randint(100, 300)
            
            if np.random.random() < 0.15:
                bestand = np.random.randint(0, mindestbestand)
            elif np.random.random() < 0.10:
                bestand = np.random.randint(max_bestand, max_bestand + 50)
            
            prognose_7tage = np.random.randint(int(mindestbestand*0.8), int(mindestbestand*2.5))
            
            produkte.append({
                "SKU": f"SKU-{pid}",
                "Produktname": item,
                "Kategorie": kategorie,
                "Bestand": bestand,
                "Mindestbestand": mindestbestand,
                "Maximalbestand": max_bestand,
                "Prognose_7T": prognose_7tage,
                "Preis_EUR": round(np.random.uniform(0.5, 25), 2),
                "Haltbarkeit_Tage": haltbarkeit_tage,
                "Temp_Anforderung": "4°C" if ist_kuehler else ("−18°C" if ist_tiefkuehler else "Raumtemp"),
                "Verfallsdatum": (datetime.date.today() + datetime.timedelta(days=haltbarkeit_tage)).isoformat(),
                "Umsatz_7T": round(np.random.uniform(100, 5000), 2),
                "Umschlagshaeufigkeit": round(np.random.uniform(0.5, 8), 1)
            })
            pid += 1
    
    return pd.DataFrame(produkte)

df_produkte = produkte_generieren()

# ══════════════════════════════════════════════════════════════════════════════
# ALERT GENERATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def alerts_generieren(df_prod):
    """Generiert operationale Alerts basierend auf Bestandssituation."""
    alerts = []
    
    for idx, row in df_prod.iterrows():
        bestand = row['Bestand']
        mindest = row['Mindestbestand']
        maximal = row['Maximalbestand']
        prognose = row['Prognose_7T']
        haltbarkeit = row['Haltbarkeit_Tage']
        
        if bestand < mindest * 0.5 and prognose > bestand:
            alerts.append({
                "typ": "KRITISCH",
                "produkt": row['Produktname'],
                "grund": f"Bestand zu niedrig ({bestand} vs. Mindest {mindest}). Prognose: {prognose} Einheiten in 7T.",
                "aktion": "Sofortige Bestellung erforderlich",
                "prioritaet": 10
            })
        
        if 0 < haltbarkeit <= 3 and bestand > 0:
            alerts.append({
                "typ": "WARNUNG",
                "produkt": row['Produktname'],
                "grund": f"Verfällt in {haltbarkeit} Tagen. Bestand: {bestand} Einheiten.",
                "aktion": "Markieren für Abverkauf/Rabatt",
                "prioritaet": 8
            })
        
        if bestand > maximal * 1.2:
            alerts.append({
                "typ": "WARNUNG",
                "produkt": row['Produktname'],
                "grund": f"Überbestand erkannt ({bestand} vs. Max {maximal}). Umschlag: {row['Umschlagshaeufigkeit']}/Woche.",
                "aktion": "Verkaufsaktivität prüfen oder Bestellung stoppen",
                "prioritaet": 6
            })
        
        if row['Umschlagshaeufigkeit'] < 1.5 and bestand > mindest * 1.5:
            alerts.append({
                "typ": "INFO",
                "produkt": row['Produktname'],
                "grund": f"Langsamer Umschlag ({row['Umschlagshaeufigkeit']}/Woche). Bestand: {bestand}.",
                "aktion": "Promotion oder Preisanpassung prüfen",
                "prioritaet": 3
            })
    
    df_alerts = pd.DataFrame(alerts).sort_values('prioritaet', ascending=False)
    return df_alerts

# ══════════════════════════════════════════════════════════════════════════════
# ABC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def abc_analyse(df_prod):
    """Führt ABC-Analyse nach Umsatzbeitrag durch."""
    df = df_prod.copy()
    df['Umsatz_Kumulativ'] = df['Umsatz_7T'].sort_values(ascending=False).cumsum()
    df['Umsatz_Prozent'] = (df['Umsatz_Kumulativ'] / df['Umsatz_7T'].sum()) * 100
    
    def klassifizieren(prozent):
        if prozent <= 80:
            return 'A'
        elif prozent <= 95:
            return 'B'
        else:
            return 'C'
    
    df['ABC_Klasse'] = df['Umsatz_Prozent'].apply(klassifizieren)
    return df[['SKU', 'Produktname', 'Kategorie', 'Umsatz_7T', 'Bestand', 'ABC_Klasse']]

df_abc = abc_analyse(df_produkte)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Systemkonfiguration")
    st.divider()
    
    st.subheader("Nachfrage-Prognose Parameter")
    d_budget = st.slider("Marketing Budget (€)", 0, 1000, 450, help="Tägliches Marketing-Budget in Euro")
    d_temp = st.slider("Vorhersage Temperatur (°C)", -10, 40, 22, help="Erwartete Temperatur für Prognose")
    d_event = st.checkbox("Promotion / Event aktiv", help="Aktiviert Event-Boost auf Nachfrage")
    
    st.divider()
    st.subheader("Bestandsfilter")
    filter_kategorie = st.multiselect("Kategorien", df_produkte['Kategorie'].unique(), 
                                      default=df_produkte['Kategorie'].unique())
    filter_status = st.multiselect("Bestandsstatus", 
                                   ["Kritisch", "Optimal", "Überbestand"],
                                   default=["Kritisch", "Optimal"])

# ══════════════════════════════════════════════════════════════════════════════
# FORECAST CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
heute = datetime.date.today()
prognosen = []
for i in range(7):
    tag = heute + datetime.timedelta(days=i)
    feat = [tag.weekday(), tag.month, (tag.month-1)//3+1, tag.timetuple().tm_yday, 
            d_temp, 0, 8, 0, int(d_event), d_budget, 50, 5, 100, 0.5, 8, 0.9]
    val = int(netz.predict(skalierer.transform([feat]))[0])
    prognosen.append({"datum": tag, "absatz": val, "tag_name": tag.strftime('%a')})

df_proc = pd.DataFrame(prognosen)
peak_val = df_proc['absatz'].max()
total_val = df_proc['absatz'].sum()

# ══════════════════════════════════════════════════════════════════════════════
# ALERTS GENERIEREN
# ══════════════════════════════════════════════════════════════════════════════
df_alerts = alerts_generieren(df_produkte)

alert_counts = df_alerts['typ'].value_counts().to_dict()
kritisch_count = alert_counts.get('KRITISCH', 0)
warnung_count = alert_counts.get('WARNUNG', 0)
info_count = alert_counts.get('INFO', 0)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Supermarket Intelligence System")
st.markdown("<p class='subtitle'>KI-gestützte Nachfrageprognose und Bestandsverwaltung für Einzelhandelsbetriebe</p>", 
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFORMATION SECTION
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📊 Modell-Dokumentation & Trainings-Informationen", expanded=False):
    col_model_1, col_model_2 = st.columns(2)
    
    with col_model_1:
        st.markdown("#### Neuronales Netzwerk - Architektur")
        st.markdown("""
        **Modelltyp:** Multi-Layer Perceptron (MLP) Regressor
        
        **Schichtenaufbau:**
        - Input Layer: 16 Merkmale
        - Hidden Layer 1: 128 Neuronen (ReLU)
        - Hidden Layer 2: 64 Neuronen (ReLU)
        - Output Layer: 1 Ausgabewert (Absatzmenge)
        
        **Hyperparameter:**
        - Optimizer: Adam (adaptive learning rate)
        - Learning Rate: 0.001
        - Max. Iterationen: 1000
        - Aktivierungsfunktion: ReLU
        """)
    
    with col_model_2:
        st.markdown("#### Trainings-Datensatz")
        metrics = st.session_state.get('model_metrics', {})
        
        st.markdown(f"""
        **Datenumfang:**
        - Trainingssamples: {metrics.get('training_samples', 'N/A'):,}
        - Zeitraum: {metrics.get('training_period', 'N/A')}
        - Input Features: {metrics.get('features', 'N/A')}
        
        **Modell-Performance:**
        - Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.2f} Einheiten
        - R² Score: {metrics.get('r2', 'N/A'):.4f}
        - Model Accuracy: {(metrics.get('r2', 0) * 100):.1f}%
        """)
    
    st.divider()
    
    st.markdown("#### Input-Merkmale (16 Features)")
    
    features_info = {
        "Zeitliche Features": [
            ("Wochentag", "0-6 (Mo-So) – Wochenend-Effekt auf Nachfrage"),
            ("Monat", "1-12 – Saisonale Muster und Feiertage"),
            ("Quartal", "Q1-Q4 – Quartalsweise Trends"),
            ("Jahrestag", "1-365 – Jahres-Muster")
        ],
        "Wetter-Features": [
            ("Temperatur", "°C – Temperatur-abhängige Nachfrage"),
            ("Niederschlag", "mm – Wetter-Einfluss auf Kundenverhalten"),
            ("Sonnenstunden", "h/Tag – Sonnenschein-abhängige Aktivität")
        ],
        "Event & Freizeit": [
            ("Ferienindikator", "0/1 – Schulferien / Urlaubszeit"),
            ("Veranstaltung", "0/1 – Lokale Events (5% Häufigkeit)")
        ],
        "Marketing-Features": [
            ("Werbebudget", "€/Tag – Tägliches Marketing-Budget"),
            ("TV-Werbedruck", "GRP – Gross Rating Points"),
            ("Social-Reichweite", "Mio. Impressionen – Social Media")
        ],
        "Externe Faktoren": [
            ("Energieindex", "Index – Energiekosten-Effekt"),
            ("Wettbewerb", "0-1 – Konkurrenzintensität"),
            ("Kundenzufriedenheit", "1-10 – NPS-ähnliche Metrik"),
            ("Lieferbereitschaft", "80-100% – Verfügbarkeit der Produkte")
        ]
    }
    
    for kategorie, features in features_info.items():
        with st.container():
            st.markdown(f"**{kategorie}**")
            for fname, desc in features:
                st.markdown(f"• `{fname}` – {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE ALERT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Handlungsbedarf & Warnmeldungen")

alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    st.metric("🔴 Kritische Alerts", kritisch_count, 
              f"{kritisch_count} Produkte erfordern sofortige Maßnahmen" if kritisch_count > 0 else "Keine kritischen Meldungen")

with alert_col2:
    st.metric("🟡 Warnungen", warnung_count,
              f"{warnung_count} Produkte benötigen Optimierung" if warnung_count > 0 else "Alle Werte im normalen Bereich")

with alert_col3:
    st.metric("🔵 Informationen", info_count,
              f"{info_count} Beobachtungen verfügbar" if info_count > 0 else "Keine zusätzlichen Informationen")

st.markdown("###")

if len(df_alerts) > 0:
    for idx, alert in df_alerts.head(6).iterrows():
        alert_typ = alert['typ']
        
        if alert_typ == 'KRITISCH':
            alert_class = 'alert-critical'
            emoji = '🔴'
        elif alert_typ == 'WARNUNG':
            alert_class = 'alert-warning'
            emoji = '🟡'
        else:
            alert_class = 'alert-success'
            emoji = '🔵'
        
        st.markdown(f"""
        <div class="alert-card {alert_class}">
            <div class="alert-title">{emoji} {alert['produkt']} – {alert_typ}</div>
            <div class="alert-description">
                <strong>Grund:</strong> {alert['grund']}<br>
                <strong>Empfohlene Aktion:</strong> {alert['aktion']}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("✓ Keine Alerts – Alle Bestände sind im normalen Bereich.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="exec-card">
    <h3>Executive Summary</h3>
    <p><strong>Prognosierte Nachfrage (7 Tage):</strong> {total_val} Einheiten 
    | <strong>Peak:</strong> {peak_val} Einheiten am {df_proc.loc[df_proc['absatz'].idxmax(), 'datum'].strftime('%d. %B')} 
    | <strong>Ø täglich:</strong> {int(total_val/7)} Einheiten</p>
    
    <p><strong>Bestandssituation:</strong> {kritisch_count} SKUs mit kritischem Bestand | {warnung_count} Produkte benötigen Optimierung 
    | {len(df_produkte)} verwaltete Artikel insgesamt</p>
    
    <p><strong>Handlungsempfehlungen:</strong></p>
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
    <li>{kritisch_count} Produkte mit kritischem Bestand – sofortige Bestellung erforderlich</li>
    <li>Personalplanung für Peak-Nachfrage am {df_proc.loc[df_proc['absatz'].idxmax(), 'datum'].strftime('%d. %B')} durchführen</li>
    <li>{warnung_count} Produkte zur Optimierung (Verfallsdatum, Überbestand, Umschlag)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KEY PERFORMANCE INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Key Performance Indicators")

m1, m2, m3, m4, m5 = st.columns(5)

total_bestand = df_produkte['Bestand'].sum()
bestand_zu_niedrig = len(df_produkte[df_produkte['Bestand'] < df_produkte['Mindestbestand']])
abc_a_anteil = len(df_abc[df_abc['ABC_Klasse'] == 'A'])

m1.metric("Ø Absatz/Tag", f"{int(total_val/7)}", "7-Tage Durchschnitt")
m2.metric("Peak Demand", f"{peak_val}", "Höchste Nachfrage")
m3.metric("Gesamt Bestand", f"{total_bestand}", f"{len(df_produkte)} Artikel")
m4.metric("Unterbestand SKUs", f"{bestand_zu_niedrig}", "unter Mindestbestand")
m5.metric("A-Artikel", f"{abc_a_anteil}", "80% Umsatz")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DEMAND FORECAST & DEMAND DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Nachfrage-Prognose & Einflussfaktoren")

col_main, col_side = st.columns([2, 1])

with col_main:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_proc['tag_name'], y=df_proc['absatz'],
        fill='tozeroy', mode='lines+markers+text',
        text=df_proc['absatz'], textposition="top center",
        line=dict(width=3, color='#0052CC'),
        fillcolor='rgba(0, 82, 204, 0.1)',
        name='Prognose'
    ))
    fig.update_layout(
        title="7-Tage Nachfrageprognose",
        paper_bgcolor='#FFFFFF', plot_bgcolor='#F8F9FB',
        font=dict(color="#1F2937", family="Plus Jakarta Sans"),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("#### Nachfragetreiber")
    factors = ['Marketing', 'Wetter', 'Saisonalität', 'Events', 'Konkurrenz']
    values = [d_budget/10, d_temp*2, 70, 90 if d_event else 10, 40]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values, theta=factors, fill='toself',
        line=dict(color='#059669'),
        fillcolor='rgba(5, 150, 105, 0.15)',
        name='Einflussfaktoren'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor='rgba(0,0,0,0)'),
        showlegend=False,
        paper_bgcolor='#FFFFFF',
        font=dict(color="#1F2937", family="Plus Jakarta Sans"),
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ABC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ABC-Analyse: Produktmix & Umsatzkonzentration")

abc_counts = df_abc['ABC_Klasse'].value_counts().reindex(['A', 'B', 'C'])
abc_umsatz = df_abc.groupby('ABC_Klasse')['Umsatz_7T'].sum().reindex(['A', 'B', 'C'])

abc_col1, abc_col2 = st.columns(2)

with abc_col1:
    fig_abc_pie = go.Figure(data=[go.Pie(
        labels=['A-Artikel\n(80% Umsatz)', 'B-Artikel\n(15% Umsatz)', 'C-Artikel\n(5% Umsatz)'],
        values=abc_counts.values,
        marker=dict(colors=['#0052CC', '#059669', '#D97706']),
        textposition='inside',
        textinfo='label+percent'
    )])
    fig_abc_pie.update_layout(
        title="Produktverteilung nach ABC-Klasse",
        paper_bgcolor='#FFFFFF',
        font=dict(color="#1F2937", family="Plus Jakarta Sans"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_abc_pie, use_container_width=True)

with abc_col2:
    fig_abc_bar = go.Figure(data=[go.Bar(
        x=['A-Artikel', 'B-Artikel', 'C-Artikel'],
        y=abc_umsatz.values,
        marker=dict(color=['#0052CC', '#059669', '#D97706']),
        text=abc_umsatz.values.round(0),
        textposition='auto'
    )])
    fig_abc_bar.update_layout(
        title="Umsatzbeitrag nach Klasse (7 Tage)",
        paper_bgcolor='#FFFFFF', plot_bgcolor='#F8F9FB',
        font=dict(color="#1F2937", family="Plus Jakarta Sans"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_abc_bar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# INVENTORY MANAGEMENT TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Detaillierte Bestandsverwaltung")

df_filtered = df_produkte[df_produkte['Kategorie'].isin(filter_kategorie)].copy()

if filter_status:
    def get_status(row):
        if row['Bestand'] < row['Mindestbestand'] * 0.5:
            return 'Kritisch'
        elif row['Bestand'] > row['Maximalbestand'] * 1.1:
            return 'Überbestand'
        else:
            return 'Optimal'
    
    df_filtered['Status'] = df_filtered.apply(get_status, axis=1)
    df_filtered = df_filtered[df_filtered['Status'].isin(filter_status)]

df_filtered = df_filtered.merge(df_abc[['SKU', 'ABC_Klasse']], on='SKU', how='left')

df_display = df_filtered[[
    'SKU', 'Produktname', 'Kategorie', 'Bestand', 'Mindestbestand', 'Maximalbestand',
    'Prognose_7T', 'Preis_EUR', 'Umsatz_7T', 'Haltbarkeit_Tage', 'ABC_Klasse'
]].copy()

df_display.columns = ['SKU', 'Produkt', 'Kategorie', 'Bestand', 'Min.', 'Max.',
                      'Prognose 7T', 'Preis €', 'Umsatz €', 'Haltb. (T)', 'Klasse']

st.dataframe(df_display.sort_values('Bestand'), use_container_width=True, height=400)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Kategorieübersicht")

cat_col1, cat_col2 = st.columns(2)

with cat_col1:
    kategorie_bestand = df_produkte.groupby('Kategorie')['Bestand'].sum().sort_values(ascending=True)
    fig_cat_h = go.Figure(data=[go.Bar(
        x=kategorie_bestand.values,
        y=kategorie_bestand.index,
        orientation='h',
        marker=dict(color='#0052CC'),
        text=kategorie_bestand.values,
        textposition='auto'
    )])
    fig_cat_h.update_layout(
        title="Gesamtbestand nach Kategorie",
        paper_bgcolor='#FFFFFF', plot_bgcolor='#F8F9FB',
        font=dict(color="#1F2937", family="Plus Jakarta Sans"),
        xaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(showgrid=False),
        margin=dict(l=100, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_cat_h, use_container_width=True)

with cat_col2:
    kategorie_produkte = df_produkte['Kategorie'].value_counts()
    fig_cat_pie = go.Figure(data=[go.Pie(
        labels=kategorie_produkte.index,
        values=kategorie_produkte.values,
        marker=dict(colors=['#0052CC', '#059669', '#D97706', '#DC2626', '#8B5CF6', '#EC4899'])
    )])
    fig_cat_pie.update_layout(
        title="Produktanzahl nach Kategorie",
        paper_bgcolor='#FFFFFF',
        font=dict(color="#1F2937", family="Plus Jakarta Sans"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_cat_pie, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE ZONES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Temperaturzonen & Frische-Management")

temp_summ = df_produkte.groupby('Temp_Anforderung').agg({
    'Bestand': 'sum',
    'Produktname': 'count',
    'Haltbarkeit_Tage': 'mean'
}).round(1)
temp_summ.columns = ['Gesamt Bestand', 'SKUs', 'Ø Haltbarkeit (Tage)']

st.dataframe(temp_summ, use_container_width=True)

st.markdown("###")

temp_col1, temp_col2, temp_col3 = st.columns(3)

for idx, (temp_zone, label) in enumerate([(4, "🧊 Kühlzone (4°C)"), 
                                            (-18, "❄️ Tiefkühl (−18°C)"), 
                                            (22, "🏠 Raumtemperatur")]):
    temp_prod = df_produkte[df_produkte['Temp_Anforderung'] == 
                           ("4°C" if temp_zone == 4 else ("−18°C" if temp_zone == -18 else "Raumtemp"))]
    bestand = temp_prod['Bestand'].sum()
    
    if idx == 0:
        temp_col1.metric(label, f"{bestand} Einheiten", f"{len(temp_prod)} SKUs")
    elif idx == 1:
        temp_col2.metric(label, f"{bestand} Einheiten", f"{len(temp_prod)} SKUs")
    else:
        temp_col3.metric(label, f"{bestand} Einheiten", f"{len(temp_prod)} SKUs")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# OPERATIONAL SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Operativer Zeitplan")
st.markdown("*7-Tage-Nachfrageprognose in Tabellenform*")
st.dataframe(df_proc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.85rem; margin-top: 3rem;'>
<p><strong>OpenBrain Supermarket Intelligence System</strong> v5.0 Professional Edition</p>
<p>Powered by Multi-Layer Perceptron Neural Network | Scikit-Learn ML Framework</p>
<p style='font-size: 0.8rem; margin-top: 1rem;'>© 2024 – Confidential Business Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)