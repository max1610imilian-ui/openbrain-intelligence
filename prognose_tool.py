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
    page_title="OpenBrain · Executive Demand Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ENTERPRISE DARK MODE CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0B0F19; --surface: #111827; --border: #1F2937; --accent: #3B82F6;
    --text: #F3F4F6; --ok: #10B981; --warn: #F59E0B; --crit: #EF4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Executive Summary Box */
.exec-card {
    background: linear-gradient(145deg, #1e293b, #111827);
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.alert-card {
    background: linear-gradient(145deg, #1e293b, #111827);
    border: 1px solid var(--border);
    border-left: 5px solid;
    padding: 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.alert-critical { border-left-color: #EF4444; }
.alert-warning { border-left-color: #F59E0B; }
.alert-success { border-left-color: #10B981; }

.alert-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}

.alert-critical .alert-title { color: #EF4444; }
.alert-warning .alert-title { color: #F59E0B; }
.alert-success .alert-title { color: #10B981; }

.alert-description {
    font-size: 0.85rem;
    color: #D1D5DB;
    line-height: 1.4;
}

.action-item {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.dot { height: 8px; width: 8px; border-radius: 50%; margin-right: 10px; display: inline-block; }

[data-testid="stMetricContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.day-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}

.product-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-right: 0.5rem;
    margin-bottom: 0.25rem;
}

.badge-a { background: rgba(59, 130, 246, 0.2); color: #60A5FA; }
.badge-b { background: rgba(16, 185, 129, 0.2); color: #34D399; }
.badge-c { background: rgba(245, 158, 11, 0.2); color: #FBBF24; }
.badge-critical { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; }
.badge-low { background: rgba(249, 115, 22, 0.2); color: #FB923C; }

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.5rem;
    vertical-align: middle;
}

.status-ok { background-color: #10B981; }
.status-warn { background-color: #F59E0B; }
.status-crit { background-color: #EF4444; }

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD NEURAL NETWORK MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def modell_laden():
    TAGE = 5000
    rng = np.random.default_rng(42)
    daten = pd.date_range("2011-01-01", periods=TAGE)
    df = pd.DataFrame({
        "Datum": daten, "Wochentag": daten.weekday, "Monat": daten.month, "Quartal": daten.quarter,
        "Jahrestag": daten.dayofyear, "Temperatur": rng.normal(15, 10, TAGE),
        "Niederschlag": rng.exponential(2, TAGE), "Sonnenstunden": rng.uniform(0, 12, TAGE),
        "Ferienindikator": rng.choice([0, 1], TAGE), "Veranstaltung": rng.choice([0, 1], TAGE, p=[0.95, 0.05]),
        "Werbebudget": rng.uniform(200, 800, TAGE), "TV_Werbedruck": rng.uniform(0, 100, TAGE),
        "Social_Reichweite": rng.uniform(0, 10, TAGE), "Energieindex": rng.normal(100, 15, TAGE),
        "Wettbewerb": rng.uniform(0, 1, TAGE), "Kundenzufriedenheit": rng.normal(8, 1, TAGE),
        "Lieferbereitschaft": rng.uniform(0.8, 1.0, TAGE)
    })
    df["Absatz"] = (60 + df["Wochentag"]*8 + np.sin(df["Monat"])*15 + rng.normal(0, 5, TAGE)).astype(int).clip(0)
    MERKMALE = ["Wochentag", "Monat", "Quartal", "Jahrestag", "Temperatur", "Niederschlag", "Sonnenstunden", 
                "Ferienindikator", "Veranstaltung", "Werbebudget", "TV_Werbedruck", "Social_Reichweite", 
                "Energieindex", "Wettbewerb", "Kundenzufriedenheit", "Lieferbereitschaft"]
    X = df[MERKMALE]
    y = df["Absatz"]
    skalierer = StandardScaler()
    X_s = skalierer.fit_transform(X)
    netz = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42).fit(X_s, y)
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
            
            # Aktuelle Bestände mit realistischen Szenarien
            bestand = np.random.randint(5, 150)
            mindestbestand = np.random.randint(10, 30)
            max_bestand = np.random.randint(100, 300)
            
            # Einige Produkte mit kritischen Beständen
            if np.random.random() < 0.15:  # 15% kritisch
                bestand = np.random.randint(0, mindestbestand)
            elif np.random.random() < 0.10:  # 10% zu viel
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
        umsatz = row['Umsatz_7T']
        
        # KRITISCH: Bestand < Mindestbestand + bevorstehende Nachfrage
        if bestand < mindest * 0.5 and prognose > bestand:
            alerts.append({
                "typ": "KRITISCH",
                "produkt": row['Produktname'],
                "grund": f"Bestand zu niedrig ({bestand} vs. Mindest {mindest}). Prognose: {prognose} Einheiten in 7T.",
                "action": "Sofortige Bestellung erforderlich",
                "prioritaet": 10
            })
        
        # WARNUNG: Verfallsdatum nahe (< 3 Tage)
        if 0 < haltbarkeit <= 3 and bestand > 0:
            alerts.append({
                "typ": "WARNUNG",
                "produkt": row['Produktname'],
                "grund": f"Verfällt in {haltbarkeit} Tagen. Bestand: {bestand} Einheiten.",
                "action": "Markieren für Abverkauf/Rabatt",
                "prioritaet": 8
            })
        
        # WARNUNG: Überbestand (> 120% Maximalbestand)
        if bestand > maximal * 1.2:
            alerts.append({
                "typ": "WARNUNG",
                "produkt": row['Produktname'],
                "grund": f"Überbestand erkannt ({bestand} vs. Max {maximal}). Umschlag: {row['Umschlagshaeufigkeit']}/Woche.",
                "action": "Verkaufsaktivität prüfen oder Bestellung stoppen",
                "prioritaet": 6
            })
        
        # INFO: Langsame Umschlagsquote + hohes Lager
        if row['Umschlagshaeufigkeit'] < 1.5 and bestand > mindest * 1.5:
            alerts.append({
                "typ": "INFO",
                "produkt": row['Produktname'],
                "grund": f"Langsamer Umschlag ({row['Umschlagshaeufigkeit']}/Woche). Bestand: {bestand}.",
                "action": "Promotion oder Preisanpassung prüfen",
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
    st.title("◈ Controls & Filter")
    st.divider()
    
    d_budget = st.slider("Marketing Budget (€)", 0, 1000, 450)
    d_temp = st.slider("Forecast Temp (°C)", -10, 40, 22)
    d_event = st.checkbox("Promotion / Event aktiv")
    
    st.divider()
    st.subheader("Bestandsfilter")
    filter_kategorie = st.multiselect("Kategorie", df_produkte['Kategorie'].unique(), 
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

# Count alerts by type
alert_counts = df_alerts['typ'].value_counts().to_dict()
kritisch_count = alert_counts.get('KRITISCH', 0)
warnung_count = alert_counts.get('WARNUNG', 0)
info_count = alert_counts.get('INFO', 0)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 style="font-family:Crimson Text; font-size:2.8rem;">Intelligence Dashboard</h1>', 
            unsafe_allow_html=True)

st.markdown("""
<p style="color:#9CA3AF; font-size:0.9rem; margin-top:-1rem;">
Echtzeit-Überwachung von Nachfrage, Bestand und Operationen | KI-gestützte Prognosen & Handlungsempfehlungen
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE ALERT SECTION (TOP PRIORITY)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### ⚠️ Handlungsbedarf & Alerts")

alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    st.metric("🔴 Kritische Alerts", kritisch_count, 
              f"{kritisch_count} Produkte sofort prüfen" if kritisch_count > 0 else "Keine")

with alert_col2:
    st.metric("🟡 Warnungen", warnung_count,
              f"{warnung_count} zur Optimierung" if warnung_count > 0 else "Status OK")

with alert_col3:
    st.metric("🔵 Informationen", info_count,
              f"{info_count} zur Beobachtung" if info_count > 0 else "Alle normal")

st.markdown("###")

# Display top 5 alerts
if len(df_alerts) > 0:
    for idx, alert in df_alerts.head(5).iterrows():
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
            <div class="alert-title">{emoji} {alert['produkt']} · {alert_typ}</div>
            <div class="alert-description">
                <strong>Grund:</strong> {alert['grund']}<br>
                <strong>Aktion:</strong> {alert['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="exec-card">
    <h3 style="margin-top:0; color:#3B82F6;">◈ Management Executive Summary</h3>
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 300px;">
            <p><strong>Aktuelle Erkenntnisse:</strong><br>
            Die KI prognostiziert für die kommende Woche ein Gesamtvolumen von <b>{total_val} Einheiten</b> (+12% zum Vormonat). 
            Der Nachfrage-Peak wird am <b>{df_proc.loc[df_proc['absatz'].idxmax(), 'datum'].strftime('%d.%m.')}</b> erwartet.
            <br><br>
            <strong>Bestandssituation:</strong> Von {len(df_produkte)} verwalteten SKUs haben {kritisch_count} kritische Bestände 
            und {warnung_count} Produkte benötigen Aufmerksamkeit.</p>
        </div>
        <div style="flex: 1; min-width: 300px; border-left: 1px solid #334155; padding-left: 1.5rem;">
            <p style="font-weight:600; color:#9CA3AF; font-size:0.8rem; text-transform:uppercase;">Handlungsbedarf</p>
            <div class="action-item"><span class="dot" style="background:#EF4444;"></span> {kritisch_count} Produkte mit kritischem Bestand – sofortige Bestellung.</div>
            <div class="action-item"><span class="dot" style="background:#F59E0B;"></span> {warnung_count} Produkte benötigen Optimierung (Verfallsdatum, Überbestand).</div>
            <div class="action-item"><span class="dot" style="background:#10B981;"></span> Personalplanung für Peak-Tag {df_proc.loc[df_proc['absatz'].idxmax(), 'datum'].strftime('%d.%m.')} optimieren.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KEY METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Key Performance Indicators")
m1, m2, m3, m4, m5 = st.columns(5)

total_bestand = df_produkte['Bestand'].sum()
bestand_zu_niedrig = len(df_produkte[df_produkte['Bestand'] < df_produkte['Mindestbestand']])
abc_a_anteil = len(df_abc[df_abc['ABC_Klasse'] == 'A'])

m1.metric("Ø Absatz / Tag", f"{int(total_val/7)}")
m2.metric("Peak Demand", f"{peak_val}")
m3.metric("Gesamt Bestand", f"{total_bestand}")
m4.metric("Unterbestand SKUs", f"{bestand_zu_niedrig}")
m5.metric("A-Artikel (80%)", f"{abc_a_anteil}")

st.markdown("###")

# ══════════════════════════════════════════════════════════════════════════════
# DEMAND FORECAST & PRODUCT MIX
# ══════════════════════════════════════════════════════════════════════════════
col_main, col_side = st.columns([2, 1])

with col_main:
    # Fortgeschrittener Area Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_proc['tag_name'], y=df_proc['absatz'],
        fill='tozeroy', mode='lines+markers+text',
        text=df_proc['absatz'], textposition="top center",
        line=dict(width=4, color='#3B82F6'),
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    fig.update_layout(
        title="Nachfrageprognose (7 Tage)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CA3AF"), margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1F2937")
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("<p style='text-align:center; font-weight:600; color:#9CA3AF;'>Nachfragetreiber</p>", unsafe_allow_html=True)
    factors = ['Marketing', 'Wetter', 'Saisonalität', 'Events', 'Konkurrenz']
    values = [d_budget/10, d_temp*2, 70, 90 if d_event else 10, 40]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values, theta=factors, fill='toself',
        line=dict(color='#10B981'), fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ABC ANALYSIS CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📈 ABC-Analyse: Produktmix & Umsatzkonzentration")

abc_counts = df_abc['ABC_Klasse'].value_counts().reindex(['A', 'B', 'C'])
abc_umsatz = df_abc.groupby('ABC_Klasse')['Umsatz_7T'].sum().reindex(['A', 'B', 'C'])

abc_col1, abc_col2 = st.columns(2)

with abc_col1:
    fig_abc_pie = go.Figure(data=[go.Pie(
        labels=abc_counts.index,
        values=abc_counts.values,
        marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B']),
        hole=0.4
    )])
    fig_abc_pie.update_layout(
        title="Produkte nach Klasse",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CA3AF"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_abc_pie, use_container_width=True)

with abc_col2:
    fig_abc_bar = go.Figure(data=[go.Bar(
        x=abc_umsatz.index,
        y=abc_umsatz.values,
        marker=dict(color=['#3B82F6', '#10B981', '#F59E0B']),
        text=abc_umsatz.values.round(0),
        textposition='auto'
    )])
    fig_abc_bar.update_layout(
        title="Umsatzbeitrag (7 Tage)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CA3AF"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1F2937"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_abc_bar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# INVENTORY DETAIL TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📦 Detaillierte Bestandsverwaltung")

# Filter products
df_filtered = df_produkte[df_produkte['Kategorie'].isin(filter_kategorie)].copy()

# Apply status filter
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

# Merge with ABC classification
df_filtered = df_filtered.merge(df_abc[['SKU', 'ABC_Klasse']], on='SKU', how='left')

# Format for display
df_display = df_filtered[[
    'SKU', 'Produktname', 'Kategorie', 'Bestand', 'Mindestbestand', 'Maximalbestand',
    'Prognose_7T', 'Preis_EUR', 'Umsatz_7T', 'Haltbarkeit_Tage', 'ABC_Klasse'
]].copy()

df_display.columns = ['SKU', 'Produkt', 'Kategorie', 'Bestand', 'Min.', 'Max.',
                      'Prognose 7T', 'Preis €', 'Umsatz €', 'Halt. (T)', 'Klasse']

# Color coding for bestand
def format_bestand(val):
    if val < 5:
        return f'<span style="color:#EF4444;">●</span> {val}'
    elif val > 120:
        return f'<span style="color:#F59E0B;">●</span> {val}'
    else:
        return f'<span style="color:#10B981;">●</span> {val}'

# Style the dataframe (convert to HTML for custom formatting)
st.dataframe(df_display.sort_values('Bestand'), use_container_width=True, height=400)

st.markdown("###")

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏪 Kategorie-Überblick")

cat_col1, cat_col2 = st.columns(2)

with cat_col1:
    kategorie_bestand = df_produkte.groupby('Kategorie')['Bestand'].sum().sort_values(ascending=True)
    fig_cat_h = go.Figure(data=[go.Bar(
        x=kategorie_bestand.values,
        y=kategorie_bestand.index,
        orientation='h',
        marker=dict(color='#3B82F6'),
        text=kategorie_bestand.values,
        textposition='auto'
    )])
    fig_cat_h.update_layout(
        title="Gesamtbestand nach Kategorie",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CA3AF"),
        xaxis=dict(showgrid=True, gridcolor="#1F2937"),
        yaxis=dict(showgrid=False),
        margin=dict(l=100, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_cat_h, use_container_width=True)

with cat_col2:
    kategorie_produkte = df_produkte['Kategorie'].value_counts()
    fig_cat_pie = go.Figure(data=[go.Pie(
        labels=kategorie_produkte.index,
        values=kategorie_produkte.values,
        marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'])
    )])
    fig_cat_pie.update_layout(
        title="Produktanzahl nach Kategorie",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CA3AF"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_cat_pie, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE REQUIREMENTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🌡️ Temperaturzonen & Frische-Management")

temp_summ = df_produkte.groupby('Temp_Anforderung').agg({
    'Bestand': 'sum',
    'Produktname': 'count',
    'Haltbarkeit_Tage': 'mean'
}).round(1)
temp_summ.columns = ['Gesamtbestand', 'Anzahl Produkte', 'Ø Haltbarkeit (Tage)']

st.dataframe(temp_summ, use_container_width=True)

# Temperature zone inventory
temp_col1, temp_col2, temp_col3 = st.columns(3)

for idx, (temp_zone, label) in enumerate([(4, "🧊 Kühlzone (4°C)"), 
                                            (-18, "❄️ Tiefkühl (-18°C)"), 
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
st.markdown("### 📅 Operativer Zeitplan – 7-Tage-Nachfrageprognose")
st.dataframe(df_proc, use_container_width=True)

st.markdown('<div style="text-align:center; color:#4B5563; font-size:0.7rem; margin-top:4rem;">OpenBrain Intelligence Unit · v5.0 Premium | Supermarket Inventory & Demand Intelligence</div>', 
            unsafe_allow_html=True)