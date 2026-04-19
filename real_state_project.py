import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import TypedDict, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langgraph.graph import StateGraph, END 

# =============================================================================
# PAGE CONFIG — must be first
# =============================================================================
st.set_page_config(
    page_title="EstateAI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)


# =============================================================================
# GLOBAL CSS — Luxury Real Estate Theme
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:      #0B1B35;
    --navy-mid:  #112240;
    --navy-card: #162844;
    --gold:      #C9A84C;
    --gold-lt:   #E8C97A;
    --cream:     #F5F0E8;
    --text-main: #E8EAF0;
    --text-muted:#8A94A8;
    --green:     #2ECC71;
    --red:       #E74C3C;
    --blue-acc:  #4A9EFF;
    --border:    rgba(201,168,76,0.25);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text-main) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

[data-testid="stSidebar"] {
    background: var(--navy-mid) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #0B1B35 0%, #1a2f52 50%, #0B1B35 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(74,158,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.15);
    border: 1px solid rgba(201,168,76,0.4);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    color: var(--gold) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 700;
    color: var(--cream) !important;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted) !important;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin: 0;
}

/* Section headings */
.section-heading {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--cream) !important;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    margin-bottom: 1.3rem;
    font-weight: 300;
}
.gold-line {
    width: 48px; height: 2px;
    background: var(--gold);
    margin-bottom: 1.8rem;
    border-radius: 2px;
}

/* Input group cards */
.input-group-card {
    background: var(--navy-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.6rem 1.2rem;
    height: 100%;
}
.input-group-icon { font-size: 1.4rem; margin-bottom: 0.2rem; }
.input-group-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--gold) !important;
    margin-bottom: 1rem;
}

/* Labels */
label, .stNumberInput label, .stSlider label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Inputs */
[data-testid="stNumberInput"] input {
    background: #0D2040 !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
    font-size: 1rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* CTA Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #C9A84C, #A8883A) !important;
    color: #0B1B35 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 2rem !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #E8C97A, #C9A84C) !important;
    box-shadow: 0 6px 28px rgba(201,168,76,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: var(--navy-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--cream) !important;
    padding: 1rem 1.3rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--gold) !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(201,168,76,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.4rem !important;
    color: var(--gold) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Sidebar pills */
.pill {
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.73rem;
    color: var(--gold) !important;
    margin: 0.15rem;
    font-weight: 500;
}

/* Sidebar section label */
.sidebar-section {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--gold) !important;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* Workflow box */
.workflow-box {
    background: #0D2040;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: var(--text-muted) !important;
    line-height: 1.9;
}
.workflow-box .node { color: var(--gold) !important; font-weight: 700; }
.workflow-box .arrow { color: var(--blue-acc) !important; }

/* Price hero */
.price-hero {
    background: linear-gradient(135deg, rgba(201,168,76,0.15), rgba(201,168,76,0.04));
    border: 1px solid rgba(201,168,76,0.45);
    border-radius: 14px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.price-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem;
}
.price-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 700;
    color: var(--gold) !important;
    line-height: 1;
}
.price-sub {
    font-size: 0.9rem;
    color: var(--text-muted) !important;
    margin-top: 0.6rem;
}

/* Metric grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.metric-card {
    background: var(--navy);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--gold) !important;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.68rem;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Report section title */
.report-section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--cream) !important;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}

/* Insight rows */
.insight-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(201,168,76,0.08);
}
.insight-num {
    min-width: 26px; height: 26px;
    background: var(--gold);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    color: #0B1B35; flex-shrink: 0;
}
.insight-text {
    font-size: 0.88rem;
    color: var(--text-main) !important;
    line-height: 1.6;
    padding-top: 0.1rem;
}

/* Advice card */
.advice-card {
    background: var(--navy);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    font-size: 0.9rem;
    color: var(--text-main) !important;
    line-height: 1.85;
}

/* Disclaimer */
.disclaimer-box {
    background: rgba(231,76,60,0.07);
    border: 1px solid rgba(231,76,60,0.3);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-top: 2rem;
}
.disclaimer-title {
    font-weight: 700;
    color: #E74C3C !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.7rem;
}
.disclaimer-text {
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    line-height: 1.75;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# TYPED STATE
# =============================================================================
class AgentState(TypedDict):
    input:           Dict[str, Any]
    predicted_price: float
    market_data:     List[str]
    comps:           List[Dict]
    final_advice:    str
    model_metrics:   Dict[str, float]
    error:           str

# =============================================================================
# DATA LOADING & CLEANING
# =============================================================================
@st.cache_data
def load_data(path: str = "V3.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error("❌ Dataset 'V3.csv' not found. Place it in the project root.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Locality"]        = df["Locality"].fillna(df["Locality"].mode()[0])
    df["carpet_area"]     = df["carpet_area"].fillna(df["carpet_area"].median())
    df["Estimated Value"] = df["Estimated Value"].fillna(df["Estimated Value"].median())
    df = df.drop_duplicates()
    cat_cols = [c for c in ["Locality","Property","Residential","Face"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    if "Date" in df.columns:
        df["Date"]  = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df = df.drop("Date", axis=1)
    return df

# =============================================================================
# MODEL TRAINING
# =============================================================================
@st.cache_resource
def train_model(_df: pd.DataFrame):
    base_features   = ["property_tax_rate","carpet_area","num_bathrooms","num_rooms","Estimated Value","Year","month"]
    encoded_cols    = sorted([c for c in _df.columns if c not in base_features + ["Sale Price"]])
    feature_columns = [c for c in base_features + encoded_cols if c in _df.columns]
    X = _df[feature_columns];  y = _df["Sale Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    metrics = {
        "R² Score":   round(r2_score(y_test, y_pred), 4),
        "MAE":        round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE":       round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "Train Rows": int(len(X_train)),
        "Test Rows":  int(len(X_test)),
    }
    return mdl, feature_columns, metrics

# =============================================================================
# INPUT VALIDATION
# =============================================================================
def validate_input(data: Dict[str, Any]) -> List[str]:
    errors = []
    if data["carpet_area"] <= 0:                    errors.append("Carpet area must be > 0.")
    if data["carpet_area"] > 50_000:                errors.append("Carpet area too large (> 50,000 sq ft).")
    if not (1 <= data["num_rooms"] <= 20):           errors.append("Rooms must be 1–20.")
    if not (1 <= data["num_bathrooms"] <= 15):       errors.append("Bathrooms must be 1–15.")
    if data["num_bathrooms"] > data["num_rooms"]:    errors.append("Bathrooms cannot exceed rooms.")
    if not (0.0 <= data["property_tax_rate"] <= 10): errors.append("Tax rate must be 0–10%.")
    if data["Estimated Value"] <= 0:                 errors.append("Estimated value must be positive.")
    if not (1900 <= data["Year"] <= 2030):           errors.append("Year must be 1900–2030.")
    return errors