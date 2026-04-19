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

# =============================================================================
# PREDICTION
# =============================================================================
def predict_price(input_data, mdl, feature_columns) -> float:
    filled = {col: input_data.get(col, 0) for col in feature_columns}
    return float(mdl.predict(pd.DataFrame([filled]))[0])

# =============================================================================
# COMPARABLE PROPERTIES
# =============================================================================
def get_comparable_properties(input_data, df_raw, n=3) -> List[Dict]:
    try:
        required = {"carpet_area","num_rooms","Sale Price"}
        if not required.issubset(df_raw.columns): return []
        df_c  = df_raw.copy()
        area  = input_data.get("carpet_area", 1000)
        rooms = input_data.get("num_rooms", 3)
        mask  = (df_c["carpet_area"] >= area * 0.7) & (df_c["carpet_area"] <= area * 1.3)
        filt  = df_c[mask & (df_c["num_rooms"] == rooms)]
        if len(filt) < n: filt = df_c[mask]
        if filt.empty: return []
        sample = filt.sample(min(n, len(filt)), random_state=42)
        return [{
            "Carpet Area (sq ft)": f"{row.get('carpet_area','N/A'):,.0f}",
            "Rooms":               int(row.get("num_rooms", 0)),
            "Bathrooms":           int(row.get("num_bathrooms", 0)),
            "Sale Price":          f"₹{row.get('Sale Price', 0):,.0f}",
            "Est. Value":          f"₹{row.get('Estimated Value', 0):,.0f}",
        } for _, row in sample.iterrows()]
    except Exception as e:
        logger.error(f"Comps error: {e}"); return []

# =============================================================================
# RAG CORPUS
# =============================================================================
MARKET_CORPUS = [
    "Gurgaon property prices are rising 8% annually driven by IT sector growth.",
    "Gurgaon Golf Course Road has premium residential properties above Rs 10,000 per sq ft.",
    "Gurgaon Dwarka Expressway offers affordable housing with excellent metro connectivity.",
    "Cyber City Gurgaon drives significant commercial real estate demand.",
    "Gurgaon residential rental yields average 3-4% and commercial 6-8% annually.",
    "Noida offers high rental yield of 4-5% for 2BHK apartments.",
    "Noida Sector 150 has emerged as a premium residential destination with sports infrastructure.",
    "Noida Extension Greater Noida West offers affordable housing below Rs 5,000 per sq ft.",
    "Noida metro connectivity boosts adjacent property values by 15-20%.",
    "Noida Authority plots appreciated 40% in three years due to infrastructure expansion.",
    "Delhi real estate is stable but expensive with limited new housing supply.",
    "South Delhi premium micro-markets like Hauz Khas see prices above Rs 25,000 per sq ft.",
    "Delhi Dwarka is a popular middle-income housing destination near the international airport.",
    "Rohini Delhi offers moderate property prices with strong social infrastructure.",
    "East Delhi markets like Laxmi Nagar offer affordable options below Rs 8,000 per sq ft.",
    "Sonipat Haryana property market is growing due to proximity to Delhi and KMP Expressway.",
    "IMT Manesar and Kundli Industrial clusters are boosting Sonipat realty demand.",
    "Haryana RERA has registered over 800 projects ensuring buyer protection.",
    "Stamp duty in Haryana is 7% for male buyers and 5% for female buyers.",
    "Haryana affordable housing scheme offers properties at fixed rates for EWS and LIG.",
    "Properties within 500 metres of metro stations appreciate 10-15% faster than city average.",
    "New highway or expressway announcements increase nearby property values within 6-12 months.",
    "Airport proximity raises commercial and residential property values significantly.",
    "Smart city project zones carry a 5-10% price premium over comparable non-smart areas.",
    "SEZ and IT park proximity increases residential rental demand substantially.",
    "NRI investment in Indian real estate is fully permitted under FEMA regulations.",
    "RERA registration is mandatory for all real estate projects above 500 sq metres in India.",
    "GST on under-construction properties is 5% without input tax credit benefit.",
    "Capital gains tax on property held over 2 years is 20% with indexation benefits.",
    "Home loan interest deduction up to Rs 2 lakh per year is available under Section 24.",
    "Principal repayment qualifies for deduction up to Rs 1.5 lakh under Section 80C.",
    "TDS at 1% is deducted by the buyer for property transactions above Rs 50 lakhs.",
    "Indian residential real estate grew 10% in 2024 with 4.1 lakh new unit launches.",
    "Luxury housing above Rs 1.5 crore has grown fastest at 18% in 2024.",
    "Affordable housing under Rs 45 lakh saw demand recovery after the COVID-19 slowdown.",
    "Co-living and co-working segments are growing at 25% annually post-pandemic.",
    "Green certified buildings command a 5-7% price premium in major Indian cities.",
    "Commercial office absorption in NCR crossed 10 million sq ft in 2024.",
    "Rental housing demand surged 30% in Bengaluru, Hyderabad and Pune in 2024.",
    "Tier 2 cities like Lucknow, Indore and Surat are emerging real estate hotspots.",
    "Rising home loan interest rates above 8.5% can reduce affordability by 10-15%.",
    "Builder project delays and cancellations remain a key risk for under-construction properties.",
    "Over-supply in certain Noida micro-markets can depress capital appreciation for years.",
    "Legal title disputes are common in older properties and require thorough due diligence.",
    "Flood-prone or seismically active zones carry higher insurance costs and investment risk.",
]

@st.cache_resource
def get_vectorstore():
    try:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(MARKET_CORPUS, emb)
    except Exception as e:
        logger.error(f"Vectorstore init failed: {e}"); return None

def retrieve_market(query: str, vs) -> List[str]:
    if vs is None: return ["Market data unavailable."]
    try:
        return [d.page_content for d in vs.similarity_search(query, k=4)]
    except Exception as e:
        return [f"Retrieval error: {e}"]
    
# =============================================================================
# LLM
# =============================================================================
@st.cache_resource
def get_llm():
    try:
        return pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception as e:
        logger.error(f"LLM init failed: {e}"); return None

def generate_advice(input_data, price, market, llm) -> str:
    if llm is None: return "AI advisor unavailable."
    prop = (
        f"Carpet Area: {input_data.get('carpet_area','N/A')} sq ft | "
        f"Rooms: {input_data.get('num_rooms','N/A')} | "
        f"Bathrooms: {input_data.get('num_bathrooms','N/A')} | "
        f"Tax Rate: {input_data.get('property_tax_rate','N/A')}% | "
        f"Estimated Value: Rs {input_data.get('Estimated Value',0):,.0f} | "
        f"Year: {input_data.get('Year','N/A')}"
    )
    mkt = " | ".join(market[:3])
    prompt = (
        "You are a certified real estate investment advisor in India. "
        "Only answer about real estate. Do not invent price figures. "
        f"Property: {prop}\nPredicted Price: Rs {price:,.0f}\nMarket: {mkt}\n\n"
        "Provide:\n1. Valuation Summary\n2. Recommendation (Buy/Hold/Avoid)\n"
        "3. Key Risk Factors\n4. Final Advice\n5. Disclaimer"
    )
    try:
        result = llm(prompt, max_new_tokens=300, truncation=True)
        advice = result[0]["generated_text"].strip()
        if not advice or len(advice) < 20:
            return "AI advisor could not generate a response for these inputs."
        if any(p in advice.lower() for p in ["ignore previous","disregard","jailbreak"]):
            return "Suspicious output detected. Please try again."
        return advice
    except Exception as e:
        return f"Advice generation failed: {e}"

# =============================================================================
# LANGGRAPH NODES
# =============================================================================
def predict_node(state: AgentState) -> AgentState:
    try:
        state["predicted_price"] = predict_price(
            state["input"], st.session_state["model"], st.session_state["feature_columns"])
    except Exception as e:
        state["error"] = f"Prediction: {e}"; state["predicted_price"] = 0.0
    return state

def rag_node(state: AgentState) -> AgentState:
    try:
        q = f"property investment {state['input'].get('carpet_area','')} sqft {state['input'].get('num_rooms','')} rooms India"
        state["market_data"] = retrieve_market(q, st.session_state["vectorstore"])
    except Exception as e:
        state["error"] = f"RAG: {e}"; state["market_data"] = []
    return state

def comps_node(state: AgentState) -> AgentState:
    try:
        state["comps"] = get_comparable_properties(state["input"], st.session_state["df_raw"])
    except Exception as e:
        state["error"] = f"Comps: {e}"; state["comps"] = []
    return state

def advisor_node(state: AgentState) -> AgentState:
    try:
        state["final_advice"] = generate_advice(
            state["input"], state["predicted_price"],
            state["market_data"], st.session_state["llm"])
    except Exception as e:
        state["error"] = f"Advisor: {e}"; state["final_advice"] = "Advisory unavailable."
    return state

@st.cache_resource
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("predict", predict_node)
    g.add_node("rag",     rag_node)
    g.add_node("comps",   comps_node)
    g.add_node("advisor", advisor_node)
    g.set_entry_point("predict")
    g.add_edge("predict","rag")
    g.add_edge("rag","comps")
    g.add_edge("comps","advisor")
    g.add_edge("advisor", END)
    return g.compile()

# =============================================================================
# INITIALISE RESOURCES
# =============================================================================
df_raw         = load_data()
df_clean       = clean_data(df_raw)
model, feature_columns, model_metrics = train_model(df_clean)
vectorstore    = get_vectorstore()
llm            = get_llm()
agent_app      = build_graph()

st.session_state["model"]           = model
st.session_state["feature_columns"] = feature_columns
st.session_state["vectorstore"]     = vectorstore
st.session_state["llm"]             = llm
st.session_state["df_raw"]          = df_raw

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-section">📊 Model Performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² Score",   f"{model_metrics['R² Score']:.4f}")
        st.metric("Train Rows", f"{model_metrics['Train Rows']:,}")
    with c2:
        st.metric("MAE",        f"₹{model_metrics['MAE']:,.0f}")
        st.metric("Test Rows",  f"{model_metrics['Test Rows']:,}")
    st.metric("RMSE", f"₹{model_metrics['RMSE']:,.0f}")

    st.markdown('<div class="sidebar-section">🔗 Agent Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-box">
        Input <span class="arrow">→</span> <span class="node">predict_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">rag_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">comps_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">advisor_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">END</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Stack</div>', unsafe_allow_html=True)
    for pill in ["Random Forest","FAISS RAG","Flan-T5","LangGraph","Streamlit"]:
        st.markdown(f'<span class="pill">{pill}</span>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("EstateAI v2.0 · For informational use only")

# =============================================================================
# HERO BANNER
# =============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🏠 AI-Powered Real Estate Intelligence</div>
    <div class="hero-title">Estate<span>AI</span></div>
    <p class="hero-sub">
        Agentic property valuation &nbsp;·&nbsp; FAISS market retrieval &nbsp;·&nbsp; Structured investment advisory
    </p>
</div>
""", unsafe_allow_html=True)