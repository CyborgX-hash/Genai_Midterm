import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="EstateAI",
    page_icon="🏠",
    layout="wide"
)

# -------------------------
# ADVANCED UI STYLING
# -------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

.stMetric {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 12px;
}

div.stButton > button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #764ba2, #667eea);
    transform: scale(1.02);
    transition: 0.2s ease-in-out;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HERO SECTION
# -------------------------
st.markdown("""
<div class="hero">
<h1>🏠 EstateAI</h1>
<h4>Intelligent Real Estate Price Prediction Platform</h4>
<p>Machine Learning powered property valuation & insights</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("V3.csv")

df = load_data()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Select Prediction Model",
    ["Random Forest (Recommended)", "Linear Regression"]
)

show_data = st.sidebar.checkbox("Show Raw Dataset Preview")

if show_data:
    st.dataframe(df.head())

# -------------------------
# DATA CLEANING (UNCHANGED)
# -------------------------
df["Locality"].fillna(df["Locality"].mode()[0], inplace=True)
df["carpet_area"].fillna(df["carpet_area"].median(), inplace=True)
df["Estimated Value"].fillna(df["Estimated Value"].median(), inplace=True)
df.drop_duplicates(inplace=True)

df = pd.get_dummies(
    df,
    columns=['Locality','Property','Residential','Face'],
    drop_first=True
)

df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in ['Sale Price','Estimated Value','carpet_area']:
    df = remove_outliers_iqr(df, col)

# -------------------------
# MARKET INSIGHTS SECTION
# -------------------------
st.markdown("## 📊 Market Insights")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.boxplot(x=df['Sale Price'], ax=ax1)
    ax1.set_title("Sale Price Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['carpet_area'], df['Sale Price'])
    ax2.set_xlabel("Carpet Area")
    ax2.set_ylabel("Sale Price")
    ax2.set_title("Area vs Sale Price")
    st.pyplot(fig2)

st.markdown("---")

# -------------------------
# MODEL TRAINING (UNCHANGED)
# -------------------------
feature_columns = [
    "property_tax_rate",
    "carpet_area",
    "num_bathrooms",
    "num_rooms",
    "Estimated Value",
    "Year",
    "month"
]

encoded_cols = [col for col in df.columns if col not in feature_columns + ["Sale Price", "day"]]
feature_columns += encoded_cols

X = df[feature_columns]
y = df["Sale Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if "Random Forest" in model_choice:
    model = RandomForestRegressor(random_state=42)
else:
    model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------
# PERFORMANCE METRICS
# -------------------------
st.markdown("## 📈 Model Performance")

rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)

m1, m2, m3 = st.columns(3)

m1.metric("R² Score", round(r2_score(y_test, y_pred), 4))
m2.metric("Mean Absolute Error", round(mean_absolute_error(y_test, y_pred), 2))
m3.metric("RMSE", rmse)

st.markdown("---")

# -------------------------
# PREDICTION SECTION
# -------------------------
st.markdown("## 🧮 Predict Property Price")

colA, colB = st.columns(2)
input_data = {}

with colA:
    input_data["property_tax_rate"] = st.number_input("Property Tax Rate", value=1.0)
    input_data["carpet_area"] = st.number_input("Carpet Area (sq ft)", value=1000.0)
    input_data["num_bathrooms"] = st.number_input("Bathrooms", value=2)
    input_data["num_rooms"] = st.number_input("Rooms", value=3)

with colB:
    input_data["Estimated Value"] = st.number_input("Estimated Market Value", value=200000.0)
    input_data["Year"] = st.number_input("Year Built", value=2020)
    input_data["month"] = st.slider("Sale Month", 1, 12, 1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Price"):

    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#667eea,#764ba2);
        padding:30px;
        border-radius:20px;
        text-align:center;
        color:white;
        font-size:24px;
        font-weight:600;">
        🏠 Predicted Sale Price <br><br>
        ₹ {round(prediction, 2)}
    </div>
    """, unsafe_allow_html=True)