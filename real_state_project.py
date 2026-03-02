import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EstateAI",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------
# PREMIUM UI STYLING
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.hero-box {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

.section-box {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    margin-bottom: 30px;
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
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
st.markdown("""
<div class="hero-box">
<h1>🏠 EstateAI</h1>
<h4>Intelligent Real Estate Price Prediction Platform</h4>
<p>Machine Learning powered property valuation & advanced analytics</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("V3.csv")

df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Select Prediction Model",
    ["Random Forest (Recommended)", "Linear Regression"]
)

show_data = st.sidebar.checkbox("Show Raw Dataset Preview")

if show_data:
    st.dataframe(df.head())

# -------------------------------------------------
# DATA CLEANING (UNCHANGED)
# -------------------------------------------------
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

# -------------------------------------------------
# REMOVE OUTLIERS
# -------------------------------------------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in ['Sale Price','Estimated Value', 'carpet_area']:
    df = remove_outliers_iqr(df,col)

# -------------------------------------------------
# MARKET INSIGHTS
# -------------------------------------------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
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

# Heatmap
st.markdown("### 🔥 Feature Correlation Heatmap ")
num_cols = ['Sale Price', 'Estimated Value', 'carpet_area',
            'num_bathrooms', 'num_rooms', 'property_tax_rate']
corr_matrix = df[num_cols].corr()

fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax_corr,
    cbar_kws={"shrink": 0.8}
)
ax_corr.set_title("Correlation Between Numerical Features")
st.pyplot(fig_corr)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MODEL TRAINING 
# -------------------------------------------------
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

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("## 📈 Model Performance")

rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)

m1, m2, m3 = st.columns(3)
m1.metric("R² Score", round(r2_score(y_test, y_pred), 4))
m2.metric("Mean Absolute Error", round(mean_absolute_error(y_test, y_pred), 2))
m3.metric("RMSE", rmse)

# Predicted vs Actual data 
st.markdown("### 📈 Predicted vs Actual Sale Price")
fig_pred, ax_pred = plt.subplots(figsize=(7, 4))
ax_pred.scatter(y_test, y_pred, alpha=0.5)
ax_pred.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
ax_pred.set_xlabel("Actual Price")
ax_pred.set_ylabel("Predicted Price")
ax_pred.set_title("Predicted vs Actual Sale Price")
st.pyplot(fig_pred)

st.markdown('</div>', unsafe_allow_html= True)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
if "Random Forest" in model_choice:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("## 🔍 Feature Importance")

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top10 = importances.nlargest(10).sort_values()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    top10.plot(kind='barh', ax=ax3)
    ax3.set_title("Top 10 Most Influential Features")
    st.pyplot(fig3)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION SECTION

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("##  🧮 Predict Property Price ")

colA, colB = st.columns(2)
input_data = {}

with colA:
    input_data["property_tax_rate"] = st.number_input("Property Tax Rates", value=1.0)
    input_data["carpet_area"] = st.number_input("Carpet Area (sq ft.)", value=1000.0)
    input_data["num_bathrooms"] = st.number_input("Bathrooms", value=2)
    input_data["num_rooms"] = st.number_input("Rooms", value=3)

with colB:
    input_data["Estimated Value"] = st.number_input("Estimated Market Value", value=200000.0)
    input_data["Year"] = st.number_input("Year Built", value=2020)
    input_data["month"] = st.slider("Sale Month", 1, 12, 1)

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
        font-size:26px;
        font-weight:600;">
        🏠 Predicted Sale Price <br><br>
        ₹ {round(prediction, 2)}
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)