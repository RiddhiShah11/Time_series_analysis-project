#  Tcs Multi-Page Dashboard 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import base64, os, io, datetime, warnings
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.arima.model import ARIMAResults
from prophet.serialize import model_from_json
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore")


try:
    import ta
except Exception:
    ta = None

# PAGE CONFIG

st.set_page_config(layout="wide", page_title="TCS Stock Forecasting Dashboard")


def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BG = get_base64("b1.jpg") if os.path.exists("b1.jpg") else ""

# CSS STYLING

page_css = f"""
<style>
/* Full-page background */
[data-testid="stAppViewContainer"] {{
  background-image: url("data:image/jpeg;base64,{BG}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

/* Sidebar styling */
[data-testid="stSidebar"] > div:first-child {{
  background-color: #000;
  color: white;
}}

/* Gradient banner */
.banner {{
  background: linear-gradient(90deg, #0a4275, #153c7a);
  padding: 35px;
  border-radius: 8px;
  color: white;
  text-align: center;
  box-shadow: 0 4px 15px rgba(0,0,0,0.4);
  font-family: 'Segoe UI', sans-serif;
}}

/* Metric cards */
.metric-card {{
  background: rgba(20, 20, 20, 0.8);  /* darker semi-transparent background */
  border-radius: 12px;
  padding: 18px;
  text-align: center;
  color: #ffffff;  /* white text */
  font-weight: 600;
  box-shadow: 0 6px 20px rgba(0,0,0,0.4);
  backdrop-filter: blur(6px);  /* glass effect */
  border: 1px solid rgba(255, 255, 255, 0.2);
}}

.metric-card h2 {{
  color: #00bfff;  /* bright blue for numbers */
  font-size: 1.8em;
}}

.metric-card h4 {{
  color: #f1f1f1;  /* light grey for labels */
  font-weight: 500;
}}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# BANNER 

st.markdown(
    '<div class="banner"><h1>📈 TCS Stock Forecasting Dashboard</h1>'
    '<p>Comprehensive Multi-Page Time Series Analysis</p></div>',
    unsafe_allow_html=True
)
st.markdown("---")

# LOAD MODELS SAFELY

@st.cache_resource
def load_models_safe():
    arima = sarima = lstm = prophet_model = None
    errors = []
    try:
        with open("arima_model.pkl", "rb") as f:
            arima = pickle.load(f)
    except Exception as e:
        errors.append(f"ARIMA load error: {e}")
    try:
        with open("sarima_model.pkl", "rb") as f:
            sarima = pickle.load(f)
    except Exception as e:
        errors.append(f"SARIMA load error: {e}")
    try:
        lstm = load_model("lstm_model.h5", compile=False)
    except Exception as e:
        errors.append(f"LSTM load error: {e}")
    try:
        with open("prophet_model.json", "r") as f:
            prophet_model = model_from_json(f.read())
    except Exception as e:
        errors.append(f"Prophet load error: {e}")
    return arima, sarima, lstm, prophet_model, errors

arima_model, sarima_model, lstm_model, prophet_model, model_load_errors = load_models_safe()


# READ LOCAL DATA

def read_data():
    if os.path.exists("tcs_stock.csv"):
        df = pd.read_csv("tcs_stock.csv")
    elif os.path.exists("tcs_stock_cleaned.csv"):
        df = pd.read_csv("tcs_stock_cleaned.csv")
    else:
        st.error("No CSV found. Please keep tcs_stock.csv in the folder.")
        st.stop()
    df["Date"] = pd.to_datetime(df["Date"])
    if "Close" not in df.columns:
        for alt in ["Adj Close", "Adj_Close", "ClosePrice"]:
            if alt in df.columns:
                df["Close"] = df[alt]
                break
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = read_data()

# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Dashboard Page",
    [
        "Overview KPIs", "Price Explorer & Candlesticks", "Forecast & Uncertainty",
        "Sentiment & News Impact", "Volatility & Risk", "Indicators Dashboard",
        "Correlations & Market Structure", "Feature Importance & Explainability",
        "Strategy Backtest & Performance", "Interactive Explorer"
    ]
)

# FORECAST HELPERS

def safe_to_numpy(x):
    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        return x.flatten()
    if hasattr(x, "to_numpy"):
        return x.to_numpy().flatten()
    try:
        return np.array(list(x)).flatten()
    except Exception:
        return np.array([])

def arima_forecast(model, steps):
    try:
        fc = model.forecast(steps=steps)
        return safe_to_numpy(fc)
    except Exception:
        try:
            fc = model.get_forecast(steps=steps)
            return safe_to_numpy(fc.predicted_mean)
        except Exception:
            return np.array([])

def sarima_forecast(model, steps):
    try:
        fc = model.forecast(steps=steps)
        return safe_to_numpy(fc)
    except Exception:
        return np.array([])

def prophet_forecast(model, steps):
    try:
        future = model.make_future_dataframe(periods=steps)
        fc = model.predict(future)
        return safe_to_numpy(fc.tail(steps)["yhat"]), fc
    except Exception:
        return np.array([]), None

def lstm_forecast(model, history_close, steps):
    try:
        scaler = MinMaxScaler((0, 1))
        arr = np.array(history_close).reshape(-1, 1).astype("float32")
        scaled = scaler.fit_transform(arr)
        last60 = scaled[-60:] if len(scaled) >= 60 else np.tile(scaled[-1], (60, 1))
        preds = []
        for _ in range(steps):
            p = model.predict(last60.reshape(1, 60, 1), verbose=0)
            preds.append(p[0, 0])
            last60 = np.append(last60[1:], p, axis=0)
        inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return inv
    except Exception:
        return np.array([])


# 1️ OVERVIEW / KPIs

if page == "Overview KPIs":
    st.header("Overview & Key Performance Indicators")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    change = latest["Close"] - prev["Close"]
    pct_change = (change / prev["Close"]) * 100 if prev["Close"] else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h4>Latest Close</h4><h2>₹ {latest["Close"]:,.2f}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h4>Change</h4><h2>{change:,.2f}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h4>% Change</h4><h2>{pct_change:.2f}%</h2></div>', unsafe_allow_html=True)
    logret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    vol = logret.std() * np.sqrt(252)
    c4.markdown(f'<div class="metric-card"><h4>Annualized Vol</h4><h2>{vol:.4f}</h2></div>', unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)

# 2️ PRICE EXPLORER

elif page == "Price Explorer & Candlesticks":
    st.header("Price Explorer & Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Close"].shift(1).fillna(df["Close"]),
        high=df["Close"] * 1.01,
        low=df["Close"] * 0.99,
        close=df["Close"])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)


# 3️ FORECAST & UNCERTAINTY

elif page == "Forecast & Uncertainty":
    st.header("Forecast & Uncertainty")
    model_sel = st.selectbox("Choose Model", ["ARIMA", "SARIMA", "LSTM", "Prophet"])
    horizon = st.slider("Days to Forecast", 1, 60, 15)
    fc = np.array([])

    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Close"], label="Historical", color="#1f77b4")

    if model_sel == "ARIMA" and arima_model:
        fc = arima_forecast(arima_model, horizon)
    elif model_sel == "SARIMA" and sarima_model:
        fc = sarima_forecast(sarima_model, horizon)
    elif model_sel == "LSTM" and lstm_model:
        fc = lstm_forecast(lstm_model, df["Close"].values, horizon)
    elif model_sel == "Prophet" and prophet_model:
        fc, _ = prophet_forecast(prophet_model, horizon)

    if len(fc) > 0:
        future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=horizon + 1)[1:]
        plt.plot(future_dates, fc, "--", color="red", label="Forecast")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Forecast not available for selected model.")


# 4️ SENTIMENT & NEWS IMPACT (PLACEHOLDER)

elif page == "Sentiment & News Impact":
    st.header("Sentiment & News Impact (Placeholder)")
    st.markdown("Simulated sentiment index compared with stock price trend.")
    rng = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    sentiment = np.sin(np.linspace(0, 10, len(rng))) + np.random.normal(0, 0.2, len(rng))
    demo = pd.DataFrame({"Date": rng, "Sentiment": sentiment})
    merged = pd.merge(df, demo, on="Date", how="inner")
    st.line_chart(merged.set_index("Date")[["Close", "Sentiment"]])


# 5️ VOLATILITY & RISK

elif page == "Volatility & Risk":
    st.header("Volatility & Risk")
    returns = df["Close"].pct_change().dropna()
    vol = returns.rolling(21).std() * np.sqrt(252)
    st.line_chart(pd.DataFrame({"Volatility": vol}))
    var = -np.percentile(returns, 5)
    st.write(f"Historical 95% VaR ≈ {var:.4%}")


# 6️ INDICATORS

elif page == "Indicators Dashboard":
    st.header("Indicators Dashboard")
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI14"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() /
                                   (-df["Close"].diff().clip(upper=0).rolling(14).mean())))
    st.line_chart(df.set_index("Date")[["Close", "SMA20", "EMA50", "RSI14"]].dropna())

# 7️ CORRELATIONS

elif page == "Correlations & Market Structure":
    st.header("Correlations & Market Structure")
    from pandas.plotting import autocorrelation_plot
    fig = plt.figure(figsize=(8, 3))
    autocorrelation_plot(df["Close"].dropna())
    st.pyplot(fig)

# 8️ FEATURE IMPORTANCE (PLACEHOLDER)

elif page == "Feature Importance & Explainability":
    st.header("Feature Importance & Explainability (Placeholder)")
    st.info("This page demonstrates explainability concept (e.g., SHAP) for advanced ML models.")
    st.markdown("- For Prophet/LSTM, SHAP or feature importances can be shown if features exist.")
    st.markdown("- Here we simulate importance for demonstration.")
    np.random.seed(0)
    feats = ["Lag1", "Lag7", "MA20", "Volume", "Volatility"]
    vals = np.random.rand(len(feats))
    fi = pd.DataFrame({"Feature": feats, "Importance": vals}).sort_values("Importance", ascending=False)
    st.bar_chart(fi.set_index("Feature"))

# 9️ STRATEGY BACKTEST

elif page == "Strategy Backtest & Performance":
    st.header("Strategy Backtest & Performance")
    fast, slow = 10, 50
    df["fast"] = df["Close"].rolling(fast).mean()
    df["slow"] = df["Close"].rolling(slow).mean()
    df["signal"] = np.where(df["fast"] > df["slow"], 1, 0)
    df["position"] = df["signal"].diff()
    capital, shares = 100000, 0
    portfolio, cash = [], capital
    for _, row in df.iterrows():
        price = row["Close"]
        if row["position"] == 1:
            shares = cash // price; cash -= shares * price
        elif row["position"] == -1:
            cash += shares * price; shares = 0
        portfolio.append(cash + shares * price)
    df["portfolio"] = portfolio
    st.line_chart(df.set_index("Date")[["portfolio", "Close"]])
    st.write(f"Final Portfolio Value: ₹ {df['portfolio'].iloc[-1]:,.2f}")


# 10 INTERACTIVE EXPLORER

elif page == "Interactive Explorer":
    st.header("Interactive Explorer")
    st.markdown("Interactively select model and forecast horizon.")
    model_sel = st.selectbox("Select Model", ["ARIMA", "SARIMA", "LSTM", "Prophet"])
    horizon = st.slider("Forecast Days", 1, 60, 15)
    if st.button("Generate Forecast"):
        fc = np.array([])
        if model_sel == "ARIMA" and arima_model:
            fc = arima_forecast(arima_model, horizon)
        elif model_sel == "SARIMA" and sarima_model:
            fc = sarima_forecast(sarima_model, horizon)
        elif model_sel == "LSTM" and lstm_model:
            fc = lstm_forecast(lstm_model, df["Close"].values, horizon)
        elif model_sel == "Prophet" and prophet_model:
            fc, _ = prophet_forecast(prophet_model, horizon)
        if len(fc) > 0:
            future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=horizon + 1)[1:]
            rez = pd.DataFrame({"Date": future_dates, "Forecast": fc})
            st.dataframe(rez)
            st.line_chart(rez.set_index("Date"))
        else:
            st.warning("Forecast not available.")
