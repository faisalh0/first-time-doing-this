import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


st.sidebar.title("🎯 Target")
ticker = st.sidebar.text_input("Enter Stock Ticker", value='AAPL')
period = st.sidebar.selectbox("Select Period", ['1mo', '3mo', '6mo', '1y'], index=2)

# Download data
data = yf.download(ticker, period=period)
if data.empty:
    st.error("Invalid ticker or no data available.")
    st.stop()

# Ensure close_prices is a Series
if isinstance(data['Close'], pd.DataFrame):
    close_prices = data['Close'].iloc[:, 0]  # use the first column if multi-column
else:
    close_prices = data['Close']

returns = close_prices.pct_change().dropna()
# === Indicators Calculation ===
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI_14'] = 100 - (100 / (1 + rs))

# Z-Score Calculation
mean_price = close_prices.mean()
std_price = close_prices.std()
latest_price = close_prices.iloc[-1]
z_score = (latest_price - mean_price) / std_price

# Main Title
st.title(f"📈 Financial Dashboard: {ticker.upper()}")

# Show line chart
st.subheader("Price Chart")
st.line_chart(close_prices)

# Show histogram of returns
st.subheader("Return Distribution")
fig, ax = plt.subplots()
ax.hist(returns, bins=30, color='skyblue', edgecolor='black')
ax.set_title("Histogram of Daily Returns")
st.pyplot(fig)

# Quant metrics
st.subheader("📊 Statistical Summary")
st.write({
    'Mean Return': returns.mean(),
    'Variance': returns.var(),
    'Standard Deviation': returns.std()
})

# 🔍 Z-Score Alert
st.subheader("🚨 Price Mismatch Detection (Z-Score)")
st.write(f"**Z-Score of Latest Price**: {z_score:.2f}")
if abs(z_score) > 2:
    st.error("⚠️ Significant price deviation detected! (Z-score > 2)")
else:
    st.success("✅ Price is within normal range (Z-score ≤ 2)")
# 📉 Interactive Z-Score Time Series Visualization
st.subheader("📊 Interactive Z-Score Over Time")

# Calculate Z-scores using rolling window
z_scores = (close_prices - close_prices.rolling(window=20).mean()) / close_prices.rolling(window=20).std()

# Create Plotly line chart
fig_z = go.Figure()

fig_z.add_trace(go.Scatter(
    x=z_scores.index,
    y=z_scores,
    mode='lines',
    name='Z-Score',
    line=dict(color='blue')
))

# Add horizontal lines for ±2 and 0
fig_z.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Mean", annotation_position="top left")
fig_z.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2 Threshold", annotation_position="top left")
fig_z.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="-2 Threshold", annotation_position="bottom left")

fig_z.update_layout(
    title=f"Z-Score Over Time - {ticker.upper()}",
    xaxis_title="Date",
    yaxis_title="Z-Score",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig_z, use_container_width=True)



# === RSI Chart ===
st.subheader("📉 Relative Strength Index (RSI 14)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], mode='lines', name='RSI'))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig_rsi.update_layout(title='RSI (14)', xaxis_title='Date', yaxis_title='RSI Value')
st.plotly_chart(fig_rsi, use_container_width=True)

# === 🔁 Beta Calculation ===
st.subheader("📐 Beta Calculation (vs S&P 500)")

benchmark = '^GSPC'
benchmark_data = yf.download(benchmark, period=period)['Close']

# Align and compute returns
combined_returns = pd.concat([close_prices, benchmark_data], axis=1).dropna()
combined_returns.columns = ['Stock', 'Benchmark']
combined_returns = combined_returns.pct_change().dropna()

# Calculate covariance and variance
cov = np.cov(combined_returns['Stock'], combined_returns['Benchmark'])[0, 1]
var = np.var(combined_returns['Benchmark'])

beta = cov / var

st.write(f"**Beta of {ticker.upper()} vs S&P 500:** {beta:.4f}")
if beta > 1:
    st.warning("⚠️ The stock is more volatile than the market.")
elif beta < 1:
    st.success("✅ The stock is less volatile than the market.")
else:
    st.info("📊 The stock moves in line with the market.")

# === 📉 Beta Visualization ===
st.subheader("📊 Beta Scatter Plot vs S&P 500")

import plotly.express as px
from sklearn.linear_model import LinearRegression

# Prepare data
x = combined_returns['Benchmark'].values.reshape(-1, 1)
y = combined_returns['Stock'].values

# Fit linear regression to get the beta visually
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Create DataFrame for Plotly
plot_df = pd.DataFrame({
    'Benchmark Returns': x.flatten(),
    'Stock Returns': y,
    'Regression Line': y_pred
})

# Plot
fig_beta = px.scatter(
    plot_df, x='Benchmark Returns', y='Stock Returns',
    title=f"Beta Visualization: {ticker.upper()} vs S&P 500",
    labels={'Benchmark Returns': 'S&P 500 Returns', 'Stock Returns': f'{ticker.upper()} Returns'},
    opacity=0.6
)

# Add regression line
fig_beta.add_traces(go.Scatter(
    x=plot_df['Benchmark Returns'],
    y=plot_df['Regression Line'],
    mode='lines',
    name='Regression Line (Beta)',
    line=dict(color='red')
))

st.plotly_chart(fig_beta, use_container_width=True)


# Correlation/Covariance (optional 2nd ticker)
st.subheader("📌 Compare with Another Ticker")
second_ticker = st.text_input("Second Ticker (Optional)", value='MSFT')

if second_ticker:
    df2 = yf.download(second_ticker, period=period)['Close']
    combined = pd.concat([close_prices, df2], axis=1).dropna()
    combined.columns = [ticker.upper(), second_ticker.upper()]

    st.write("Correlation Matrix:")
    st.write(combined.pct_change().dropna().corr())

    st.write("Covariance Matrix:")
    st.write(combined.pct_change().dropna().cov())
