import yfinance as yf
import pandas as pd
import numpy as np

# Choose stock ticker and date range
ticker = 'AAPL'
data = yf.download(ticker, period='6mo')

# Use only the 'Close' prices
close_prices = data['Close']

# Calculate daily returns
returns = close_prices.pct_change().dropna()

# Calculate basic stats
mean_return = returns.mean()
variance = returns.var()
std_dev = returns.std()

# Create a summary DataFrame
summary = pd.DataFrame({
    'Metric': ['Mean Return', 'Variance', 'Standard Deviation'],
    'Value': [mean_return, variance, std_dev]
})

# Print everything
print("📈 Basic Statistics for", ticker)
print(summary)
