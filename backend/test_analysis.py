"""
Simple test script to verify portfolio analysis works
Run this to test your correlation calculations before connecting to frontend
"""

import yfinance as yf
import pandas as pd
import numpy as np

def test_portfolio_analysis():
    # Test with some popular stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    print(f"Fetching data for: {tickers}")
    print("This may take a moment...\n")
    
    # Download 1 year of data
    data = yf.download(tickers, period='1y', progress=False)
    
    # Debug: print what we got
    print("Data columns:", data.columns)
    print("Data shape:", data.shape)
    
    # Extract Close prices (yfinance 1.2.0 uses 'Close' not 'Adj Close')
    if 'Close' in data.columns:
        stock_data = data['Close']
    elif isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0):
        stock_data = data['Close']
    else:
        # Fallback - just use the data as-is
        stock_data = data
    
    # Calculate correlation matrix
    correlation_matrix = stock_data.corr()
    
    print("=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    print(correlation_matrix.round(3))
    print()
    
    # Calculate returns
    returns = stock_data.pct_change()
    
    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)
    
    print("=" * 60)
    print("ANNUALIZED VOLATILITY")
    print("=" * 60)
    for ticker, vol in volatility.items():
        print(f"{ticker}: {vol:.2%}")
    print()
    
    # Current prices
    print("=" * 60)
    print("CURRENT PRICES")
    print("=" * 60)
    for ticker, price in stock_data.iloc[-1].items():
        print(f"{ticker}: ${price:.2f}")
    print()
    
    # 50-day moving average
    ma_50 = stock_data.rolling(window=50).mean().iloc[-1]
    
    print("=" * 60)
    print("50-DAY MOVING AVERAGE")
    print("=" * 60)
    for ticker, ma in ma_50.items():
        print(f"{ticker}: ${ma:.2f}")
    print()

if __name__ == "__main__":
    test_portfolio_analysis()