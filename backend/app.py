from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from Next.js frontend

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    """
    Receives a list of stock tickers and returns correlation matrix
    """
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Download 1 year of historical data
        print(f"Fetching data for: {tickers}")
        raw_data = yf.download(tickers, period='1y', progress=False)
        
        # Debug: print data structure
        print(f"Data columns: {raw_data.columns}")
        print(f"Is MultiIndex: {isinstance(raw_data.columns, pd.MultiIndex)}")
        
        # Handle different yfinance response formats
        stock_data = None
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multiple tickers - try Close first, then Adj Close
            if 'Close' in raw_data.columns.get_level_values(0):
                stock_data = raw_data['Close']
                print("Using 'Close' from MultiIndex")
            elif 'Adj Close' in raw_data.columns.get_level_values(0):
                stock_data = raw_data['Adj Close']
                print("Using 'Adj Close' from MultiIndex")
        else:
            # Single level columns - try Close first, then Adj Close
            if 'Close' in raw_data.columns:
                stock_data = raw_data[['Close']].rename(columns={'Close': tickers[0]}) if len(tickers) == 1 else raw_data['Close']
                print("Using 'Close' from single level")
            elif 'Adj Close' in raw_data.columns:
                stock_data = raw_data[['Adj Close']].rename(columns={'Adj Close': tickers[0]}) if len(tickers) == 1 else raw_data['Adj Close']
                print("Using 'Adj Close' from single level")
        
        if stock_data is None or stock_data.empty:
            return jsonify({'error': 'Could not extract price data from yfinance response', 'columns': str(raw_data.columns)}), 500
        
        # Handle single ticker (returns Series instead of DataFrame)
        if len(tickers) == 1:
            return jsonify({
                'message': 'Need at least 2 tickers for correlation analysis',
                'tickers': tickers
            }), 400
        
        # Calculate correlation matrix
        correlation_matrix = stock_data.corr()
        
        # Calculate additional metrics
        returns = stock_data.pct_change()
        
        # Annualized volatility (individual stocks)
        volatility = returns.std() * np.sqrt(252)
        
        # 50-day moving average (most recent value)
        ma_50 = stock_data.rolling(window=50).mean().iloc[-1]
        
        # PORTFOLIO METRICS (Equal Weighted)
        # Calculate equal-weighted portfolio returns
        portfolio_returns = returns.mean(axis=1)  # Equal weight = average across stocks
        
        # Fetch S&P 500 data for beta calculation
        sp500_raw = yf.download('^GSPC', period='1y', progress=False)
        
        # Extract S&P 500 prices
        if isinstance(sp500_raw.columns, pd.MultiIndex):
            if 'Close' in sp500_raw.columns.get_level_values(0):
                sp500_prices = sp500_raw['Close'].squeeze()
            elif 'Adj Close' in sp500_raw.columns.get_level_values(0):
                sp500_prices = sp500_raw['Adj Close'].squeeze()
        else:
            if 'Close' in sp500_raw.columns:
                sp500_prices = sp500_raw['Close']
            elif 'Adj Close' in sp500_raw.columns:
                sp500_prices = sp500_raw['Adj Close']
            else:
                sp500_prices = sp500_raw.iloc[:, 0]  # First column
        
        sp500_returns = sp500_prices.pct_change()
        
        # Align dates between portfolio and S&P 500
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'sp500': sp500_returns
        }).dropna()
        
        # Portfolio Sharpe Ratio (assuming 0% risk-free rate)
        portfolio_return = aligned_data['portfolio'].mean() * 252  # Annualized
        portfolio_volatility = aligned_data['portfolio'].std() * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
        
        # Portfolio Beta (relative to S&P 500)
        covariance = aligned_data['portfolio'].cov(aligned_data['sp500'])
        sp500_variance = aligned_data['sp500'].var()
        beta = covariance / sp500_variance if sp500_variance != 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + aligned_data['portfolio']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Prepare response
        response = {
            'correlation': correlation_matrix.round(3).to_dict(),
            'volatility': volatility.round(4).to_dict(),
            'current_prices': stock_data.iloc[-1].round(2).to_dict(),
            'ma_50': ma_50.round(2).to_dict(),
            'portfolio_metrics': {
                'sharpe_ratio': round(sharpe_ratio, 3),
                'beta': round(beta, 3),
                'max_drawdown': round(max_drawdown, 4),
                'annualized_return': round(portfolio_return, 4),
                'annualized_volatility': round(portfolio_volatility, 4)
            },
            'tickers': tickers
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Portfolio Analyzer API is running'}), 200

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Returns portfolio overview stats"""
    try:
        portfolio_data = {
            'totalValue': 156843.50,
            'todayGain': 3521.20,
            'todayGainPercent': 2.3,
            'ytdReturn': 18.5,
            'cashAvailable': 25400.00,
            'buyingPower': 50800.00
        }
        return jsonify(portfolio_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holdings', methods=['GET'])
def get_holdings():
    """Returns user's current stock holdings"""
    try:
        holdings = [
            {'symbol': 'AAPL', 'shares': 50, 'avgCost': 185.30, 'current': 234.50},
            {'symbol': 'MSFT', 'shares': 30, 'avgCost': 405.20, 'current': 421.30},
            {'symbol': 'VOO', 'shares': 25, 'avgCost': 418.50, 'current': 486.80},
            {'symbol': 'BRK.B', 'shares': 40, 'avgCost': 380.40, 'current': 412.60}
        ]
        
        response = []
        for holding in holdings:
            gain_amount = (holding['current'] - holding['avgCost']) * holding['shares']
            gain_percent = ((holding['current'] - holding['avgCost']) / holding['avgCost']) * 100
            total_value = holding['current'] * holding['shares']
            
            response.append({
                'symbol': holding['symbol'],
                'shares': holding['shares'],
                'avgCost': round(holding['avgCost'], 2),
                'current': round(holding['current'], 2),
                'gainAmount': round(gain_amount, 2),
                'gainPercent': round(gain_percent, 1),
                'totalValue': round(total_value, 2)
            })
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """Returns market watchlist with live data"""
    try:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corp.',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corp.'
        }
        
        # Download all data at once (more efficient than individual calls)
        data = yf.download(tickers, period='5d', progress=False)
        
        watchlist = []
        for ticker in tickers:
            try:
                # Get current price from today's close
                current_price = data['Close'][ticker].iloc[-1] if len(tickers) > 1 else data['Close'].iloc[-1]
                
                # Get price from 1 day ago to calculate change
                prev_price = data['Close'][ticker].iloc[-2] if len(tickers) > 1 else data['Close'].iloc[-2]
                
                # Calculate percentage change
                change = ((current_price - prev_price) / prev_price) * 100

                # Get volume (formatted)
                volume = data['Volume'][ticker].iloc[-1] if len(tickers) > 1 else data['Volume'].iloc[-1]
                volume_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"
                
                watchlist.append({
                    'symbol': ticker,
                    'name': names.get(ticker, ticker),
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'volume': volume_str
                })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        return jsonify(watchlist), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, port=5001, host='127.0.0.1')