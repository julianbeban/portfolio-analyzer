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
        
        # Extract Close prices
        if 'Close' in raw_data.columns:
            stock_data = raw_data['Close']
        elif isinstance(raw_data.columns, pd.MultiIndex) and 'Close' in raw_data.columns.get_level_values(0):
            stock_data = raw_data['Close']
        else:
            stock_data = raw_data
        
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
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # 50-day moving average (most recent value)
        ma_50 = stock_data.rolling(window=50).mean().iloc[-1]
        
        # Prepare response
        response = {
            'correlation': correlation_matrix.round(3).to_dict(),
            'volatility': volatility.round(4).to_dict(),
            'current_prices': stock_data.iloc[-1].round(2).to_dict(),
            'ma_50': ma_50.round(2).to_dict(),
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

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5001, host='127.0.0.1')
