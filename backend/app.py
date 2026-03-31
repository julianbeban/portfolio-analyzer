from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from models import db, User

load_dotenv()

app = Flask(__name__)
CORS(app,resources={r"/api/*": {"origins": "http://localhost:3000"}})  # Allow requests from Next.js frontend
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)

# Create tables
with app.app_context():
    db.create_all()

# ==================== AUTH ROUTES ====================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validation
        if not data or not data.get('email') or not data.get('password') or not data.get('displayName'):
            return jsonify({'error': 'Missing required fields: email, password, displayName'}), 400
        
        email = data.get('email').strip().lower()
        password = data.get('password')
        display_name = data.get('displayName').strip()
        
        # Check password length
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create new user
        user = User(email=email, display_name=display_name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Create token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict(),
            'access_token': access_token
        }), 201
    
    except Exception as e:
        db.session.rollback()
        print(f"Signup error: {str(e)}")
        return jsonify({'error': 'Failed to create user'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user and return JWT token"""
    try:
        data = request.get_json()
        
        # Validation
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing email or password'}), 400
        
        email = data.get('email').strip().lower()
        password = data.get('password')
        
        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Create token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': access_token
        }), 200
    
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()}), 200
    
    except Exception as e:
        print(f"Get user error: {str(e)}")
        return jsonify({'error': 'Failed to get user'}), 500

# ==================== PORTFOLIO ROUTES ====================

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    """
    Receives a list of stock tickers and returns correlation matrix
    """
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        weights = data.get('weights', None)  

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        if not isinstance(tickers, list):
            return jsonify({'error': 'Tickers must be a list'}), 400
        
        if len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers for correlation analysis'}), 400
        
        if len(tickers) > 20:
            return jsonify({'error': 'Maximum 20 tickers allowed'}), 400
        
        # Clean tickers (uppercase, strip whitespace)
        tickers = [ticker.strip().upper() for ticker in tickers]

        if weights is not None:
            if not isinstance(weights, list):
                return jsonify({'error': 'Weights must be a list'}), 400
            
            if len(weights) != len(tickers):
                return jsonify({'error': 'Number of weights must match number of tickers'}), 400
            
            if any(w < 0 for w in weights):
                return jsonify({'error': 'Weights must be non-negative'}), 400
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight == 0:
                return jsonify({'error': 'Weights cannot all be zero'}), 400
            
            weights = [w / total_weight for w in weights]
        else:
            # Default to equal weights
            weights = [1.0 / len(tickers)] * len(tickers)
        
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
        
        # 200-day moving average (most recent value)
        ma_200 = stock_data.rolling(window=200).mean().iloc[-1]
        
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
                sp500_prices = sp500_raw.iloc[:, 0]
        
        sp500_returns = sp500_prices.pct_change()
        
        # Individual stock betas and Sharpe ratios
        individual_betas = {}
        individual_sharpe_ratios = {}
        
        for ticker in stock_data.columns:
            # Align stock returns with S&P 500 returns
            aligned = pd.DataFrame({
                'stock': returns[ticker],
                'sp500': sp500_returns
            }).dropna()
            
            # Beta calculation
            covariance = aligned['stock'].cov(aligned['sp500'])
            sp500_variance = aligned['sp500'].var()
            beta = covariance / sp500_variance if sp500_variance != 0 else 0
            individual_betas[ticker] = round(beta, 3)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            stock_return = aligned['stock'].mean() * 252  # Annualized
            stock_volatility = aligned['stock'].std() * np.sqrt(252)
            sharpe = stock_return / stock_volatility if stock_volatility != 0 else 0
            individual_sharpe_ratios[ticker] = round(sharpe, 3)
        
        # RSI (Relative Strength Index) for each stock
        def calculate_rsi(prices, period=14):
            """Calculate RSI for a price series"""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]  # Most recent value
        
        rsi_values = {}
        for ticker in stock_data.columns:
            rsi = calculate_rsi(stock_data[ticker])
            rsi_values[ticker] = round(rsi, 2) if not pd.isna(rsi) else None
        
        # PORTFOLIO METRICS (Equal Weighted)
        # Calculate equal-weighted portfolio returns
        portfolio_returns = returns.mean(axis=1)  # Equal weight = average across stocks

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
            'ma_200': ma_200.round(2).to_dict(),  
            'individual_betas': individual_betas,  
            'individual_sharpe_ratios': individual_sharpe_ratios,  
            'rsi': rsi_values, 
            'weights': weights,
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