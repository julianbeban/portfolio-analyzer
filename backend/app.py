from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from models import db, User, Transaction, Holding
import math
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
from datetime import datetime, timedelta

_watchlist_cache = {
    'data': None,
    'expires': None
}

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

load_dotenv()

CORS(app,resources={r"/api/*": {"origins": "http://localhost:3000"}})  # Allow requests from Next.js 

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///portfolio.db')

# Initialize extensions
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()
    
def _validate_ticker(ticker: str) -> tuple[bool, str]:
    """
    Validate a stock ticker before hitting yfinance.
    Returns (is_valid, error_message)
    """
    # Format check — tickers are 1-5 uppercase letters, 
    # some have dots (BRK.B) or hyphens (BF-B)
    if not ticker:
        return False, 'Ticker is required'
    
    if len(ticker) > 10:
        return False, f'Invalid ticker: {ticker}'
    
    if not re.match(r'^[A-Z0-9][A-Z0-9.\-]{0,8}[A-Z0-9]$|^[A-Z]$', ticker):
        return False, f'Invalid ticker format: {ticker}'
    
    # Verify ticker actually exists via yfinance
    try:
        tick = yf.Ticker(ticker)
        info = tick.info
        # yfinance returns minimal dict for invalid tickers
        if not info or info.get('quoteType') is None:
            return False, f'Ticker not found: {ticker}'
        # Reject if it's not a tradeable equity/ETF
        valid_types = {'EQUITY', 'ETF', 'MUTUALFUND'}
        quote_type = info.get('quoteType', '').upper()
        if quote_type not in valid_types:
            return False, f'{ticker} is not a stock or ETF (type: {quote_type})'
    except Exception as e:
        return False, f'Could not verify ticker {ticker}: {str(e)}'
    
    return True, ''

def _extract_price(data: pd.DataFrame, ticker: str, tickers: list, row_idx: int = -1) -> float | None:
    """
    Safely extract a single price from a yfinance DataFrame.
    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex) responses.
    Returns None if the value is missing or NaN.
    """
    try:
        if len(tickers) == 1:
            # Single ticker — data['Close'] is a Series
            price = data['Close'].iloc[row_idx]
        else:
            # Multiple tickers — data['Close'] is a DataFrame, select by ticker
            price = data['Close'][ticker].iloc[row_idx]

        # Covers numpy NaN, pandas NA, None
        if price is None or pd.isna(price):
            return None

        return float(price)

    except (IndexError, KeyError, TypeError):
        return None
    
def _extract_volume(data: pd.DataFrame, ticker: str, tickers: list, row_idx: int = -1) -> float:
    """
    Safely extract volume from a yfinance DataFrame.
    Returns 0 if missing.
    """
    try:
        if len(tickers) == 1:
            volume = data['Volume'].iloc[row_idx]
        else:
            volume = data['Volume'][ticker].iloc[row_idx]

        if volume is None or pd.isna(volume):
            return 0.0

        return float(volume)

    except (IndexError, KeyError, TypeError):
        return 0.0
    
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
        
        if len(email) > 254:
            return jsonify({'error': 'Invalid email'}), 400
        if len(display_name) > 100:
            return jsonify({'error': 'Display name too long'}), 400
        if len(password) > 128:
            return jsonify({'error': 'Password too long'}), 400
        
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
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        print(f"Signup error: {str(e)}")
        return jsonify({'error': 'Failed to create user'}), 500

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
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
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict()
        }), 200
    
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

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
    try:
        user_id = request.headers.get('X-User-ID')
        
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_id = int(user_id)
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        holdings = Holding.query.filter_by(user_id=user_id).all()
        if not holdings:
            return jsonify({
                'totalValue': 0, 'todayGain': 0, 'todayGainPercent': 0,
                'portfolioReturn': 0, 'ytdReturn': 0, 'buyingPower': 0
            }), 200
        
        # Batch fetch all holdings data at once
        tickers = [h.ticker for h in holdings]
        ticker_data = {}
        
        try:
            # Fetch 2 days of data to get current and previous close
            data = yf.download(tickers, period='2d', progress=False)
            
            # Handle single ticker case (returns Series instead of DataFrame)
            for ticker in tickers:
                ticker_data[ticker] = {
                    'current': _extract_price(data, ticker, tickers, -1),
                    'previous': _extract_price(data, ticker, tickers, -2)
                }
        except Exception as e:
            print(f"Batch download failed: {e}, falling back to individual calls")
            # Fallback: use individual ticker fetch
            for holding in holdings:
                try:
                    tick = yf.Ticker(holding.ticker)
                    current = tick.info.get('currentPrice', holding.average_cost)
                    previous = tick.info.get('previousClose', current)
                    ticker_data[holding.ticker] = {'current': current, 'previous': previous}
                except:
                    ticker_data[holding.ticker] = {'current': holding.average_cost, 'previous': holding.average_cost}
        
        total_value = 0
        total_investment = 0
        total_gain = 0
        total_previous_value = 0
        
        for holding in holdings:
            try:
                prices = ticker_data.get(holding.ticker, {})
                current_price = prices.get('current', holding.average_cost)
                yesterday_price = prices.get('previous', current_price)
                
                # Handle NaN values
                if current_price is None or (isinstance(current_price, float) and math.isnan(current_price)):
                    current_price = holding.average_cost
                if yesterday_price is None or (isinstance(yesterday_price, float) and math.isnan(yesterday_price)):
                    yesterday_price = current_price
                
                total_investment += holding.shares * holding.average_cost
                total_value += holding.shares * current_price
                total_previous_value += holding.shares * yesterday_price
                total_gain += (current_price - yesterday_price) * holding.shares
            except Exception as e:
                print(f"Error processing {holding.ticker}: {e}")
                # Use average cost as fallback for this holding
                total_investment += holding.shares * holding.average_cost
                total_value += holding.shares * holding.average_cost
                total_previous_value += holding.shares * holding.average_cost
                continue
        
        lifetime_gain = total_value - total_investment
        portfolio_return = ((lifetime_gain) / total_investment * 100) if total_investment > 0 else 0
        today_gain_percent = ((total_value - total_previous_value) / total_previous_value * 100) if total_previous_value > 0 else 0
        
        return jsonify({
            'totalValue': float(round(total_value, 2)),
            'todayGain': float(round(total_gain, 2)),
            'todayGainPercent': float(round(today_gain_percent, 2)),
            'portfolioReturn': float(round(lifetime_gain, 2)),
            'portfolioReturnPercent': float(round(portfolio_return, 2)),
            'ytdReturn': 18.5,
            'cashAvailable': 0.0,
            'buyingPower': 0
        }), 200
    except Exception as e:
        print(f"Portfolio error: {e}")
        return jsonify({'error': 'Failed to fetch portfolio data'}), 500

@app.route('/api/holdings', methods=['GET'])
def get_holdings():
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_id = int(user_id)  # Convert to int
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_holdings = Holding.query.filter_by(user_id=user_id).all()
        if not user_holdings:
            return jsonify([]), 200
        
        # Batch fetch all holdings data at once
        tickers = [h.ticker for h in user_holdings]
        ticker_prices = {}
        
        try:
            # Fetch current prices for all tickers at once
            data = yf.download(tickers, period='1d', progress=False)
            
            # Handle single ticker case
            for ticker in tickers:
                ticker_prices[ticker] = _extract_price(data, ticker, tickers)
        except Exception as e:
            print(f"Batch download failed: {e}, falling back to individual calls")
            # Fallback: use individual ticker fetch
            for ticker in tickers:
                try:
                    tick = yf.Ticker(ticker)
                    price = tick.info.get('currentPrice')
                    ticker_prices[ticker] = price if price is not None else None
                except:
                    ticker_prices[ticker] = None
        
        response = []
        for holding in user_holdings:
            try:
                current_price = ticker_prices.get(holding.ticker)
                
                # Handle None or NaN values - use average cost as fallback
                if current_price is None:
                    current_price = holding.average_cost
                
                gain_amount = (current_price - holding.average_cost) * holding.shares
                gain_percent = ((current_price - holding.average_cost) / holding.average_cost) * 100 if holding.average_cost > 0 else 0
                total_value = current_price * holding.shares
                
                response.append({
                    'symbol': holding.ticker,
                    'shares': holding.shares,
                    'avgCost': round(holding.average_cost, 2),
                    'current': round(current_price, 2),
                    'gainAmount': round(gain_amount, 2),
                    'gainPercent': round(gain_percent, 1),
                    'totalValue': round(total_value, 2)
                })
            except Exception as e:
                print(f"Error processing {holding.ticker}: {e}")
                # Use average cost as fallback
                response.append({
                    'symbol': holding.ticker,
                    'shares': holding.shares,
                    'avgCost': round(holding.average_cost, 2),
                    'current': round(holding.average_cost, 2),
                    'gainAmount': 0,
                    'gainPercent': 0,
                    'totalValue': round(holding.average_cost * holding.shares, 2)
                })
                continue
        
        return jsonify(response), 200
    except Exception as e:
        print(f"Holdings error: {str(e)}")
        return jsonify({'error': str(e)}), 500

_watchlist_cache = {
    'data': None,
    'expires': None
}

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    global _watchlist_cache

    # Serve from cache if still valid (5 minute TTL)
    if _watchlist_cache['data'] and datetime.now() < _watchlist_cache['expires']:
        return jsonify(_watchlist_cache['data']), 200

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    names = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corp.',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corp.'
    }

    try:
        watchlist = []

        try:
            data = yf.download(tickers, period='5d', progress=False)
            data_available = True
        except Exception as e:
            print(f"Failed to download data: {e}")
            data_available = False

        for ticker in tickers:
            try:
                current_price = 0
                prev_price = 0
                change = 0
                volume_str = "0K"

                if data_available:
                    try:
                        current_price = _extract_price(data, ticker, tickers, -1)
                        prev_price = _extract_price(data, ticker, tickers, -2)
                        volume = _extract_volume(data, ticker, tickers)

                        if current_price is None or prev_price is None:
                            raise Exception("NaN values in batch data")

                        change = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                        volume_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"

                    except Exception as batch_err:
                        print(f"Batch data failed for {ticker}: {batch_err}, using individual fetch")
                        try:
                            ticker_obj = yf.Ticker(ticker)
                            current_price = ticker_obj.info.get('currentPrice')
                            prev_price = ticker_obj.info.get('previousClose')
                            volume = ticker_obj.info.get('volume', 0)

                            if current_price is None:
                                current_price = prev_price if prev_price else 0
                            if prev_price is None:
                                prev_price = current_price if current_price else 0

                            change = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                            volume_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"
                        except Exception as fallback_err:
                            print(f"Individual fetch failed for {ticker}: {fallback_err}")
                            current_price = 0
                            change = 0
                            volume_str = "0K"
                else:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        current_price = ticker_obj.info.get('currentPrice')
                        prev_price = ticker_obj.info.get('previousClose')
                        volume = ticker_obj.info.get('volume', 0)

                        if current_price is None:
                            current_price = prev_price if prev_price else 0
                        if prev_price is None:
                            prev_price = current_price if current_price else 0

                        change = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                        volume_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"
                    except Exception as fallback_err:
                        print(f"Individual fetch failed for {ticker}: {fallback_err}")
                        current_price = 0
                        change = 0
                        volume_str = "0K"

                watchlist.append({
                    'symbol': ticker,
                    'name': names.get(ticker, ticker),
                    'price': round(current_price, 2) if current_price else 0,
                    'change': round(change, 2),
                    'volume': volume_str
                })

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                watchlist.append({
                    'symbol': ticker,
                    'name': names.get(ticker, ticker),
                    'price': 0,
                    'change': 0,
                    'volume': "0K"
                })

        # Cache and return on success
        _watchlist_cache['data'] = watchlist
        _watchlist_cache['expires'] = datetime.now() + timedelta(minutes=5)
        return jsonify(watchlist), 200

    except Exception as e:
        print(f"Watchlist error: {str(e)}")
        return jsonify({'error': 'Failed to fetch watchlist data'}), 500

# ==================== PORTFOLIO MANAGEMENT ====================

@app.route('/api/import/csv', methods=['POST'])
def import_csv():
    """Import transactions from CSV file"""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_id = int(user_id)
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get column mapping from request (optional)
        column_mapping = request.form.get('columnMapping')
        if column_mapping:
            import json
            try:
                column_mapping = json.loads(column_mapping)
            except json.JSONDecodeError:
                column_mapping = None
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        # Auto-detect columns if mapping not provided
        if not column_mapping:
            column_mapping = _detect_csv_columns(df.columns.tolist())
        
        # Validate mapping has required columns
        required = ['ticker', 'transaction_type', 'shares', 'price', 'date']
        if not all(v is not None for k, v in column_mapping.items() if k in required):
            return jsonify({
                'error': 'Missing required columns',
                'required': required,
                'detected_columns': df.columns.tolist()
            }), 400
        
        # Process transactions
        transactions = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                # Fixed
                ticker = str(row.iloc[column_mapping['ticker']]).strip().upper()
                trans_type = str(row.iloc[column_mapping['transaction_type']]).strip().upper()
                shares = float(row.iloc[column_mapping['shares']])
                price = float(row.iloc[column_mapping['price']])
                date_str = str(row.iloc[column_mapping['date']])
                
                # Validate transaction type
                if trans_type not in ['BUY', 'SELL']:
                    errors.append(f"Row {idx+2}: Invalid transaction type '{trans_type}'")
                    continue
                
                # Validate shares and price
                if shares <= 0 or price <= 0:
                    errors.append(f"Row {idx+2}: Shares and price must be positive")
                    continue
                
                # Parse date
                try:
                    from dateutil import parser
                    trans_date = parser.parse(date_str)
                except:
                    errors.append(f"Row {idx+2}: Invalid date format '{date_str}'")
                    continue
                
                # Get commission if available
                commission = 0
                if column_mapping.get('commission') is not None:
                    try:
                        commission = float(row[column_mapping['commission']])
                    except:
                        commission = 0
                
                # Get notes if available
                notes = None
                if column_mapping.get('notes') is not None:
                    notes = str(row[column_mapping['notes']]).strip() if pd.notna(row[column_mapping['notes']]) else None
                
                transactions.append({
                    'ticker': ticker,
                    'type': trans_type,
                    'shares': shares,
                    'price': price,
                    'date': trans_date,
                    'commission': commission,
                    'notes': notes
                })
            except Exception as e:
                errors.append(f"Row {idx+2}: {str(e)}")
        
        if not transactions:
            return jsonify({'error': 'No valid transactions found', 'details': errors}), 400
        
        # Save transactions to database
        try:
            for trans_data in transactions:
                transaction = Transaction(
                    user_id=user_id,
                    ticker=trans_data['ticker'],
                    transaction_type=trans_data['type'],
                    shares=trans_data['shares'],
                    price=trans_data['price'],
                    transaction_date=trans_data['date'],
                    commission=trans_data['commission'],
                    notes=trans_data['notes']
                )
                db.session.add(transaction)
                
            # Validate all unique tickers first
            unique_tickers = list(set(t['ticker'] for t in transactions))
            for ticker in unique_tickers:
                is_valid, error_msg = _validate_ticker(ticker)
                if not is_valid:
                    errors.append(f'Invalid ticker: {error_msg}')
                    # Remove all transactions with this ticker
                    transactions = [t for t in transactions if t['ticker'] != ticker]

            if not transactions:
                return jsonify({'error': 'No valid transactions after ticker validation', 'details': errors}), 400
            
            # Commit all transactions
            db.session.commit()
            
            # Update holdings based on transactions
            _update_holdings(user_id)
            
            return jsonify({
                'success': True,
                'imported': len(transactions),
                'errors': errors if errors else [],
                'message': f"Successfully imported {len(transactions)} transaction(s)"
            }), 200
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    except Exception as e:
        print(f"CSV import error: {str(e)}")
        return jsonify({'error': f'Import failed: {str(e)}'}), 500

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Get user's transaction history"""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid user ID'}), 400

        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        transactions = Transaction.query.filter_by(user_id=user_id).order_by(
            Transaction.transaction_date.desc(),
            Transaction.id.desc()
        ).all()
        
        return jsonify([t.to_dict() for t in transactions]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transactions', methods=['POST'])
def add_transaction():
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        user_id = int(user_id)
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required = ['ticker', 'type', 'shares', 'price', 'date']
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        ticker = str(data['ticker']).strip().upper()
        trans_type = str(data['type']).strip().upper()

        # Validate transaction type early before the yfinance call
        if trans_type not in ['BUY', 'SELL']:
            return jsonify({'error': 'Transaction type must be BUY or SELL'}), 400

        # Validate shares and price before the yfinance call
        try:
            shares = float(data['shares'])
            price = float(data['price'])
            commission = float(data.get('commission', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Shares, price, and commission must be numbers'}), 400

        if shares <= 0:
            return jsonify({'error': 'Shares must be greater than zero'}), 400
        if price <= 0:
            return jsonify({'error': 'Price must be greater than zero'}), 400
        if commission < 0:
            return jsonify({'error': 'Commission cannot be negative'}), 400
        if shares > 1_000_000:
            return jsonify({'error': 'Shares value is unrealistically large'}), 400
        if price > 1_000_000:
            return jsonify({'error': 'Price value is unrealistically large'}), 400

        # Validate ticker exists
        is_valid, error_msg = _validate_ticker(ticker)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Validate date
        try:
            from dateutil import parser as date_parser
            trans_date = date_parser.parse(str(data['date']))
            if trans_date.date() > datetime.now().date():
                return jsonify({'error': 'Transaction date cannot be in the future'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400

        notes = str(data['notes']).strip()[:500] if data.get('notes') else None

        transaction = Transaction(
            user_id=user_id,
            ticker=ticker,
            transaction_type=trans_type,
            shares=shares,
            price=price,
            transaction_date=trans_date,
            commission=commission,
            notes=notes
        )
        db.session.add(transaction)
        db.session.commit()
        _update_holdings(user_id)

        return jsonify(transaction.to_dict()), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# ==================== HELPER FUNCTIONS ====================

def _detect_csv_columns(columns):
    """Auto-detect CSV column mapping"""
    columns_lower = [c.lower().strip() for c in columns]
    
    mapping = {
        'ticker': None,
        'transaction_type': None,
        'shares': None,
        'price': None,
        'date': None,
        'commission': None,
        'notes': None
    }
    
    # Ticker patterns
    ticker_patterns = ['ticker', 'symbol', 'stock', 'symbol']
    trans_type_patterns = ['type', 'transaction', 'action', 'side']
    shares_patterns = ['shares', 'quantity', 'qty', 'amount']
    price_patterns = ['price', 'rate', 'cost']
    date_patterns = ['date', 'trade date', 'transaction date']
    commission_patterns = ['commission', 'fee', 'fees']
    notes_patterns = ['notes', 'memo', 'description']
    
    for i, col in enumerate(columns_lower):
        # Ticker
        if mapping['ticker'] is None and any(p in col for p in ticker_patterns):
            mapping['ticker'] = i
        # Type
        elif mapping['transaction_type'] is None and any(p in col for p in trans_type_patterns):
            mapping['transaction_type'] = i
        # Shares
        elif mapping['shares'] is None and any(p in col for p in shares_patterns):
            mapping['shares'] = i
        # Price
        elif mapping['price'] is None and any(p in col for p in price_patterns):
            mapping['price'] = i
        # Date
        elif mapping['date'] is None and any(p in col for p in date_patterns):
            mapping['date'] = i
        # Commission
        elif mapping['commission'] is None and any(p in col for p in commission_patterns):
            mapping['commission'] = i
        # Notes
        elif mapping['notes'] is None and any(p in col for p in notes_patterns):
            mapping['notes'] = i
    
    return mapping

def _update_holdings(user_id):
    transactions = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.transaction_date).all()
    
    holdings_dict = {}
    for trans in transactions:
        ticker = trans.ticker
        if ticker not in holdings_dict:
            holdings_dict[ticker] = {
                'shares': 0,
                'total_cost': 0,
                'last_price': trans.price
            }
        
        before = holdings_dict[ticker]['shares']
        
        if trans.transaction_type == 'BUY':
            holdings_dict[ticker]['shares'] += trans.shares
            holdings_dict[ticker]['total_cost'] += (trans.shares * trans.price) + trans.commission
            holdings_dict[ticker]['last_price'] = trans.price
        else:  # SELL
            if holdings_dict[ticker]['shares'] > 0:
                sell_shares = min(trans.shares, holdings_dict[ticker]['shares'])
                cost_per_share = holdings_dict[ticker]['total_cost'] / holdings_dict[ticker]['shares']
                holdings_dict[ticker]['total_cost'] -= cost_per_share * sell_shares
                holdings_dict[ticker]['shares'] -= sell_shares
        
        after = holdings_dict[ticker]['shares']
        print(f"DEBUG: {ticker} shares: {before} -> {after} after {trans.transaction_type} {trans.shares}")
    
    print(f"DEBUG: final holdings_dict: {holdings_dict}")

    # Write to database
    for ticker, holding_data in holdings_dict.items():
        if holding_data['shares'] > 0:
            avg_cost = holding_data['total_cost'] / holding_data['shares']
            holding = Holding.query.filter_by(user_id=user_id, ticker=ticker).first()
            if holding:
                holding.shares = holding_data['shares']
                holding.average_cost = avg_cost
            else:
                holding = Holding(
                    user_id=user_id,
                    ticker=ticker,
                    shares=holding_data['shares'],
                    average_cost=avg_cost
                )
                db.session.add(holding)
        else:
            # Position fully sold — remove holding
            holding = Holding.query.filter_by(user_id=user_id, ticker=ticker).first()
            if holding:
                db.session.delete(holding)

    db.session.commit()

@app.route('/api/price-history', methods=['POST'])
def get_price_history():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        date_str = data.get('date')
        
        if not ticker or not date_str:
            return jsonify({'error': 'ticker and date are required'}), 400
        
        # Parse date
        from datetime import datetime, timedelta
        try:
            from dateutil import parser
            date_obj = parser.parse(date_str)
        except:
            return jsonify({'error': 'Invalid date format'}), 400
        
        # Fetch price for a small range around the date (since market may be closed)
        start_date = date_obj - timedelta(days=5)
        end_date = date_obj + timedelta(days=1)
        
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        if hist.empty:
            return jsonify({'error': f'No price data found for {ticker} around {date_str}'}), 404
        
        # Strip timezone from hist.index to make it naive (assuming UTC)
        hist.index = hist.index.tz_localize(None)
        
        # Ensure date_obj is naive for consistency
        date_obj_naive = date_obj.replace(tzinfo=None) if date_obj.tzinfo else date_obj
        date_obj_ts = pd.Timestamp(date_obj_naive.date())
        
        # Find closest date to the requested date
        # Use pd.Series to enable .abs() on TimedeltaIndex
        time_diffs = pd.Series(hist.index - date_obj_ts)
        closest_idx = time_diffs.abs().argmin()
        closest_date = hist.index[closest_idx].date()
        
        price = float(hist['Close'].iloc[closest_idx])
        
        return jsonify({
            'ticker': ticker,
            'date': closest_date.isoformat(),
            'price': round(price, 2),
            'source': 'historical'
        }), 200
    
    except Exception as e:
        print(f"Price history error: {str(e)}")
        return jsonify({'error': f'Failed to fetch price: {str(e)}'}), 500

@app.route('/api/sync/holdings', methods=['POST'])
def sync_holdings():
    """Manually sync holdings from transactions (for maintenance/debugging)"""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_id = int(user_id)
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Recalculate holdings for this user
        _update_holdings(user_id)
        
        # Return updated holdings
        user_holdings = Holding.query.filter_by(user_id=user_id).all()
        
        return jsonify({
            'success': True,
            'message': f'Holdings synced. Total holdings: {len(user_holdings)}',
            'holdings': [h.to_dict() for h in user_holdings]
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Sync failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, port=5001, host='127.0.0.1')