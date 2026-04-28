'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

interface PortfolioMetrics {
  sharpe_ratio: number;
  beta: number;
  max_drawdown: number;
  annualized_return: number;
  annualized_volatility: number;
}

interface AnalysisResults {
  correlation: { [key: string]: { [key: string]: number } };
  volatility: { [key: string]: number };
  current_prices: { [key: string]: number };
  ma_50: { [key: string]: number };
  ma_200: { [key: string]: number };
  individual_betas: { [key: string]: number };
  individual_sharpe_ratios: { [key: string]: number };
  rsi: { [key: string]: number | null };
  weights: number[];
  portfolio_metrics: PortfolioMetrics;
  tickers: string[];
}

interface Holding {
  symbol: string;
  shares: number;
  avgCost: number;
  current: number;
  gainAmount: number;
  gainPercent: number;
  totalValue: number;
}

export default function Portfolio() {
  const [loading, setLoading] = useState<boolean>(true);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);

  useEffect(() => {
    fetchAndAnalyze();
  }, []);

  const fetchAndAnalyze = async () => {
    setLoading(true);
    setError(null);
  
    try {
      // Get user ID from localStorage (or however your auth works)
      const userId = localStorage.getItem('userId');
      
      let holdingsData: Holding[];
      
      try {
        // 1. Try to fetch current holdings
        const holdingsResponse = await axios.get<Holding[]>('http://localhost:5001/api/holdings', {
          headers: userId ? { 'X-User-ID': userId } : {}
        });
        holdingsData = holdingsResponse.data;
      } catch (err: any) {
        // If unauthorized, use mock data for testing
        if (err.response?.status === 401) {
          console.log('Not authenticated, using mock data');
          holdingsData = [
            { symbol: 'AAPL', shares: 50, avgCost: 185.30, current: 234.50, gainAmount: 0, gainPercent: 0, totalValue: 0 },
            { symbol: 'MSFT', shares: 30, avgCost: 405.20, current: 421.30, gainAmount: 0, gainPercent: 0, totalValue: 0 },
            { symbol: 'VOO', shares: 25, avgCost: 418.50, current: 486.80, gainAmount: 0, gainPercent: 0, totalValue: 0 },
          ];
        } else {
          throw err;
        }
      }
      
      setHoldings(holdingsData);
  
      if (!holdingsData || holdingsData.length === 0) {
        setError('No holdings found. Add some stocks to your portfolio first.');
        setLoading(false);
        return;
      }
  
      if (holdingsData.length < 2) {
        setError('Need at least 2 holdings for correlation analysis.');
        setLoading(false);
        return;
      }
  
      // 2. Extract tickers and shares (use shares as weights)
      const tickers = holdingsData.map(h => h.symbol);
      const weights = holdingsData.map(h => h.shares);
  
      // 3. Analyze portfolio
      const analysisResponse = await axios.post<AnalysisResults>(
        'http://localhost:5001/api/analyze',
        {
          tickers: tickers,
          weights: weights
        },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
  
      // Parse the response if it's a string
      const data = typeof analysisResponse.data === 'string' 
        ? JSON.parse(analysisResponse.data) 
        : analysisResponse.data;
  
      setResults(data);
    } catch (err: any) {
      console.error('Error in fetchAndAnalyze:', err);
      setError(err.response?.data?.error || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
        <h1 className="page-title">Portfolio Analysis</h1>
        <p>Loading your portfolio data...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 className="page-title">Portfolio Analysis</h1>
      <p className="page-subtitle" style={{ marginBottom: '30px' }}>
        Advanced metrics for your {holdings.length} holdings
      </p>

      {/* Error Display */}
      {error && (
        <div className="error-banner" style={{ marginBottom: '20px', borderRadius: '8px' }}>
          <p className="error-text"><strong>Error:</strong> {error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && results.portfolio_metrics && (
        <div>
          {/* Portfolio Metrics */}
          <div style={{ marginBottom: '30px' }}>
            <h2 className="section-title">Portfolio Metrics</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
              <MetricCard label="Beta" value={results.portfolio_metrics.beta.toFixed(3)} />
              <MetricCard label="Max Drawdown" value={`${(results.portfolio_metrics.max_drawdown * 100).toFixed(2)}%`} />
              <MetricCard label="Annual Return" value={`${(results.portfolio_metrics.annualized_return * 100).toFixed(2)}%`} />
              <MetricCard label="Annual Volatility" value={`${(results.portfolio_metrics.annualized_volatility * 100).toFixed(2)}%`} />
            </div>
          </div>

          {/* Individual Stock Metrics */}
          <div style={{ marginBottom: '30px' }}>
            <h2 className="section-title">Individual Stock Metrics</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: 'var(--background)', borderBottom: '2px solid #ddd' }}>
                    <th className="table-header">Ticker</th>
                    <th className="table-header">Current Price</th>
                    <th className="table-header">50-day MA</th>
                    <th className="table-header">200-day MA</th>
                    <th className="table-header">Beta</th>
                    <th className="table-header">Sharpe Ratio</th>
                    <th className="table-header">RSI</th>
                    <th className="table-header">Volatility</th>
                  </tr>
                </thead>
                <tbody>
                  {results.tickers.map(ticker => (
                    <tr key={ticker} className="table-row">
                      <td className="table-cell"><strong>{ticker}</strong></td>
                      <td className="table-cell">{results.current_prices[ticker] ? `$${results.current_prices[ticker].toFixed(2)}` : 'N/A'}</td>
                      <td className="table-cell">{results.ma_50[ticker] ? `$${results.ma_50[ticker].toFixed(2)}` : 'N/A'}</td>
                      <td className="table-cell">{results.ma_200[ticker] ? `$${results.ma_200[ticker].toFixed(2)}` : 'N/A'}</td>
                      <td className="table-cell">{results.individual_betas[ticker] ?? 'N/A'}</td>
                      <td className="table-cell">{results.individual_sharpe_ratios[ticker] ?? 'N/A'}</td>
                      <td className="table-cell">
                        {results.rsi[ticker] !== null && results.rsi[ticker] !== undefined ? (
                          <span className={getRSIClass(results.rsi[ticker])}>
                            {results.rsi[ticker]!.toFixed(0)}
                          </span>
                        ) : 'N/A'}
                      </td>
                      <td className="table-cell">{results.volatility[ticker] ? (results.volatility[ticker] * 100).toFixed(2) + '%' : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Correlation Matrix */}
          <div>
            <h2 className="section-title">Correlation Matrix</h2>
            <p className="text-muted" style={{ marginBottom: '15px' }}>
              Shows how your holdings move together (1.0 = perfect correlation, -1.0 = inverse correlation)
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: 'var(--background)' }}>
                    <th className="table-header">Ticker</th>
                    {results.tickers.map(ticker => (
                      <th key={ticker} className="table-header">{ticker}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.tickers.map(ticker1 => (
                    <tr key={ticker1} className="table-row">
                      <td className="table-cell"><strong>{ticker1}</strong></td>
                      {results.tickers.map(ticker2 => {
                        const corr = results.correlation[ticker1][ticker2];
                        return (
                          <td 
                            key={ticker2} 
                            className="table-cell"
                            style={{
                              backgroundColor: getCorrelationColor(corr),
                              fontWeight: ticker1 === ticker2 ? 'bold' : 'normal'
                            }}
                          >
                            {corr !== null && corr !== undefined ? corr.toFixed(2) : 'N/A'}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper component for metric cards
function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card-pad">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

// Helper function for RSI styling
function getRSIClass(rsi: number | null): string {
  if (rsi === null) return '';
  if (rsi > 70) return 'badge-negative'; // Overbought
  if (rsi < 30) return 'badge-positive'; // Oversold
  return '';
}

// Helper function for correlation heatmap colors
function getCorrelationColor(value: number | null): string {
  if (value === null || value === undefined) return 'transparent';
  if (value >= 0.8) return 'rgba(34, 197, 94, 0.2)'; // Strong positive - green
  if (value >= 0.5) return 'rgba(59, 130, 246, 0.2)'; // Moderate positive - blue
  if (value >= -0.5) return 'rgba(234, 179, 8, 0.1)'; // Weak - yellow
  return 'rgba(239, 68, 68, 0.2)'; // Negative - red
}