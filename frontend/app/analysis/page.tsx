'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';

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

export default function AnalysisPage() {
  const router = useRouter();
  const [loading, setLoading] = useState<boolean>(true);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [userId, setUserId] = useState<number | null>(null);
  const hasFetched = useRef(false);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;

    const initializeAuth = async () => {
      try {
        const userRes = await fetch('/api/user');
        const { userId: id } = await userRes.json();

        if (!id) {
          router.push('/signin');
          return;
        }

        setUserId(id);
        fetchAndAnalyze(id);
      } catch (error) {
        console.error('Auth error:', error);
        router.push('/signin');
      }
    };

    initializeAuth();
  }, [router]);

  const fetchAndAnalyze = async (id: number) => {
    setLoading(true);
    setError(null);

    try {
      // 1. Fetch current holdings
      const holdingsRes = await fetch('/api/proxy?endpoint=holdings', {
        headers: {
          'X-User-ID': id.toString()
        }
      });

      if (!holdingsRes.ok) {
        throw new Error('Failed to fetch holdings');
      }

      const holdingsData: Holding[] = await holdingsRes.json();
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

  // Prevent rendering until auth is verified
  if (!userId) {
    return null;
  }

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black flex flex-col">
      <Header />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 w-full">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-black dark:text-white mb-2">Portfolio Analysis</h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            Advanced metrics for your {holdings.length} holdings
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-700 dark:text-red-300"><strong>Error:</strong> {error}</p>
          </div>
        )}

        {loading ? (
          <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-8 text-center">
            <p className="text-neutral-600 dark:text-neutral-400">Loading your portfolio data...</p>
          </div>
        ) : results && results.portfolio_metrics ? (
          <div>
            {/* Portfolio Metrics */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Portfolio Metrics</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard label="Beta" value={results.portfolio_metrics.beta.toFixed(3)} />
                <MetricCard label="Max Drawdown" value={`${(results.portfolio_metrics.max_drawdown * 100).toFixed(2)}%`} />
                <MetricCard label="Annual Return" value={`${(results.portfolio_metrics.annualized_return * 100).toFixed(2)}%`} />
                <MetricCard label="Annual Volatility" value={`${(results.portfolio_metrics.annualized_volatility * 100).toFixed(2)}%`} />
              </div>
            </div>

            {/* YTD Portfolio Performance Chart */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">YTD Portfolio Performance</h2>
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <YTDPortfolioChart tickers={results.tickers} weights={results.weights} />
              </div>
            </div>

            {/* Individual Stock Metrics */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Individual Stock Metrics</h2>
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/50">
                        <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">Ticker</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Current Price</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">50-day MA</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">200-day MA</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Beta</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Sharpe Ratio</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">RSI</th>
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Volatility</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.tickers.map(ticker => (
                        <tr key={ticker} className="border-b border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition">
                          <td className="px-6 py-4 text-sm font-semibold text-black dark:text-white">{ticker}</td>
                          <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                            {results.current_prices[ticker] ? `$${results.current_prices[ticker].toFixed(2)}` : 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                            {results.ma_50[ticker] ? `$${results.ma_50[ticker].toFixed(2)}` : 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                            {results.ma_200[ticker] ? `$${results.ma_200[ticker].toFixed(2)}` : 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                            {results.individual_betas[ticker] ?? 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                            {results.individual_sharpe_ratios[ticker] ?? 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm text-right">
                            {results.rsi[ticker] !== null && results.rsi[ticker] !== undefined ? (
                              <span className={getRSIClass(results.rsi[ticker])}>
                                {results.rsi[ticker]!.toFixed(0)}
                              </span>
                            ) : 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                            {results.volatility[ticker] ? (results.volatility[ticker] * 100).toFixed(2) + '%' : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Correlation Matrix */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-2">Correlation Matrix</h2>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                Shows how your holdings move together (1.0 = perfect correlation, -1.0 = inverse correlation)
              </p>
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/50">
                        <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">Ticker</th>
                        {results.tickers.map(ticker => (
                          <th key={ticker} className="px-6 py-3 text-center text-sm font-semibold text-neutral-900 dark:text-neutral-100">{ticker}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.tickers.map(ticker1 => (
                        <tr key={ticker1} className="border-b border-neutral-200 dark:border-neutral-700">
                          <td className="px-6 py-4 text-sm font-semibold text-black dark:text-white">{ticker1}</td>
                          {results.tickers.map(ticker2 => {
                            const corr = results.correlation[ticker1][ticker2];
                            return (
                              <td 
                                key={ticker2} 
                                className="px-6 py-4 text-sm text-center text-neutral-600 dark:text-neutral-400"
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

            {/* Metrics Explanation */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Understanding the Metrics</h2>
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold text-black dark:text-white mb-2">Beta</h3>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      Beta measures how much a stock or portfolio moves relative to the overall market (S&P 500). 
                      A beta of 1.0 means it moves in line with the market, while a beta above 1.0 indicates higher volatility and below 1.0 indicates lower volatility.
                    </p>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-black dark:text-white mb-2">Sharpe Ratio</h3>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      The Sharpe Ratio measures risk-adjusted returns, showing how much return you're getting for the level of risk taken. 
                      Higher values are better—a ratio above 1.0 is considered good, and above 2.0 is very good.
                    </p>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-black dark:text-white mb-2">RSI (Relative Strength Index)</h3>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      RSI is a momentum indicator ranging from 0 to 100 that helps identify overbought or oversold conditions. 
                      Values above 70 suggest a stock may be overbought (potentially overvalued), while values below 30 suggest it may be oversold (potentially undervalued).
                    </p>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-black dark:text-white mb-2">Volatility</h3>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      Volatility measures how much a stock's price fluctuates over time, expressed as an annualized percentage. 
                      Higher volatility means larger price swings and potentially higher risk, while lower volatility indicates more stable, predictable price movements.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </main>

      <Footer />
    </div>
  );
}

// YTD Portfolio Chart Component
// YTD Portfolio Chart Component
function YTDPortfolioChart({ tickers, weights }: { tickers: string[]; weights: number[] }) {
  const [chartData, setChartData] = useState<{ date: string; value: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await axios.post(
          'http://localhost:5001/api/ytd-chart',
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

        const { dates, values } = response.data;
        
        const chartPoints = dates.map((date: string, idx: number) => ({
          date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          value: values[idx]
        }));

        setChartData(chartPoints);
      } catch (error: any) {
        console.error('Error fetching chart data:', error);
        setErrorMsg(error.response?.data?.error || 'Failed to load chart data');
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [tickers, weights]);

  if (loading) {
    return <div className="text-center py-8 text-neutral-600 dark:text-neutral-400">Loading chart...</div>;
  }

  if (errorMsg || chartData.length === 0) {
    return <div className="text-center py-8 text-neutral-600 dark:text-neutral-400">{errorMsg || 'No data available'}</div>;
  }

  // Simple SVG line chart
  const width = 900;
  const height = 400;
  const padding = 60;
  
  const minValue = Math.min(...chartData.map(d => d.value));
  const maxValue = Math.max(...chartData.map(d => d.value));
  const range = maxValue - minValue || 1;

  const points = chartData
    .map((d, i) => {
      const x = padding + (i / (chartData.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((d.value - minValue) / range) * (height - 2 * padding);
      return `${x},${y}`;
    })
    .join(' ');

  // Calculate current performance
  const currentValue = chartData[chartData.length - 1]?.value || 100;
  const performancePercent = currentValue - 100;
  const performanceColor = performancePercent >= 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)';

  return (
    <div className="w-full">
      <div className="mb-4 text-center">
        <div className="text-3xl font-bold" style={{ color: performanceColor }}>
          {performancePercent >= 0 ? '+' : ''}{performancePercent.toFixed(2)}%
        </div>
        <div className="text-sm text-neutral-500 dark:text-neutral-400">
          YTD Performance (Starting Value: 100)
        </div>
      </div>
      
      <div className="w-full overflow-x-auto">
        <svg width={width} height={height} className="mx-auto">
          {/* Grid lines */}
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="currentColor" strokeWidth="2" className="text-neutral-300 dark:text-neutral-700" />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="currentColor" strokeWidth="2" className="text-neutral-300 dark:text-neutral-700" />
          
          {/* Horizontal grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(pct => {
            const y = height - padding - pct * (height - 2 * padding);
            const value = minValue + pct * range;
            return (
              <g key={pct}>
                <line
                  x1={padding}
                  y1={y}
                  x2={width - padding}
                  y2={y}
                  stroke="currentColor"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                  className="text-neutral-200 dark:text-neutral-800"
                />
                <text
                  x={padding - 10}
                  y={y + 4}
                  textAnchor="end"
                  className="text-xs fill-neutral-600 dark:fill-neutral-400"
                >
                  {value.toFixed(1)}
                </text>
              </g>
            );
          })}
          
          {/* 100 baseline (starting value) */}
          {minValue <= 100 && maxValue >= 100 && (
            <line
              x1={padding}
              y1={height - padding - ((100 - minValue) / range) * (height - 2 * padding)}
              x2={width - padding}
              y2={height - padding - ((100 - minValue) / range) * (height - 2 * padding)}
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray="5,5"
              className="text-neutral-400 dark:text-neutral-600"
            />
          )}
          
          {/* Area fill under the line */}
          <polygon
            points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`}
            fill={performanceColor}
            opacity="0.1"
          />
          
          {/* Line chart */}
          <polyline
            points={points}
            fill="none"
            stroke={performanceColor}
            strokeWidth="3"
          />
          
          {/* Date labels */}
          {[0, Math.floor(chartData.length / 2), chartData.length - 1].map(idx => {
            const x = padding + (idx / (chartData.length - 1)) * (width - 2 * padding);
            return (
              <text
                key={idx}
                x={x}
                y={height - padding + 20}
                textAnchor="middle"
                className="text-xs fill-neutral-600 dark:fill-neutral-400"
              >
                {chartData[idx]?.date}
              </text>
            );
          })}
        </svg>
      </div>
      
      <p className="text-xs text-center text-neutral-500 dark:text-neutral-400 mt-4">
        Portfolio indexed to 100 at start of {new Date().getFullYear()}. Current value: {currentValue.toFixed(2)}
      </p>
    </div>
  );
}

// Helper component for metric cards
function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">{label}</p>
      <p className="text-2xl font-bold text-black dark:text-white">{value}</p>
    </div>
  );
}

// Helper function for RSI styling
function getRSIClass(rsi: number | null): string {
  if (rsi === null) return 'text-neutral-600 dark:text-neutral-400';
  if (rsi > 70) return 'text-red-600 dark:text-red-400 font-semibold'; // Overbought
  if (rsi < 30) return 'text-green-600 dark:text-green-400 font-semibold'; // Oversold
  return 'text-neutral-600 dark:text-neutral-400';
}

// Helper function for correlation heatmap colors
function getCorrelationColor(value: number | null): string {
  if (value === null || value === undefined) return 'transparent';
  if (value >= 0.8) return 'rgba(34, 197, 94, 0.2)'; // Strong positive - green
  if (value >= 0.5) return 'rgba(59, 130, 246, 0.2)'; // Moderate positive - blue
  if (value >= -0.5) return 'rgba(234, 179, 8, 0.1)'; // Weak - yellow
  return 'rgba(239, 68, 68, 0.2)'; // Negative - red
}