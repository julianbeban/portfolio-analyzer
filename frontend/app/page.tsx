'use client';

import { useState, useEffect, useRef } from 'react';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';

interface PortfolioStat {
  label: string;
  value: string;
  change: string;
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

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  volume: string;
}

export default function Home() {
  const [portfolioStats, setPortfolioStats] = useState<PortfolioStat[]>([
    { label: "Total Value", value: "$0.00", change: "–" },
    { label: "Today's Gain/Loss", value: "$0.00", change: "–" },
    { label: "Year to Date", value: "+0.00%", change: "–" },
    { label: "Portfolio Return", value: "$0.00", change: "–" }
  ]);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [watchlist, setWatchlist] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const hasFetched = useRef(false);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;

    const fetchData = async () => {
      try {
      setLoading(true);
      const baseUrl = '/api/proxy';

      // Get user ID from API route
      const userRes = await fetch('/api/user');
      const { userId } = await userRes.json();

      // If not logged in, just load watchlist
      if (!userId) {
        const watchlistRes = await fetch(`${baseUrl}?endpoint=watchlist`);
        if (watchlistRes.ok) {
          const watchlistData = await watchlistRes.json();
          setWatchlist(watchlistData);
        }
        setError('');
        setLoading(false);
        return;
      }

      const headers = {
        'X-User-ID': userId.toString()
      };

      // Try to fetch authenticated data, but don't break if it fails
      const [statsRes, holdingsRes, watchlistRes] = await Promise.all([
        fetch(`${baseUrl}?endpoint=portfolio`, { headers }),
        fetch(`${baseUrl}?endpoint=holdings`, { headers }),
        fetch(`${baseUrl}?endpoint=watchlist`, { headers })
      ]);

      // Always load watchlist (public data)
      if (watchlistRes.ok) {
        const watchlistData = await watchlistRes.json();
        setWatchlist(watchlistData);
      }

      // Load portfolio/holdings if available, otherwise keep defaults
      if (statsRes.ok && holdingsRes.ok) {
        const statsData = await statsRes.json();
        const holdingsData = await holdingsRes.json();

        const totalValue = Number(statsData.totalValue || 0);
        const todayGain = Number(statsData.todayGain || 0);
        const todayGainPercent = Number(statsData.todayGainPercent || 0);
        const ytdReturn = Number(statsData.ytdReturn || 0);
        const portfolioReturn = Number(statsData.portfolioReturn || 0);
        const portfolioReturnPercent = Number(statsData.portfolioReturnPercent || 0);

        setPortfolioStats([
          {
            label: "Total Value",
            value: `$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
            change: `${portfolioReturnPercent >= 0 ? '+' : ''}${portfolioReturnPercent.toFixed(2)}%`
          },
          {
            label: "Today's Gain/Loss",
            value: `${todayGain >= 0 ? '+' : ''}$${todayGain.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
            change: `${todayGainPercent >= 0 ? '+' : ''}${todayGainPercent.toFixed(2)}%`
          },
          {
            label: "Year to Date",
            value: `${ytdReturn >= 0 ? '+' : ''}${ytdReturn.toFixed(2)}%`,
            change: "vs S&P 500"
          },
          {
            label: "Portfolio Return",
            value: `${portfolioReturn >= 0 ? '+' : ''}$${portfolioReturn.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
            change: `${portfolioReturnPercent >= 0 ? '+' : ''}${portfolioReturnPercent.toFixed(2)}%`
          }
        ]);
        setHoldings(holdingsData || []);
      }

      setError('');
    } catch (err) {
      console.error('Error fetching data:', err);
      // Don't set error - use default values instead
    } finally {
      setLoading(false);
    }
  };

  fetchData();
}, []);

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black">
      <Header />

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <p className="text-neutral-600 dark:text-neutral-400">Loading portfolio data...</p>
          </div>
        ) : (
          <>
            {/* Portfolio Overview Stats */}
            <section className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Portfolio Overview</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {portfolioStats.map((stat, idx) => (
                  <div key={idx} className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                    <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">{stat.label}</p>
                    <p className="text-2xl font-bold text-black dark:text-white mb-1">{stat.value}</p>
                    <p className={`text-xs font-medium ${
                      stat.change.includes('-') && stat.change !== '–'
                        ? 'text-red-600 dark:text-red-400'
                        : 'text-green-600 dark:text-green-400'
                    }`}>{stat.change}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Holdings Section */}
            <section className="mb-8">
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Your Holdings</h2>
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800">
                      <tr>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Symbol</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Shares</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Avg Cost</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Current Price</th>
                        <th className="px-6 py-4 text-right text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Gain/Loss</th>
                        <th className="px-6 py-4 text-right text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {holdings.map((holding, idx) => (
                        <tr key={idx} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition">
                          <td className="px-6 py-4">
                            <div>
                              <p className="font-semibold text-black dark:text-white">{holding.symbol}</p>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-neutral-700 dark:text-neutral-300">{holding.shares}</td>
                          <td className="px-6 py-4 text-neutral-700 dark:text-neutral-300">${holding.avgCost}</td>
                          <td className="px-6 py-4 font-semibold text-black dark:text-white">${holding.current}</td>
                          <td className="px-6 py-4 text-right">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              holding.gainPercent >= 0
                                ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                                : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
                            }`}>
                              {holding.gainPercent >= 0 ? '+' : ''}{holding.gainPercent}%
                            </span>
                          </td>
                          <td className="px-6 py-4 text-right font-semibold text-black dark:text-white">${holding.totalValue}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* Watchlist Section */}
            <section>
              <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Market Watchlist</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {watchlist.map((stock, idx) => (
                  <div key={idx} className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-5 hover:shadow-lg transition cursor-pointer">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <p className="text-lg font-bold text-black dark:text-white">{stock.symbol}</p>
                        <p className="text-xs text-neutral-500 dark:text-neutral-400">{stock.name}</p>
                      </div>
                      <span className={`text-sm font-semibold px-2 py-1 rounded ${
                        stock.change >= 0
                          ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                          : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
                      }`}>
                        {stock.change >= 0 ? '+' : ''}{stock.change}%
                      </span>
                    </div>
                    <p className="text-2xl font-bold text-black dark:text-white mb-2">${stock.price}</p>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">Vol: {stock.volume}</p>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}
