'use client';

import { useState, useEffect, useRef } from 'react';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  volume: string;
}

export default function WatchlistPage() {
  const [watchlist, setWatchlist] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const hasFetched = useRef(false);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;

    fetchWatchlist();
  }, []);

  const fetchWatchlist = async () => {
    try {
      setLoading(true);
      setError(''); // Clear any previous errors
      const watchlistRes = await fetch('/api/proxy?endpoint=watchlist');

      if (watchlistRes.ok) {
        const watchlistData = await watchlistRes.json();
        setWatchlist(watchlistData);
        setError(''); // Ensure error is cleared on success
      } else {
        setError('Failed to load watchlist');
        setWatchlist([]);
      }
    } catch (err) {
      console.error('Error fetching watchlist:', err);
      setError('Error loading watchlist');
      setWatchlist([]);
    } finally {
      setLoading(false);
    }
  };

  const gainers = watchlist.filter(s => s.change > 0);
  const losers = watchlist.filter(s => s.change < 0);

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black flex flex-col">
      <Header />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 w-full">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-black dark:text-white mb-2">Market Watchlist</h1>
          <p className="text-neutral-600 dark:text-neutral-400">Track top stocks and market movements</p>
        </div>

        {error && watchlist.length > 0 && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        {loading ? (
          <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-8 text-center">
            <p className="text-neutral-600 dark:text-neutral-400">Loading market data...</p>
          </div>
        ) : (
          <>
            {/* Market Overview Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">Total Stocks</p>
                <p className="text-2xl font-bold text-black dark:text-white">{watchlist.length}</p>
              </div>

              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">Gainers</p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">{gainers.length}</p>
              </div>

              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">Losers</p>
                <p className="text-2xl font-bold text-red-600 dark:text-red-400">{losers.length}</p>
              </div>
            </div>

            {/* Watchlist Table */}
            <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/50">
                      <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Symbol
                      </th>
                      <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Company Name
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Price
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Change
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Volume
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {watchlist.map((stock) => (
                      <tr
                        key={stock.symbol}
                        className="border-b border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition"
                      >
                        <td className="px-6 py-4 text-sm font-semibold text-black dark:text-white">
                          {stock.symbol}
                        </td>
                        <td className="px-6 py-4 text-sm text-neutral-600 dark:text-neutral-400">
                          {stock.name}
                        </td>
                        <td className="px-6 py-4 text-sm text-right font-semibold text-black dark:text-white">
                          ${stock.price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </td>
                        <td className={`px-6 py-4 text-sm font-semibold text-right ${
                          stock.change >= 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}>
                          {stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                          {stock.volume}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Market Info */}
            <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">Market Information</h3>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                Watchlist data is updated daily with the latest closing prices. Prices shown are from the most recent trading day.
              </p>
            </div>
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}
