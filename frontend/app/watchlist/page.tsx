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

interface QuickBuyState {
  open: boolean;
  symbol: string;
  currentPrice: number | null;
  priceLoading: boolean;
  shares: string;
  submitting: boolean;
  message: string;
  success: boolean;
}

const QUICK_BUY_DEFAULT: QuickBuyState = {
  open: false,
  symbol: '',
  currentPrice: null,
  priceLoading: false,
  shares: '',
  submitting: false,
  message: '',
  success: false
};

export default function WatchlistPage() {
  const [watchlist, setWatchlist] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [userId, setUserId] = useState<number | null>(null);
  const [quickBuy, setQuickBuy] = useState<QuickBuyState>(QUICK_BUY_DEFAULT);
  const hasFetched = useRef(false);
  const sharesInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;

    const init = async () => {
      // Auth check — same pattern as holdings/transactions
      try {
        const userRes = await fetch('/api/user');
        const { userId: id } = await userRes.json();
        if (id) setUserId(id);
        // No redirect — watchlist is viewable without auth,
        // quick buy just won't be available
      } catch {
        // Non-fatal — user just won't be able to quick buy
      }

      fetchWatchlist();
    };

    init();
  }, []);

  // Focus shares input when modal opens
  useEffect(() => {
    if (quickBuy.open && sharesInputRef.current) {
      setTimeout(() => sharesInputRef.current?.focus(), 50);
    }
  }, [quickBuy.open]);

  // Close modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && quickBuy.open) closeQuickBuy();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [quickBuy.open]);

  const fetchWatchlist = async () => {
    try {
      setLoading(true);
      setError('');
      const res = await fetch('/api/proxy?endpoint=watchlist');
      if (res.ok) {
        setWatchlist(await res.json());
      } else {
        setError('Failed to load watchlist');
        setWatchlist([]);
      }
    } catch {
      setError('Error loading watchlist');
      setWatchlist([]);
    } finally {
      setLoading(false);
    }
  };

  // ==================== QUICK BUY ====================

  const openQuickBuy = async (stock: Stock) => {
    // Open modal immediately with the watchlist price as placeholder
    setQuickBuy({
      ...QUICK_BUY_DEFAULT,
      open: true,
      symbol: stock.symbol,
      currentPrice: stock.price, // Use watchlist price instantly
      priceLoading: true          // Then fetch fresh price
    });

    // Fetch fresh price in the background
    try {
      const today = new Date().toLocaleDateString('en-CA'); // gives YYYY-MM-DD in local time
      const res = await fetch('/api/proxy?endpoint=price-history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: stock.symbol, date: today })
      });

      if (res.ok) {
        const data = await res.json();
        setQuickBuy(prev => ({
          ...prev,
          currentPrice: data.price,
          priceLoading: false
        }));
      } else {
        // Fresh fetch failed — keep the watchlist price, not a fatal error
        setQuickBuy(prev => ({ ...prev, priceLoading: false }));
      }
    } catch {
      setQuickBuy(prev => ({ ...prev, priceLoading: false }));
    }
  };

  const closeQuickBuy = () => {
    if (quickBuy.submitting) return; // Don't close mid-submit
    setQuickBuy(QUICK_BUY_DEFAULT);
  };

  const handleQuickBuySubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!userId) {
      setQuickBuy(prev => ({ ...prev, message: 'You must be signed in to buy' }));
      return;
    }

    const shares = parseFloat(quickBuy.shares);
    if (!shares || shares <= 0) {
      setQuickBuy(prev => ({ ...prev, message: 'Enter a valid number of shares' }));
      return;
    }
    if (!quickBuy.currentPrice) {
      setQuickBuy(prev => ({ ...prev, message: 'Price not available' }));
      return;
    }

    setQuickBuy(prev => ({ ...prev, submitting: true, message: '' }));

    try {
      const res = await fetch('/api/proxy?endpoint=transactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId.toString()
        },
        body: JSON.stringify({
          ticker: quickBuy.symbol,
          type: 'BUY',
          shares,
          price: quickBuy.currentPrice,
          date: new Date().toLocaleDateString('en-CA'),
          commission: 0
        })
      });

      if (res.ok) {
        setQuickBuy(prev => ({
          ...prev,
          submitting: false,
          success: true,
          message: `✓ Bought ${shares} share${shares !== 1 ? 's' : ''} of ${quickBuy.symbol}`
        }));
        // Close after short delay so user sees confirmation
        setTimeout(() => setQuickBuy(QUICK_BUY_DEFAULT), 1500);
      } else {
        const err = await res.json();
        setQuickBuy(prev => ({
          ...prev,
          submitting: false,
          message: err.error || 'Transaction failed'
        }));
      }
    } catch {
      setQuickBuy(prev => ({
        ...prev,
        submitting: false,
        message: 'Network error — please try again'
      }));
    }
  };

  const gainers = watchlist.filter(s => s.change > 0);
  const losers = watchlist.filter(s => s.change < 0);

  const estimatedTotal =
    quickBuy.currentPrice && quickBuy.shares
      ? (parseFloat(quickBuy.shares) || 0) * quickBuy.currentPrice
      : null;

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
                      <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">Symbol</th>
                      <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">Company Name</th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Price</th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Change</th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">Volume</th>
                      {/* Only show action column if logged in */}
                      {userId && (
                        <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                          Action
                        </th>
                      )}
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
                        {userId && (
                          <td className="px-6 py-4 text-right">
                            <button
                              onClick={() => openQuickBuy(stock)}
                              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded transition"
                            >
                              Quick Buy
                            </button>
                          </td>
                        )}
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

      {/* ==================== QUICK BUY MODAL ==================== */}
      {quickBuy.open && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/50 z-40"
            onClick={closeQuickBuy}
          />

          {/* Modal */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
              className="bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-700 shadow-xl w-full max-w-sm"
              onClick={e => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-200 dark:border-neutral-700">
                <div>
                  <h2 className="text-lg font-bold text-black dark:text-white">
                    Buy {quickBuy.symbol}
                  </h2>
                  <p className="text-sm text-neutral-500 dark:text-neutral-400">
                    {watchlist.find(s => s.symbol === quickBuy.symbol)?.name}
                  </p>
                </div>
                <button
                  onClick={closeQuickBuy}
                  disabled={quickBuy.submitting}
                  className="text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200 transition text-xl leading-none disabled:opacity-50"
                >
                  ✕
                </button>
              </div>

              {/* Modal Body */}
              <form onSubmit={handleQuickBuySubmit} className="px-6 py-5 space-y-4">

                {/* Current Price */}
                <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 flex items-center justify-between">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Current Price</span>
                  <span className="text-lg font-bold text-black dark:text-white">
                    {quickBuy.priceLoading ? (
                      <span className="text-sm text-neutral-400 dark:text-neutral-500">Fetching...</span>
                    ) : quickBuy.currentPrice ? (
                      `$${quickBuy.currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}`
                    ) : (
                      <span className="text-sm text-red-500">Unavailable</span>
                    )}
                  </span>
                </div>

                {/* Shares Input */}
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Shares <span className="text-red-500">*</span>
                  </label>
                  <input
                    ref={sharesInputRef}
                    type="number"
                    value={quickBuy.shares}
                    onChange={e => setQuickBuy(prev => ({ ...prev, shares: e.target.value, message: '' }))}
                    placeholder="0.00"
                    step="0.01"
                    min="0.01"
                    disabled={quickBuy.submitting || quickBuy.success}
                    className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white placeholder-neutral-400 focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  />
                </div>

                {/* Estimated Total */}
                {estimatedTotal !== null && estimatedTotal > 0 && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-neutral-600 dark:text-neutral-400">Estimated Total</span>
                    <span className="font-semibold text-black dark:text-white">
                      ${estimatedTotal.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </span>
                  </div>
                )}

                {/* Message */}
                {quickBuy.message && (
                  <div className={`p-3 rounded text-sm ${
                    quickBuy.success
                      ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                      : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                  }`}>
                    {quickBuy.message}
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3 pt-1">
                  <button
                    type="button"
                    onClick={closeQuickBuy}
                    disabled={quickBuy.submitting}
                    className="flex-1 px-4 py-2 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded-lg hover:bg-neutral-200 dark:hover:bg-neutral-700 transition font-medium disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={quickBuy.submitting || quickBuy.success || !quickBuy.currentPrice || !quickBuy.shares}
                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium disabled:opacity-50"
                  >
                    {quickBuy.submitting ? 'Buying...' : quickBuy.success ? '✓ Done' : 'Confirm Buy'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </>
      )}
    </div>
  );
}