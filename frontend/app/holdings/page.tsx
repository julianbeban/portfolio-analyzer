'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';

interface Holding {
  symbol: string;
  shares: number;
  avgCost: number;
  current: number;
  gainAmount: number;
  gainPercent: number;
  totalValue: number;
}

export default function HoldingsPage() {
  const router = useRouter();
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
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
        fetchHoldings(id);
      } catch (error) {
        console.error('Auth error:', error);
        router.push('/signin');
      }
    };

    initializeAuth();
  }, [router]);

  const fetchHoldings = async (id: number) => {
    try {
      setLoading(true);
      const holdingsRes = await fetch('/api/proxy?endpoint=holdings', {
        headers: {
          'X-User-ID': id.toString()
        }
      });

      if (holdingsRes.ok) {
        const holdingsData = await holdingsRes.json();
        setHoldings(holdingsData);
        setError(''); // Clear error on successful fetch
      } else {
        setError('Failed to load holdings');
        setHoldings([]);
      }
    } catch (err) {
      console.error('Error fetching holdings:', err);
      setError('Error loading holdings');
      setHoldings([]);
    } finally {
      setLoading(false);
    }
  };

  const totalValue = holdings.reduce((sum, h) => sum + h.totalValue, 0);
  const totalGain = holdings.reduce((sum, h) => sum + h.gainAmount, 0);
  const totalGainPercent = totalValue > 0 ? (totalGain / (totalValue - totalGain)) * 100 : 0;

  // Prevent rendering until auth is verified
  if (!userId) {
    return null;
  }

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black flex flex-col">
      <Header />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 w-full">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-black dark:text-white mb-2">Your Holdings</h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            {holdings.length} {holdings.length === 1 ? 'stock' : 'stocks'} in your portfolio
          </p>
        </div>

        {error && holdings.length > 0 && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        {loading ? (
          <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-8 text-center">
            <p className="text-neutral-600 dark:text-neutral-400">Loading holdings...</p>
          </div>
        ) : holdings.length === 0 ? (
          <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-8 text-center">
            <p className="text-neutral-600 dark:text-neutral-400">No holdings yet. Add your first transaction to get started.</p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">Total Portfolio Value</p>
                <p className="text-2xl font-bold text-black dark:text-white">
                  ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </p>
              </div>

              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">Total Gain/Loss</p>
                <div className="flex items-baseline gap-2">
                  <p className={`text-2xl font-bold ${
                    totalGain >= 0 
                      ? 'text-green-600 dark:text-green-400' 
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    ${totalGain.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </p>
                  <p className={`text-sm font-medium ${
                    totalGainPercent >= 0 
                      ? 'text-green-600 dark:text-green-400' 
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {totalGainPercent > 0 ? '+' : ''}{totalGainPercent.toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">Holdings</p>
                <p className="text-2xl font-bold text-black dark:text-white">{holdings.length}</p>
              </div>
            </div>

            {/* Holdings Table */}
            <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/50">
                      <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Symbol
                      </th>
                      <th className="px-6 py-3 text-left text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Shares
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Avg Cost
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Current Price
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Total Value
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Gain/Loss
                      </th>
                      <th className="px-6 py-3 text-right text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        Return %
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {holdings.map((holding) => (
                      <tr
                        key={holding.symbol}
                        className="border-b border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition"
                      >
                        <td className="px-6 py-4 text-sm font-semibold text-black dark:text-white">
                          {holding.symbol}
                        </td>
                        <td className="px-6 py-4 text-sm text-neutral-600 dark:text-neutral-400">
                          {holding.shares.toLocaleString('en-US', { maximumFractionDigits: 2 })}
                        </td>
                        <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                          ${holding.avgCost.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </td>
                        <td className="px-6 py-4 text-sm text-right text-neutral-600 dark:text-neutral-400">
                          ${holding.current.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </td>
                        <td className="px-6 py-4 text-sm font-semibold text-right text-black dark:text-white">
                          ${holding.totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </td>
                        <td className={`px-6 py-4 text-sm font-semibold text-right ${
                          holding.gainAmount >= 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}>
                          {holding.gainAmount >= 0 ? '+' : ''}
                          ${holding.gainAmount.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </td>
                        <td className={`px-6 py-4 text-sm font-semibold text-right ${
                          holding.gainPercent >= 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}>
                          {holding.gainPercent > 0 ? '+' : ''}{holding.gainPercent.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}