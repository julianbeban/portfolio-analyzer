'use client';

import { useState, useEffect, FormEvent, ChangeEvent, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { getLocalDateString } from '@/app/lib/date';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';

interface Transaction {
  id?: number;
  ticker: string;
  type: 'BUY' | 'SELL';
  shares: number;
  price: number;
  date: string;
  commission: number;
  notes?: string;
  total?: number;
}

interface ColumnDetection {
  ticker: number | null;
  transaction_type: number | null;
  shares: number | null;
  price: number | null;
  date: number | null;
  commission: number | null;
  notes: number | null;
}

interface ImportResult {
  success: boolean;
  imported?: number;
  errors?: string[];
  message?: string;
}

const COLUMN_LABELS = {
  ticker: 'Ticker/Symbol',
  transaction_type: 'Transaction Type (BUY/SELL)',
  shares: 'Shares/Quantity',
  price: 'Price',
  date: 'Date',
  commission: 'Commission (Optional)',
  notes: 'Notes (Optional)'
};

type TabType = 'add' | 'import' | 'history';

export default function TransactionsPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [userId, setUserId] = useState<number | null>(null);
  const hasFetched = useRef(false);
  
  // Tab state
  const [activeTab, setActiveTab] = useState<TabType>('history');
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState('');
  const [autofillLoading, setAutofillLoading] = useState(false);
  
  // Add transaction form
  const [formData, setFormData] = useState<Transaction>({
    ticker: '',
    type: 'BUY',
    shares: 0,
    price: 0,
    date: getLocalDateString(),
    commission: 0,
    notes: ''
  });

  // Import CSV form
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string[][]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [mapping, setMapping] = useState<ColumnDetection | null>(null);
  const [importLoading, setImportLoading] = useState(false);
  const [importMessage, setImportMessage] = useState('');
  const [importStage, setImportStage] = useState<'upload' | 'map' | 'result'>('upload');

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;
    
    const checkAuth = async () => {
      try {
        const userRes = await fetch('/api/user');
        const { userId: id } = await userRes.json();

        if (!id) {
          router.push('/signin');
          return;
        }

        setUserId(id);
        fetchTransactions(id);
      } catch (error) {
        console.error('Auth error:', error);
        router.push('/signin');
      }
    };

    checkAuth();
  }, [router]);

  const fetchTransactions = async (id: number) => {
    try {
      setLoading(true);
      const res = await fetch('/api/proxy?endpoint=transactions', {
        headers: { 'X-User-ID': id.toString() }
      });

      if (res.ok) {
        const data = await res.json();
        setTransactions(data);
      }
    } catch (error) {
      console.error('Error fetching transactions:', error);
    } finally {
      setLoading(false);
    }
  };

  // ==================== ADD TRANSACTION ====================

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    
    setFormData(prev => ({
      ...prev,
      [name]: ['shares', 'price', 'commission'].includes(name) ? parseFloat(value) || 0 : value
    }));
  };

  const handleAutofillPrice = async () => {
    if (!formData.ticker || !formData.date) {
      setMessage('Please select a ticker and date first');
      return;
    }

    setAutofillLoading(true);
    setMessage('');

    try {
      const res = await fetch('/api/proxy?endpoint=price-history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: formData.ticker.toUpperCase(),
          date: formData.date
        })
      });

      if (res.ok) {
        const data = await res.json();
        setFormData(prev => ({
          ...prev,
          price: data.price
        }));
        setMessage(`✓ Price filled: $${data.price}`);
      } else {
        const error = await res.json();
        setMessage(`Unable to find price data: ${error.error}`);
      }
    } catch (error) {
      setMessage(`Error fetching price: ${error}`);
    } finally {
      setAutofillLoading(false);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setMessage('');

    try {
      if (!userId) {
        setMessage('Please log in to add transactions');
        setSubmitting(false);
        return;
      }

      if (!formData.ticker || formData.shares <= 0 || formData.price <= 0) {
        setMessage('Please fill in all required fields with positive values');
        setSubmitting(false);
        return;
      }

      const res = await fetch('/api/proxy?endpoint=transactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId.toString()
        },
        body: JSON.stringify(formData)
      });

      if (res.ok) {
        setMessage('✓ Transaction added successfully');
        setFormData({
          ticker: '',
          type: 'BUY',
          shares: 0,
          price: 0,
          date: new Date().toLocaleDateString('en-CA'),
          commission: 0,
          notes: ''
        });
        
        setTimeout(() => {
          fetchTransactions(userId);
          setActiveTab('history');
        }, 500);
      } else {
        const error = await res.json();
        setMessage(`Error: ${error.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error}`);
    } finally {
      setSubmitting(false);
    }
  };

  // ==================== IMPORT CSV ====================

  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setImportMessage('');

    try {
      const text = await selectedFile.text();
      const lines = text.split('\n').slice(0, 6);
      const rows = lines.map(line => line.split(',').map(cell => cell.trim()));
      
      setColumns(rows[0]);
      setPreview(rows.slice(1));

      const detectedMapping = detectColumns(rows[0]);
      setMapping(detectedMapping);
      setImportStage('map');
    } catch (error) {
      setImportMessage(`Failed to read file: ${error}`);
    }
  };

  const detectColumns = (headers: string[]): ColumnDetection => {
    const mapping: ColumnDetection = {
      ticker: null,
      transaction_type: null,
      shares: null,
      price: null,
      date: null,
      commission: null,
      notes: null
    };

    const lowerHeaders = headers.map(h => h.toLowerCase());

    for (let i = 0; i < lowerHeaders.length; i++) {
      const header = lowerHeaders[i];
      
      if (!mapping.ticker && /ticker|symbol|stock/.test(header)) {
        mapping.ticker = i;
      } else if (!mapping.transaction_type && /type|transaction|action|side/.test(header)) {
        mapping.transaction_type = i;
      } else if (!mapping.shares && /shares|quantity|qty|amount/.test(header)) {
        mapping.shares = i;
      } else if (!mapping.price && /price|rate|cost/.test(header)) {
        mapping.price = i;
      } else if (!mapping.date && /date|trade/.test(header)) {
        mapping.date = i;
      } else if (!mapping.commission && /commission|fee/.test(header)) {
        mapping.commission = i;
      } else if (!mapping.notes && /notes|memo|description/.test(header)) {
        mapping.notes = i;
      }
    }

    return mapping;
  };

  const handleMappingChange = (field: keyof ColumnDetection, value: number | null) => {
    setMapping(prev => prev ? { ...prev, [field]: value } : null);
  };

  const handleImport = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!file || !mapping || !userId) return;

    const required = ['ticker', 'transaction_type', 'shares', 'price', 'date'];
    const incomplete = required.filter(field => mapping[field as keyof ColumnDetection] === null);
    
    if (incomplete.length > 0) {
      setImportMessage(`Please map required columns: ${incomplete.join(', ')}`);
      return;
    }

    setImportLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('columnMapping', JSON.stringify(mapping));

      const response = await fetch('/api/proxy?endpoint=import/csv', {
        method: 'POST',
        headers: {
          'X-User-ID': userId.toString()
        },
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setImportMessage(`✓ ${data.message}`);
        setImportStage('result');
        
        setTimeout(() => {
          setFile(null);
          setPreview([]);
          setColumns([]);
          setMapping(null);
          setImportStage('upload');
          fetchTransactions(userId);
          setActiveTab('history');
        }, 2000);
      } else {
        setImportMessage(data.error || 'Import failed');
      }
    } catch (error) {
      setImportMessage(`Error: ${error}`);
    } finally {
      setImportLoading(false);
    }
  };

  // Prevent rendering until auth is verified
  if (!userId) {
    return null;
  }

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black">
      <Header />

      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-black dark:text-white">Transactions</h1>
          <p className="text-neutral-600 dark:text-neutral-400">Manage your portfolio trades</p>
        </div>

        {/* Tabs */}
        <div className="mb-8 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex gap-4">
            {(['add', 'import', 'history'] as TabType[]).map(tab => (
              <button
                key={tab}
                onClick={() => {
                  setActiveTab(tab);
                  setMessage('');
                  setImportMessage('');
                }}
                className={`px-4 py-3 font-medium transition border-b-2 ${
                  activeTab === tab
                    ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-200'
                }`}
              >
                {tab === 'add' && '+ Add Transaction'}
                {tab === 'import' && '📥 Import CSV'}
                {tab === 'history' && '📋 History'}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
          
          {/* ADD TRANSACTION TAB */}
          {activeTab === 'add' && (
            <form onSubmit={handleSubmit} className="space-y-6">
              <h2 className="text-xl font-semibold text-black dark:text-white">Add New Transaction</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Ticker <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    name="ticker"
                    value={formData.ticker}
                    onChange={handleInputChange}
                    placeholder="e.g., AAPL"
                    className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:ring-2 focus:ring-blue-500 uppercase"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Type <span className="text-red-500">*</span>
                  </label>
                  <select
                    name="type"
                    value={formData.type}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="BUY">Buy</option>
                    <option value="SELL">Sell</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Shares <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    name="shares"
                    value={formData.shares || ''}
                    onChange={handleInputChange}
                    placeholder="0.00"
                    step="0.01"
                    min="0"
                    className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Price <span className="text-red-500">*</span>
                    {formData.type === 'SELL' && <span className="text-xs text-neutral-500 ml-2">(current/market price)</span>}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      name="price"
                      value={formData.price || ''}
                      onChange={handleInputChange}
                      placeholder="0.00"
                      step="0.01"
                      min="0"
                      disabled={formData.type === 'SELL'}
                      className="flex-1 px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    />
                    <button
                      type="button"
                      onClick={handleAutofillPrice}
                      disabled={autofillLoading || !formData.ticker || !formData.date}
                      className="px-3 py-2 bg-neutral-200 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200 rounded hover:bg-neutral-300 dark:hover:bg-neutral-600 transition font-medium disabled:opacity-50 whitespace-nowrap"
                    >
                      {autofillLoading ? 'Loading...' : formData.type === 'SELL' ? 'Fetch Market' : 'Auto-fill'}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Date <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="date"
                    name="date"
                    value={formData.date}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Commission
                  </label>
                  <input
                    type="number"
                    name="commission"
                    value={formData.commission || ''}
                    onChange={handleInputChange}
                    placeholder="0.00"
                    step="0.01"
                    min="0"
                    className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Notes
                </label>
                <textarea
                  name="notes"
                  value={formData.notes}
                  onChange={handleInputChange}
                  placeholder="Optional notes about this transaction"
                  rows={3}
                  className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {message && (
                <div className={`p-3 rounded ${
                  message.startsWith('✓')
                    ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                    : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                }`}>
                  {message}
                </div>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium disabled:opacity-50"
              >
                {submitting ? 'Adding...' : 'Add Transaction'}
              </button>
            </form>
          )}

          {/* IMPORT CSV TAB */}
          {activeTab === 'import' && (
            <div>
              <h2 className="text-xl font-semibold text-black dark:text-white mb-6">Import CSV File</h2>
              
              {importStage === 'upload' && (
                <div>
                  <div className="border-2 border-dashed border-neutral-300 dark:border-neutral-600 rounded-lg p-8 text-center cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-800 transition">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileSelect}
                      className="hidden"
                      id="csv-input"
                    />
                    <label htmlFor="csv-input" className="cursor-pointer block">
                      <div className="text-4xl mb-2">📄</div>
                      <p className="font-semibold text-black dark:text-white mb-1">Choose CSV file</p>
                      <p className="text-sm text-neutral-500 dark:text-neutral-400">or drag and drop</p>
                    </label>
                  </div>

                  {file && (
                    <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
                      <p className="text-sm text-green-700 dark:text-green-300">✓ {file.name} selected</p>
                    </div>
                  )}

                  {importMessage && !importMessage.startsWith('✓') && (
                    <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
                      <p className="text-sm text-red-700 dark:text-red-300">{importMessage}</p>
                    </div>
                  )}
                </div>
              )}

              {importStage === 'map' && mapping && (
                <form onSubmit={handleImport}>
                  <div className="space-y-6 mb-8">
                    <div>
                      <h3 className="text-lg font-semibold text-black dark:text-white mb-4">Map CSV Columns</h3>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">Select which column contains each field</p>

                      <div className="space-y-4">
                        {Object.entries(COLUMN_LABELS).map(([key, label]) => {
                          const isRequired = !['commission', 'notes'].includes(key);
                          return (
                            <div key={key}>
                              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                                {label}
                                {isRequired && <span className="text-red-500 ml-1">*</span>}
                              </label>
                              <select
                                value={mapping[key as keyof ColumnDetection] ?? ''}
                                onChange={(e) => handleMappingChange(key as keyof ColumnDetection, e.target.value ? parseInt(e.target.value) : null)}
                                className="w-full px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded text-black dark:text-white focus:ring-2 focus:ring-blue-500"
                              >
                                <option value="">-- Select Column --</option>
                                {columns.map((col, idx) => (
                                  <option key={idx} value={idx}>{col}</option>
                                ))}
                              </select>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-md font-semibold text-black dark:text-white mb-2">Preview</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead className="border-b border-neutral-200 dark:border-neutral-700">
                            <tr>
                              {columns.map((col, idx) => (
                                <th key={idx} className="px-3 py-2 text-left text-neutral-700 dark:text-neutral-300 whitespace-nowrap">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {preview.slice(0, 3).map((row, ridx) => (
                              <tr key={ridx} className="border-b border-neutral-100 dark:border-neutral-800">
                                {row.map((cell, cidx) => (
                                  <td key={cidx} className="px-3 py-2 text-neutral-700 dark:text-neutral-300 whitespace-nowrap">
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>

                  {importMessage && !importMessage.startsWith('✓') && (
                    <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
                      <p className="text-sm text-red-700 dark:text-red-300">{importMessage}</p>
                    </div>
                  )}

                  <div className="flex gap-4">
                    <button
                      type="button"
                      onClick={() => {
                        setImportStage('upload');
                        setFile(null);
                        setPreview([]);
                      }}
                      className="px-4 py-2 bg-neutral-200 dark:bg-neutral-700 text-black dark:text-white rounded hover:bg-neutral-300 dark:hover:bg-neutral-600 transition"
                    >
                      Back
                    </button>
                    <button
                      type="submit"
                      disabled={importLoading}
                      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition disabled:opacity-50"
                    >
                      {importLoading ? 'Importing...' : 'Import Transactions'}
                    </button>
                  </div>
                </form>
              )}

              {importStage === 'result' && importMessage && (
                <div className={`p-6 rounded-lg ${
                  importMessage.startsWith('✓')
                    ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                    : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                }`}>
                  <div className={`text-lg font-semibold ${
                    importMessage.startsWith('✓') ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'
                  }`}>
                    {importMessage}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* HISTORY TAB */}
          {activeTab === 'history' && (
            <div>
              <h2 className="text-xl font-semibold text-black dark:text-white mb-6">Transaction History</h2>
              
              {loading ? (
                <div className="p-8 text-center">
                  <p className="text-neutral-600 dark:text-neutral-400">Loading transactions...</p>
                </div>
              ) : transactions.length === 0 ? (
                <div className="p-8 text-center">
                  <p className="text-neutral-600 dark:text-neutral-400">No transactions yet. Start by adding one!</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800">
                      <tr>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Date</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Ticker</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Type</th>
                        <th className="px-6 py-4 text-right text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Shares</th>
                        <th className="px-6 py-4 text-right text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Price</th>
                        <th className="px-6 py-4 text-right text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Total</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase">Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {transactions.map((trans, idx) => (
                        <tr key={idx} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition">
                          <td className="px-6 py-4 text-sm text-neutral-700 dark:text-neutral-300">
                            {new Date(trans.date).toLocaleDateString()}
                          </td>
                          <td className="px-6 py-4 font-semibold text-black dark:text-white">{trans.ticker}</td>
                          <td className="px-6 py-4">
                            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                              trans.type === 'BUY'
                                ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                                : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
                            }`}>
                              {trans.type}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-right text-neutral-700 dark:text-neutral-300">{trans.shares}</td>
                          <td className="px-6 py-4 text-right text-neutral-700 dark:text-neutral-300">${trans.price.toFixed(2)}</td>
                          <td className="px-6 py-4 text-right font-semibold text-black dark:text-white">
                            ${((trans.shares * trans.price) + trans.commission).toFixed(2)}
                          </td>
                          <td className="px-6 py-4 text-sm text-neutral-600 dark:text-neutral-400 max-w-xs truncate">
                            {trans.notes || '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
