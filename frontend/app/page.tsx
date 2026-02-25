export default function Home() {
  const portfolioStats = [
    { label: "Total Value", value: "$156,843.50", change: "+5.2%" },
    { label: "Today's Gain/Loss", value: "$3,521.20", change: "+2.3%" },
    { label: "Year to Date", value: "+18.5%", change: "vs S&P 500" },
    { label: "Cash Available", value: "$25,400.00", change: "5.0% of total" },
  ];

  const watchlist = [
    { symbol: "AAPL", name: "Apple Inc.", price: "$234.50", change: "+2.1%", volume: "52.3M" },
    { symbol: "MSFT", name: "Microsoft Corp.", price: "$421.30", change: "-0.5%", volume: "18.9M" },
    { symbol: "GOOGL", name: "Alphabet Inc.", price: "$189.75", change: "+1.8%", volume: "21.4M" },
    { symbol: "AMZN", name: "Amazon.com Inc.", price: "$198.40", change: "+3.2%", volume: "38.7M" },
    { symbol: "TSLA", name: "Tesla Inc.", price: "$285.60", change: "-1.3%", volume: "94.2M" },
    { symbol: "NVDA", name: "NVIDIA Corp.", price: "$876.20", change: "+4.7%", volume: "32.1M" },
  ];

  const holdings = [
    { symbol: "AAPL", shares: "50", avgCost: "$185.30", current: "$234.50", gain: "+26.5%", value: "$11,725" },
    { symbol: "MSFT", shares: "30", avgCost: "$405.20", current: "$421.30", gain: "+4.0%", value: "$12,639" },
    { symbol: "VOO", shares: "25", avgCost: "$418.50", current: "$486.80", gain: "+16.3%", value: "$12,170" },
    { symbol: "BRK.B", shares: "40", avgCost: "$380.40", current: "$412.60", gain: "+8.5%", value: "$16,504" },
  ];

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black">
      {/* Header/Navigation */}
      <header className="border-b border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-950">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-blue-800">
              <span className="text-lg font-bold text-white">P</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-black dark:text-white">Portfolio Analyzer</h1>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">Investment Dashboard</p>
            </div>
          </div>
          <nav className="hidden md:flex gap-8 text-sm font-medium text-neutral-700 dark:text-neutral-300">
            <a href="#" className="hover:text-black dark:hover:text-white transition">Dashboard</a>
            <a href="#" className="hover:text-black dark:hover:text-white transition">Holdings</a>
            <a href="#" className="hover:text-black dark:hover:text-white transition">Watchlist</a>
            <a href="#" className="hover:text-black dark:hover:text-white transition">Analysis</a>
            <a href="#" className="hover:text-black dark:hover:text-white transition">Settings</a>
          </nav>
          <div className="flex gap-2">
            <button className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition">
              Add Stock
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        
        {/* Portfolio Overview Stats */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-black dark:text-white mb-4">Portfolio Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {portfolioStats.map((stat, idx) => (
              <div key={idx} className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">{stat.label}</p>
                <p className="text-2xl font-bold text-black dark:text-white mb-1">{stat.value}</p>
                <p className="text-xs font-medium text-green-600 dark:text-green-400">{stat.change}</p>
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
                        <span className="px-3 py-1 rounded-full bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 text-sm font-medium">
                          {holding.gain}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right font-semibold text-black dark:text-white">{holding.value}</td>
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
                    stock.change.startsWith('+') 
                      ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' 
                      : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
                  }`}>
                    {stock.change}
                  </span>
                </div>
                <p className="text-2xl font-bold text-black dark:text-white mb-2">${stock.price}</p>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">Vol: {stock.volume}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-950 mt-16">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm text-neutral-600 dark:text-neutral-400">
            <p>&copy; 2026 Portfolio Analyzer. Market data delayed by 15 minutes.</p>
            <div className="flex gap-6 mt-4 md:mt-0">
              <a href="#" className="hover:text-black dark:hover:text-white transition">Privacy Policy</a>
              <a href="#" className="hover:text-black dark:hover:text-white transition">Terms of Service</a>
              <a href="#" className="hover:text-black dark:hover:text-white transition">Contact</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
