import Link from 'next/link';
import UserMenu from './UserMenu';

export default function Header() {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-950">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-blue-800">
            <span className="text-lg font-bold text-white">P</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-black dark:text-white">Portfolio Analyzer</h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">Investment Dashboard</p>
          </div>
        </Link>
        <nav className="hidden md:flex gap-8 text-sm font-medium text-neutral-700 dark:text-neutral-300">
          <Link href="/" className="hover:text-black dark:hover:text-white transition">Dashboard</Link>
          <Link href="/holdings" className="hover:text-black dark:hover:text-white transition">Holdings</Link>
          <Link href="/watchlist" className="hover:text-black dark:hover:text-white transition">Watchlist</Link>
          <Link href="/analysis" className="hover:text-black dark:hover:text-white transition">Analysis</Link>
          <Link href="/settings" className="hover:text-black dark:hover:text-white transition">Settings</Link>
        </nav>
        <div className="flex gap-2">
          <UserMenu/>
        </div>
      </div>
    </header>
  );
}
