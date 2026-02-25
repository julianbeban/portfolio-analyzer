import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="border-t border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-950 mt-16">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center text-sm text-neutral-600 dark:text-neutral-400">
          <p>&copy; 2026 Portfolio Analyzer. Market data delayed by 15 minutes.</p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <Link href="#" className="hover:text-black dark:hover:text-white transition">Privacy Policy</Link>
            <Link href="#" className="hover:text-black dark:hover:text-white transition">Terms of Service</Link>
            <Link href="#" className="hover:text-black dark:hover:text-white transition">Contact</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
