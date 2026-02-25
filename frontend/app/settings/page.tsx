'use client';

import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black flex flex-col">
      <Header />
      
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-black dark:text-white mb-2">Settings</h1>
          <p className="text-neutral-600 dark:text-neutral-400">Manage your preferences and configuration</p>
        </div>

        {/* Placeholder content */}
        <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-8 text-center">
          <p className="text-neutral-600 dark:text-neutral-400">Settings content coming soon...</p>
        </div>
      </main>

      <Footer />
    </div>
  );
}
