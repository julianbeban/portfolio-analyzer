'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { logout } from '@/app/lib/auth-actions';

interface User {
  id: number;
  email: string;
  displayName: string;
}

export default function UserMenu() {
  const [user, setUser] = useState<User | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkSession = async () => {
      try {
        const response = await fetch('/api/auth/session');
        if (response.ok) {
          const data = await response.json();
          setUser(data.user);
        }
      } catch (error) {
        console.error('Failed to check session:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkSession();
  }, []);

  if (isLoading) {
    return null;
  }

  if (!user) {
    return (
      <Link href="/signin" className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition">
        Sign in
      </Link>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-neutral-200 dark:bg-neutral-800 text-black dark:text-white hover:bg-neutral-300 dark:hover:bg-neutral-700 transition"
      >
        <span className="text-sm font-medium truncate max-w-[150px]">{user.displayName}</span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-neutral-900 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700 py-1 z-50">
          <div className="px-4 py-2 border-b border-neutral-200 dark:border-neutral-700">
            <p className="text-sm font-medium text-black dark:text-white">{user.displayName}</p>
            <p className="text-xs text-neutral-600 dark:text-neutral-400">{user.email}</p>
          </div>
          <Link
            href="/settings"
            className="block px-4 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
          >
            Settings
          </Link>
          <button
            onClick={async () => {
              await logout();
            }}
            className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  );
}
