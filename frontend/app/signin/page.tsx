'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';
import { login } from '@/app/lib/auth-actions';

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    if (!email || !password) {
      setError('Please fill in all fields');
      setIsLoading(false);
      return;
    }

    try {
      const result = await login(email, password);

      if (!result.success) {
        setError(result.error || 'Login failed');
      } else {
        router.push('/');
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-black flex flex-col">
      <Header />

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-md">
          {/* Logo Section */}
          <div className="text-center mb-8">
            <div className="flex h-16 w-16 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-blue-800 mx-auto mb-4">
              <span className="text-3xl font-bold text-white">P</span>
            </div>
            <h1 className="text-3xl font-bold text-black dark:text-white mb-2">Portfolio Analyzer</h1>
            <p className="text-neutral-600 dark:text-neutral-400">Investment Dashboard</p>
          </div>

          {/* Sign In Card */}
          <div className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-700 p-8 shadow-sm">
            <h2 className="text-2xl font-bold text-black dark:text-white mb-1">Welcome back</h2>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-6">Sign in to your account to continue</p>

            {/* Error Message */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Email Field */}
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-black dark:text-white mb-2">
                  Email Address
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  disabled={isLoading}
                  required
                  className="w-full px-4 py-2.5 rounded-lg border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition disabled:opacity-50"
                />
              </div>

              {/* Password Field */}
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-black dark:text-white mb-2">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  disabled={isLoading}
                  required
                  className="w-full px-4 py-2.5 rounded-lg border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-black dark:text-white placeholder-neutral-400 dark:placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition disabled:opacity-50"
                />
              </div>

              {/* Remember Me & Forgot Password */}
              <div className="flex items-center justify-between text-sm">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    className="w-4 h-4 rounded border-neutral-300 dark:border-neutral-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-neutral-700 dark:text-neutral-300">Remember me</span>
                </label>
                <Link href="#" className="text-blue-600 hover:text-blue-700 dark:hover:text-blue-400 font-medium transition">
                  Forgot password?
                </Link>
              </div>

              {/* Sign In Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-2.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:bg-blue-400 transition mt-6"
              >
                {isLoading ? 'Signing in...' : 'Sign In'}
              </button>
            </form>

            {/* Divider */}
            <div className="flex items-center gap-3 my-6">
              <div className="flex-1 h-px bg-neutral-200 dark:bg-neutral-700"></div>
              <span className="text-sm text-neutral-500 dark:text-neutral-400">OR</span>
              <div className="flex-1 h-px bg-neutral-200 dark:bg-neutral-700"></div>
            </div>

            {/* Sign Up Link */}
            <p className="text-center text-sm text-neutral-600 dark:text-neutral-400">
              Don't have an account?{' '}
              <Link href="/signup" className="text-blue-600 hover:text-blue-700 dark:hover:text-blue-400 font-medium transition">
                Create one
              </Link>
            </p>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
