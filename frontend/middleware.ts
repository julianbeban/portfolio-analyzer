import { NextRequest, NextResponse } from 'next/server';
import { getIronSession } from 'iron-session';
import { sessionConfig } from '@/app/lib/session';
import type { SessionData } from '@/app/lib/session';
import { cookies } from 'next/headers'

// const sessionConfig = {
//   password: process.env.SESSION_SECRET || 'your-secret-key-min-32-characters-long-change-this',
//   cookieName: 'auth-session',
//   cookieOptions: {
//     secure: process.env.NODE_ENV === 'production',
//     httpOnly: true,
//     sameSite: 'lax' as const,
//     maxAge: 60 * 60 * 24 * 7,
//   },
// };

// Protected routes that require authentication
const protectedRoutes = ['/holdings', '/analysis', '/watchlist', '/settings'];

// Public routes that should redirect to dashboard if authenticated
const publicAuthRoutes = ['/signin', '/signup'];

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;

  // Get session
  // inside middleware:
    const cookieStore = await cookies()
    const session = await getIronSession<SessionData>(cookieStore, sessionConfig)
  const isLoggedIn = !!session.user;

  // Redirect authenticated users away from auth pages
  if (publicAuthRoutes.includes(path) && isLoggedIn) {
    return NextResponse.redirect(new URL('/', request.url));
  }

  // Redirect unauthenticated users to signin
  if (protectedRoutes.includes(path) && !isLoggedIn) {
    return NextResponse.redirect(new URL('/signin', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    // Match all routes except static files and such
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
