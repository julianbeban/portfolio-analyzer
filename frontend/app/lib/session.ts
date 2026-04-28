import { getIronSession } from 'iron-session';
import { cookies } from 'next/headers';

export interface SessionData {
  user?: {
    id: number;
    email: string;
    displayName: string;
  };
  isLoggedIn?: boolean;
}

export const sessionConfig = {
  password: process.env.SESSION_SECRET || 'your-secret-key-min-32-characters-long-change-this',
  cookieName: 'auth-session',
  cookieOptions: {
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    sameSite: 'lax' as const,
    maxAge: 60 * 60 * 8, // 8 hours
  },
};

export async function getSession() {
  const cookieStore = await cookies();
  const session = await getIronSession<SessionData>(cookieStore, sessionConfig);
  return session;
}

export async function saveSession(data: SessionData) {
  const session = await getSession();
  session.user = data.user;
  session.isLoggedIn = !!data.user;
  await session.save();
}

export async function deleteSession() {
  const session = await getSession();
  session.destroy();
}
