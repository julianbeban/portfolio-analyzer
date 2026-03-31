'use server';

import { getSession, saveSession, deleteSession } from '@/app/lib/session';
import { redirect } from 'next/navigation';

interface LoginResponse {
  success: boolean;
  error?: string;
}

export async function login(email: string, password: string): Promise<LoginResponse> {
  try {
    const response = await fetch('http://localhost:5001/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data.error || 'Login failed',
      };
    }

    // Store user data in session
    await saveSession({
      user: {
        id: data.user.id,
        email: data.user.email,
        displayName: data.user.displayName,
      },
      isLoggedIn: true,
    });

    return { success: true };
  } catch (error) {
    console.error('Login error:', error);
    return {
      success: false,
      error: 'An error occurred during login',
    };
  }
}

export async function signup(
  email: string,
  password: string,
  displayName: string
): Promise<LoginResponse> {
  try {
    const response = await fetch('http://localhost:5001/api/auth/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, displayName }),
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data.error || 'Signup failed',
      };
    }

    // Store user data in session
    await saveSession({
      user: {
        id: data.user.id,
        email: data.user.email,
        displayName: data.user.displayName,
      },
      isLoggedIn: true,
    });

    return { success: true };
  } catch (error) {
    console.error('Signup error:', error);
    return {
      success: false,
      error: 'An error occurred during signup',
    };
  }
}

export async function logout() {
  await deleteSession();
  redirect('/signin');
}
