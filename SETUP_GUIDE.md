# Setup Instructions

## Backend (Flask)

Open a terminal and run:
```bash
cd /Users/bryanluna/repos/portfolio-analyzer/backend
pip install -r requirements.txt
python app.py
```

The backend will run on `http://localhost:5001`

## Frontend (Next.js)

Open another terminal and run:
```bash
cd /Users/bryanluna/repos/portfolio-analyzer/frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:3000`

## Environment Setup

### Backend
Create/update `.env` file in `/backend`:
```
DATABASE_URL=sqlite:///portfolio.db
JWT_SECRET_KEY=your-secret-key-change-in-production-12345
FLASK_ENV=development
```

### Frontend
Create/update `.env.local` file in `/frontend`:
```
SESSION_SECRET=your-super-secret-session-key-min-32-characters-long
NODE_ENV=development
```

## Authentication System

The app implements **session-based authentication** using iron-session with encrypted HTTP-only cookies (following Next.js recommendations).

### Key Features
- ✅ Secure HTTP-only encrypted cookies (no localStorage)
- ✅ Server-side session management
- ✅ Middleware-based route protection
- ✅ Automatic redirect for authenticated users
- ✅ Server Actions for auth operations
- ✅ Password hashing with Werkzeug (backend)

### Architecture

**Backend (Flask)**:
- User model with password hashing
- RESTful auth endpoints
- SQLite database for user storage

**Frontend (Next.js App Router)**:
- `/app/lib/session.ts` - Session configuration and management
- `/app/lib/auth-actions.ts` - Server Actions for login/signup/logout  
- `/middleware.ts` - Route protection middleware
- `/app/api/auth/session/route.ts` - Session status endpoint
- `/app/signin/page.tsx` - Login page
- `/app/signup/page.tsx` - Registration page

### Protected Routes
- `/holdings` - Requires authentication
- `/analysis` - Requires authentication
- `/watchlist` - Requires authentication
- `/settings` - Requires authentication

### Public Routes
- `/` - Home
- `/signin` - Login (redirects to home if authenticated)
- `/signup` - Registration (redirects to home if authenticated)

## Testing Authentication

1. **Create Account**:
   - Go to `http://localhost:3000/signup`
   - Fill in email, password (min 8 chars), and name
   - You'll be redirected to home after successful signup

2. **Login**:
   - Go to `http://localhost:3000/signin`
   - Enter credentials
   - You'll be redirected to home after successful login

3. **Session Persistence**:
   - After login, refresh the page
   - Session persists due to encrypted HTTP-only cookie

4. **Logout**:
   - Logout functionality in settings page destroys session

## API Endpoints

### Authentication Endpoints

- `POST /api/auth/signup` - Register new user
  ```json
  { "email": "user@example.com", "password": "password123", "displayName": "John Doe" }
  ```

- `POST /api/auth/login` - Login user
  ```json
  { "email": "user@example.com", "password": "password123" }
  ```

- `GET /api/auth/me` - Get current user (requires JWT token)

- `GET /api/auth/session` - Check session status (client-side)

### Portfolio Endpoints

- `GET /api/portfolio` - Portfolio stats
- `GET /api/holdings` - Your holdings
- `GET /api/watchlist` - Market watchlist
- `POST /api/analyze` - Portfolio analysis

## Security Features

1. **HTTP-Only Cookies**: Session stored in secure, httpOnly cookies (immune to XSS)
2. **CSRF Protection**: Same-site cookie policy
3. **Encryption**: Sessions encrypted on the client-side before storage
4. **Password Hashing**: Werkzeug's PBKDF2 hashing
5. **Route Protection**: Middleware validates all requests
6. **Server Actions**: Form handling on server-side only

## Best Practices Implemented

Following [Next.js Authentication Guide](https://nextjs.org/docs/pages/guides/authentication):
- Session management with encrypted cookies
- Middleware for route protection (optimistic checks)
- Server Actions for auth operations
- Type-safe session data
- Proper error handling and validation
- Environment-based security settings

## Development vs Production

When deploying to production:
1. Set `SESSION_SECRET` to a strong, random 32+ character string
2. Set `NODE_ENV=production`
3. Use `https://` for secure cookie transmission
4. Enable `secure` flag in cookie options
5. Set proper CORS headers for your domain
