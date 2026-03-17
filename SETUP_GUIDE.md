# Setup Instructions

## Backend (Flask)

Open a terminal and run:
```bash
cd /Users/bryanluna/repos/portfolio-analyzer/backend
source venv/bin/activate
python app.py
```

The backend will run on `http://localhost:5001`

## Frontend (Next.js)

Open another terminal and run:
```bash
cd /Users/bryanluna/repos/portfolio-analyzer/frontend
npm run dev
```

The frontend will run on `http://localhost:3000`

## API Endpoints

- `GET /api/portfolio` - Portfolio stats
- `GET /api/holdings` - Your holdings
- `GET /api/watchlist` - Market watchlist
- `POST /api/analyze` - Portfolio analysis

The frontend now fetches data from the backend!
