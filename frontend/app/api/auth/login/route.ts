export async function POST(request: Request) {
  try {
    const JSON_DATA = await request.json();
    const { email, password } = JSON_DATA;

    if (!email || !password) {
      return Response.json({ error: 'Missing email or password' }, { status: 400 });
    }

    const backendUrl = 'http://localhost:5001/api/auth/login';
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      return Response.json(data, { status: response.status });
    }

    return Response.json(data, { status: response.status });
  } catch (error) {
    console.error('Login proxy error:', error);
    return Response.json({ error: 'Failed to login' }, { status: 500 });
  }
}
