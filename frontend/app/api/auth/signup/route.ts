export async function POST(request: Request) {
  try {
    const { email, password, displayName } = await request.json();

    if (!email || !password || !displayName) {
      return Response.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const backendUrl = 'http://localhost:5001/api/auth/signup';
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, displayName }),
    });

    const data = await response.json();

    if (!response.ok) {
      return Response.json(data, { status: response.status });
    }

    return Response.json(data, { status: response.status });
  } catch (error) {
    console.error('Signup proxy error:', error);
    return Response.json({ error: 'Failed to signup' }, { status: 500 });
  }
}
