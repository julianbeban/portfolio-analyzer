export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const endpoint = searchParams.get('endpoint');
  
  // Get headers from the original request
  const headers = new Headers();
  const userId = request.headers.get('X-User-ID');
  if (userId) {
    headers.set('X-User-ID', userId);
  }

  if (!endpoint) {
    return Response.json({ error: 'No endpoint specified' }, { status: 400 });
  }

  try {
    const backendUrl = `http://localhost:5001/api/${endpoint}`;
    const response = await fetch(backendUrl, {headers});
    const data = await response.json();
    
    return Response.json(data, { status: response.status });
  } catch (error) {
    console.error('Proxy error:', error);
    return Response.json({ error: 'Failed to fetch from backend' }, { status: 500 });
  }
}
