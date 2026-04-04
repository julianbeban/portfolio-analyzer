import { getSession } from '@/app/lib/session';

export async function GET() {
  const session = await getSession();
  return Response.json({
    userId: session?.user?.id || null
  });
}