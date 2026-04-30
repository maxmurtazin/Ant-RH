import { NextRequest } from "next/server";

function getApiBase() {
  return process.env.API_BASE || process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8084";
}

async function proxy(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  const { path = [] } = await context.params;
  const apiBase = getApiBase();
  const rest = Array.isArray(path) ? path.join("/") : "";
  const qs = req.nextUrl.search ? req.nextUrl.search : "";
  const target = `${apiBase.replace(/\/+$/, "")}/${rest}${qs}`;

  const method = req.method || "GET";
  const headers = new Headers(req.headers);
  headers.delete("host");

  // Preserve content-type and forward body for non-GET/HEAD.
  let body: ArrayBuffer | undefined = undefined;
  if (method !== "GET" && method !== "HEAD") {
    body = await req.arrayBuffer();
  }

  try {
    const upstream = await fetch(target, { method, headers, body });
    const buf = await upstream.arrayBuffer();
    const outHeaders = new Headers(upstream.headers);
    return new Response(buf, { status: upstream.status, headers: outHeaders });
  } catch (e: any) {
    return Response.json(
      {
        error: "backend_unreachable",
        message: e?.message || String(e),
        api_base: apiBase,
      },
      { status: 502 }
    );
  }
}

export async function GET(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(req, context);
}
export async function POST(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(req, context);
}
export async function DELETE(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(req, context);
}
export async function PUT(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(req, context);
}

