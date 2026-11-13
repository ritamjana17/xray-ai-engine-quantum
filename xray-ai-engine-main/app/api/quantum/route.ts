import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

/**
 * Secure server-side proxy to the separately hosted PennyLane service.
 * Set QUANTUM_SERVICE_URL to the service base URL (for example https://.../predict).
 * Never expose the service URL/token in client-side code.
 */
export async function POST(req: NextRequest) {
  const serviceUrl = process.env.QUANTUM_SERVICE_URL;
  if (!serviceUrl) {
    return NextResponse.json(
      { error: "Quantum analysis is not configured yet. Set QUANTUM_SERVICE_URL after deploying the Python service." },
      { status: 503 }
    );
  }

  try {
    const body = await req.json();
    if (typeof body.imageBase64 !== "string" || !body.imageBase64.startsWith("data:image/")) {
      return NextResponse.json({ error: "A valid imageBase64 data URL is required." }, { status: 400 });
    }

    const headers = new Headers({ "Content-Type": "application/json" });
    if (process.env.QUANTUM_SERVICE_TOKEN) {
      headers.set("Authorization", `Bearer ${process.env.QUANTUM_SERVICE_TOKEN}`);
    }

    const upstream = await fetch(serviceUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({ imageBase64: body.imageBase64 }),
      signal: AbortSignal.timeout(55_000),
      cache: "no-store",
    });

    const payload = await upstream.json().catch(() => ({ error: "Quantum service returned an invalid response." }));
    return NextResponse.json(payload, { status: upstream.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown quantum service error";
    return NextResponse.json({ error: `Quantum service request failed: ${message}` }, { status: 502 });
  }
}
