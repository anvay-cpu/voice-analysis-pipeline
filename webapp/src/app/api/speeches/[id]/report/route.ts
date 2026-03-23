import { NextRequest } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

const API_BASE = process.env.API_URL || "http://localhost:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  // Try the FastAPI backend first
  try {
    const res = await fetch(`${API_BASE}/api/reports/${id}`, {
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      const data = await res.json();
      return new Response(JSON.stringify(data), {
        headers: { "Content-Type": "application/json" },
      });
    }
  } catch {
    // FastAPI not available, fall through to file-based lookup
  }

  // Fallback: look for report JSON in public/data directory
  const dataDir = path.join(process.cwd(), "public", "data");
  const filePath = path.join(dataDir, `${id}.json`);

  try {
    const data = await readFile(filePath, "utf-8");
    return new Response(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(JSON.stringify({ error: `Report '${id}' not found` }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }
}
