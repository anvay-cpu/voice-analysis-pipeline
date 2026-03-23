import { NextRequest, NextResponse } from "next/server";
import { execSync } from "child_process";
import path from "path";

export async function POST(req: NextRequest) {
  try {
    const report = await req.json();

    // Already enriched — skip
    if (report._enriched) {
      return NextResponse.json(report);
    }

    const scriptPath = path.resolve(process.cwd(), "..", "scripts", "enrich_report.py");

    const result = execSync(`python3 "${scriptPath}"`, {
      input: JSON.stringify(report),
      timeout: 200_000, // 200s — Claude calls take time
      maxBuffer: 10 * 1024 * 1024,
      encoding: "utf-8",
    });

    const enriched = JSON.parse(result);
    return NextResponse.json(enriched);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Enrichment failed:", message);
    return NextResponse.json(
      { error: "Enrichment failed", detail: message },
      { status: 500 }
    );
  }
}
