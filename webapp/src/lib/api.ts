import type { UploadResponse, JobStatus, SessionEntry, SpeechReport } from "./types";

const LOCAL_API = "http://localhost:8000";

function getBackendUrl(): string {
  if (typeof window !== "undefined") {
    const saved = localStorage.getItem("colab_url");
    if (saved) return saved.replace(/\/+$/, "");
  }
  return LOCAL_API;
}

/** Headers needed for ngrok free tier (skips browser warning interstitial) */
function ngrokHeaders(): Record<string, string> {
  const url = getBackendUrl();
  if (url.includes("ngrok")) {
    return { "ngrok-skip-browser-warning": "true" };
  }
  return {};
}

export function setBackendUrl(url: string) {
  if (typeof window !== "undefined") {
    if (url) {
      localStorage.setItem("colab_url", url.replace(/\/+$/, ""));
    } else {
      localStorage.removeItem("colab_url");
    }
  }
}

export function getStoredBackendUrl(): string {
  if (typeof window !== "undefined") {
    return localStorage.getItem("colab_url") || "";
  }
  return "";
}

export async function checkBackendHealth(): Promise<{
  ok: boolean;
  url: string;
  latency_ms: number;
}> {
  const url = getBackendUrl();
  const start = Date.now();
  try {
    const res = await fetch(`${url}/api/health`, {
      headers: ngrokHeaders(),
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      return { ok: true, url, latency_ms: Date.now() - start };
    }
    return { ok: false, url, latency_ms: Date.now() - start };
  } catch {
    return { ok: false, url, latency_ms: Date.now() - start };
  }
}

export async function uploadVideo(
  file: File,
  title: string,
  profile: string
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", title || file.name);
  formData.append("profile", profile);

  const res = await fetch(`${getBackendUrl()}/api/upload`, {
    method: "POST",
    headers: ngrokHeaders(),
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${getBackendUrl()}/api/jobs/${jobId}`, {
    headers: ngrokHeaders(),
  });
  if (!res.ok) throw new Error("Failed to get job status");
  return res.json();
}

export async function getReport(sessionId: string): Promise<SpeechReport> {
  const res = await fetch(`${getBackendUrl()}/api/reports/${sessionId}`, {
    headers: ngrokHeaders(),
  });
  if (!res.ok) throw new Error(`Report '${sessionId}' not found`);
  return res.json();
}

export async function enrichReport(report: SpeechReport): Promise<SpeechReport> {
  /** Enrich a report with detailed Claude coaching (runs locally via claude -p). */
  if ((report as Record<string, unknown>)._enriched) return report;
  try {
    const res = await fetch("/api/enrich", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(report),
    });
    if (!res.ok) throw new Error("Enrichment failed");
    return res.json();
  } catch {
    return report; // Fall back to template coaching
  }
}

export async function getSessions(): Promise<SessionEntry[]> {
  const res = await fetch(`${getBackendUrl()}/api/sessions`, {
    headers: ngrokHeaders(),
  });
  if (!res.ok) throw new Error("Failed to load sessions");
  const data = await res.json();
  return data.sessions;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${getBackendUrl()}/api/sessions/${sessionId}`, {
    method: "DELETE",
    headers: ngrokHeaders(),
  });
  if (!res.ok) throw new Error("Failed to delete session");
}
