"use client";

import { useState, useEffect } from "react";
import { getSessions, deleteSession } from "@/lib/api";
import type { SessionEntry } from "@/lib/types";

interface Props {
  onSessionSelect: (sessionId: string) => void;
}

function statusColor(status: string): string {
  if (status === "COMPLETE") return "#4ade80";
  if (status === "PROCESSING") return "#FFB000";
  if (status === "FAILED") return "#ef4444";
  return "#9f8e78";
}

function toneColor(tone: string | null): string {
  if (!tone) return "#9f8e78";
  const t = tone.toUpperCase();
  if (t === "PERSUASIVE") return "#4ade80";
  if (t === "ASSERTIVE") return "#60a5fa";
  if (t === "INFORMATIVE") return "#a78bfa";
  if (t === "INSPIRATIONAL") return "#FFB000";
  if (t === "CAUTIONARY") return "#ef4444";
  if (t === "HUMOROUS") return "#fbbf24";
  return "#9f8e78";
}

function formatFileSize(bytes: number | null): string {
  if (!bytes) return "—";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function HistoryView({ onSessionSelect }: Props) {
  const [sessions, setSessions] = useState<SessionEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  async function loadSessions() {
    try {
      setLoading(true);
      const data = await getSessions();
      setSessions(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(sessionId: string, e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    } catch {
      // ignore
    }
  }

  const filtered = sessions.filter((s) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      s.id.toLowerCase().includes(q) ||
      s.title.toLowerCase().includes(q) ||
      (s.dominant_tone || "").toLowerCase().includes(q)
    );
  });

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
      {/* Search bar */}
      <div className="flex items-center" style={{ gap: "8px", marginBottom: "16px" }}>
        <div className="flex-1 bevel-recessed flex items-center" style={{ padding: "0 8px", height: "28px" }}>
          <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#9f8e78", marginRight: "8px" }}>search</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="SEARCH_HISTORY_DB..."
            style={{ background: "transparent", border: "none", color: "#FFB000", fontSize: "12px", flex: 1, outline: "none", padding: 0 }}
          />
        </div>
        <button
          onClick={loadSessions}
          className="tactile-button"
          style={{ padding: "4px 16px", fontSize: "11px", color: "#FFB000", fontWeight: 700, border: "none", height: "28px" }}
        >
          [REFRESH]
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: "8px 12px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", marginBottom: "16px", fontSize: "12px", color: "#ef4444" }}>
          {error}. Make sure the API server is running (uvicorn src.api_server:app --port 8000)
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ padding: "24px", textAlign: "center", color: "#9f8e78", fontSize: "13px" }}>
          LOADING_SESSIONS...
        </div>
      )}

      {/* Empty state */}
      {!loading && filtered.length === 0 && !error && (
        <div style={{ padding: "48px", textAlign: "center" }}>
          <span className="material-symbols-outlined" style={{ fontSize: "48px", color: "#524533", marginBottom: "16px", display: "block" }}>folder_open</span>
          <div style={{ fontSize: "14px", color: "#9f8e78", marginBottom: "8px" }}>NO_SESSIONS_FOUND</div>
          <div style={{ fontSize: "11px", color: "#524533" }}>Upload a speech video to get started</div>
        </div>
      )}

      {/* History Table */}
      {filtered.length > 0 && (
        <table className="w-full" style={{ borderCollapse: "collapse", fontSize: "13px" }}>
          <thead>
            <tr style={{ background: "#002244", borderBottom: "1px solid rgba(96,165,250,0.5)" }}>
              <th className="text-left" style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700 }}>TIMESTAMP</th>
              <th className="text-left" style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700 }}>SESSION_ID</th>
              <th className="text-left" style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700 }}>TITLE</th>
              <th style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700 }}>SIZE</th>
              <th style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700, textAlign: "right" }}>SCORE</th>
              <th style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700 }}>TONE</th>
              <th style={{ padding: "8px", color: "#FFB000", fontSize: "11px", fontWeight: 700, textAlign: "center" }}>STATUS</th>
              <th style={{ padding: "8px", width: "40px" }}></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((entry, i) => (
              <tr
                key={entry.id}
                onClick={() => entry.status === "COMPLETE" && onSessionSelect(entry.id)}
                className="event-item cursor-pointer"
                style={{
                  background: i % 2 === 0 ? "#000" : "#0a0a0a",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  opacity: entry.status === "COMPLETE" ? 1 : 0.7,
                }}
              >
                <td style={{ padding: "8px", color: "#9f8e78", fontSize: "12px" }}>
                  {entry.created_at ? new Date(entry.created_at).toLocaleString() : "—"}
                </td>
                <td style={{ padding: "8px" }}>
                  <span className="amber-glow" style={{ color: "#FFB000" }}>{entry.id}</span>
                </td>
                <td style={{ padding: "8px", color: "#e2e2e2", maxWidth: "200px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {entry.title}
                </td>
                <td style={{ padding: "8px", color: "#9f8e78", textAlign: "center", fontSize: "12px" }}>
                  {formatFileSize(entry.file_size)}
                </td>
                <td className="font-bold" style={{ padding: "8px", color: "#FFB000", textAlign: "right" }}>
                  {entry.score != null ? entry.score.toFixed(1) : "—"}
                </td>
                <td style={{ padding: "8px", textAlign: "center" }}>
                  <span style={{ color: toneColor(entry.dominant_tone), fontSize: "12px" }}>
                    {entry.dominant_tone || "—"}
                  </span>
                </td>
                <td style={{ padding: "8px", textAlign: "center" }}>
                  <span className="font-bold" style={{ color: statusColor(entry.status), fontSize: "11px" }}>
                    [{entry.status}]
                  </span>
                </td>
                <td style={{ padding: "8px" }}>
                  <span
                    onClick={(e) => handleDelete(entry.id, e)}
                    className="material-symbols-outlined cursor-pointer"
                    style={{ fontSize: "14px", color: "#524533" }}
                    title="Delete session"
                  >
                    delete
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
