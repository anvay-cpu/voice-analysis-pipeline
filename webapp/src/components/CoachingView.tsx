"use client";

import { useState } from "react";
import type { SpeechReport } from "@/lib/types";

export default function CoachingView({ report }: { report: SpeechReport }) {
  const { coaching_feedback, disruption_events, timeline } = report;
  const [currentIdx, setCurrentIdx] = useState(0);

  const current = coaching_feedback[currentIdx];

  function findEvent(timeSec: number) {
    return disruption_events.find((e) => e.time_sec === timeSec);
  }

  function getTimelineAt(timeSec: number) {
    if (timeSec < 0 || timeSec >= timeline.length) return null;
    return timeline[timeSec];
  }

  const matchedEvent = current ? findEvent(current.time_sec) : null;
  const timelineData = current ? getTimelineAt(current.time_sec) : null;

  function priorityBadge(block: typeof current) {
    if (!block) return { label: "LOW", bg: "#353535", color: "#FFB000" };
    const ev = findEvent(block.time_sec);
    if (ev?.status === "SLOW") return { label: "HIGH", bg: "#9c1209", color: "#fff" };
    if (block.type === "emotion_mismatch") return { label: "MED", bg: "#353535", color: "#FFB000" };
    if (ev?.status === "GOOD") return { label: "LOW", bg: "#353535", color: "#FFB000" };
    return { label: "MED", bg: "#353535", color: "#FFB000" };
  }

  return (
    <div className="flex flex-1 min-h-0">
      {/* Main Content */}
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
        {/* Event Navigation */}
        <div className="flex items-center justify-center" style={{ gap: "16px", marginBottom: "32px" }}>
          <button
            onClick={() => setCurrentIdx(Math.max(0, currentIdx - 1))}
            disabled={currentIdx === 0}
            className="bevel-raised cursor-pointer disabled:opacity-30"
            style={{
              background: "#2a2a2a",
              color: "#FFB000",
              fontWeight: 700,
              padding: "4px 16px",
              border: "none",
              fontSize: "14px",
            }}
          >
            {"\u25C4"}
          </button>
          <div
            className="bevel-recessed"
            style={{
              padding: "4px 32px",
              color: "#fd8b00",
              fontWeight: 700,
              letterSpacing: "0.1em",
              fontSize: "14px",
            }}
          >
            EVENT {currentIdx + 1} OF {coaching_feedback.length}
          </div>
          <button
            onClick={() => setCurrentIdx(Math.min(coaching_feedback.length - 1, currentIdx + 1))}
            disabled={currentIdx >= coaching_feedback.length - 1}
            className="bevel-raised cursor-pointer disabled:opacity-30"
            style={{
              background: "#2a2a2a",
              color: "#FFB000",
              fontWeight: 700,
              padding: "4px 16px",
              border: "none",
              fontSize: "14px",
            }}
          >
            {"\u25BA"}
          </button>
        </div>

        {/* Coaching Event Blocks */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {coaching_feedback.map((block, i) => {
            const badge = priorityBadge(block);

            return (
              <div
                key={i}
                id={`coaching-${i}`}
                onClick={() => setCurrentIdx(i)}
                style={{
                  background: "#131313",
                  borderLeft: "4px solid #FFB000",
                  padding: "16px",
                  cursor: "pointer",
                  opacity: i === currentIdx ? 1 : 0.6,
                  transition: "opacity 0.15s ease",
                }}
              >
                {/* Header with priority badge */}
                <div className="flex justify-between items-start" style={{ marginBottom: "8px" }}>
                  <h3 style={{ color: "#fd8b00", fontWeight: 700, fontSize: "14px", textTransform: "uppercase", margin: 0 }}>
                    [{block.timestamp}] {block.label.toUpperCase().replace(/ /g, "_")}
                  </h3>
                  <span
                    style={{
                      fontSize: "11px",
                      background: badge.bg,
                      color: badge.color,
                      padding: "2px 8px",
                      fontWeight: 700,
                      flexShrink: 0,
                    }}
                  >
                    PRIORITY: {badge.label}
                  </span>
                </div>

                {/* Feedback text */}
                <p style={{ color: "rgba(255, 176, 0, 0.8)", fontSize: "14px", lineHeight: "1.6", marginBottom: "12px", marginTop: 0 }}>
                  {block.feedback}
                </p>

                {/* Action item */}
                <div className="flex items-center" style={{ gap: "8px", color: "#4ade80", fontSize: "14px", fontWeight: 700 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>lightbulb</span>
                  <span>ACTION: {getActionText(block)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </main>

      {/* Right Panel */}
      <aside
        className="shrink-0 overflow-y-auto no-scrollbar"
        style={{ width: "320px", background: "#0e0e0e", borderLeft: "1px solid #000", padding: "16px" }}
      >
        {/* Session Statistics */}
        <h2
          className="amber-glow"
          style={{ color: "#fd8b00", fontWeight: 700, fontSize: "14px", textTransform: "uppercase", marginBottom: "16px", marginTop: 0 }}
        >
          SESSION_STATISTICS
        </h2>
        <div style={{ marginBottom: "24px" }}>
          <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: "8px", fontSize: "13px" }}>
            {timelineData ? (
              <>
                <span style={{ color: "#9f8e78" }}>AVG_PITCH:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>{timelineData.voice.pitch_cv.toFixed(2)} HZ</span>
                <span style={{ color: "#9f8e78" }}>WPM_AVG:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>{timelineData.voice.syllable_rate.toFixed(1)}</span>
                <span style={{ color: "#9f8e78" }}>RESONANCE:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>
                  {matchedEvent?.composure && matchedEvent.composure >= 80 ? "OPTIMAL" : "MODERATE"}
                </span>
                <span style={{ color: "#9f8e78" }}>JITTER:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>
                  {(timelineData.voice.pitch_cv * 100).toFixed(2)}%
                </span>
              </>
            ) : (
              <>
                <span style={{ color: "#9f8e78" }}>COMPOSURE:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>
                  {matchedEvent?.composure ?? "---"}%
                </span>
                <span style={{ color: "#9f8e78" }}>RECOVERY:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>
                  {matchedEvent?.recovery_sec ?? "---"}s
                </span>
                <span style={{ color: "#9f8e78" }}>STATUS:</span>
                <span style={{ color: "#FFB000", fontWeight: 700, textAlign: "right" }}>
                  {matchedEvent?.status ?? "---"}
                </span>
              </>
            )}
          </div>

          {/* Intensity bar */}
          <div style={{ marginTop: "16px" }}>
            <div style={{ fontSize: "11px", color: "#9f8e78", marginBottom: "4px", textTransform: "uppercase", fontWeight: 700 }}>
              INTENSITY
            </div>
            <div style={{ height: "8px", background: "#353535", width: "100%", position: "relative" }}>
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  height: "100%",
                  width: `${matchedEvent?.composure ?? 72}%`,
                  background: "#FFB000",
                }}
              />
            </div>
            <div className="flex justify-between" style={{ fontSize: "10px", color: "#9f8e78", marginTop: "4px" }}>
              <span>INTENSITY_L</span>
              <span>{matchedEvent?.composure ?? 72}%</span>
              <span>INTENSITY_R</span>
            </div>
          </div>
        </div>

        {/* Event Log Quick Select */}
        <h2
          className="amber-glow"
          style={{ color: "#fd8b00", fontWeight: 700, fontSize: "14px", textTransform: "uppercase", marginBottom: "16px", marginTop: 0 }}
        >
          EVENT_LOG_QUICK_SELECT
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          {coaching_feedback.map((block, i) => {
            const isActive = i === currentIdx;
            const shortLabel = block.label.slice(0, 14).toUpperCase().replace(/ /g, "_");
            return (
              <button
                key={i}
                onClick={() => {
                  setCurrentIdx(i);
                  document.getElementById(`coaching-${i}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
                }}
                className={`w-full text-left cursor-pointer flex justify-between items-center ${isActive ? "bevel-raised" : "bevel-recessed"}`}
                style={{
                  padding: "8px",
                  fontSize: "13px",
                  border: "none",
                  color: isActive ? "#60a5fa" : "rgba(255, 176, 0, 0.6)",
                  background: isActive ? "rgba(30, 58, 138, 0.4)" : "#1b1b1b",
                }}
              >
                <span>{String(i + 1).padStart(2, "0")} {shortLabel}</span>
                <span style={{ fontSize: "10px", opacity: 0.6 }}>
                  {isActive ? "SELECTED" : block.timestamp}
                </span>
              </button>
            );
          })}
        </div>
      </aside>
    </div>
  );
}

function getActionText(block: { type: string; feedback: string }): string {
  if (block.type === "transition") return "Implement tactical pause before transition.";
  if (block.type === "emotion_mismatch") return "Align vocal tone with intended message emotion.";
  if (block.feedback.toLowerCase().includes("filler")) return "Replace transitions with silent breath cycles.";
  return "Review and apply corrective technique in next session.";
}
