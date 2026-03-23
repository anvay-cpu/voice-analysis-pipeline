"use client";

import type { SpeechReport } from "@/lib/types";
import { scoreColor } from "@/lib/utils";
import LEDBar from "./LEDBar";

export default function PracticeView({ report }: { report: SpeechReport }) {
  const { scores, practice_plan } = report;
  const dims = Object.entries(scores).filter(([k]) => k !== "overall");

  function targetScore(current: number, dimKey: string): number {
    const exercise = practice_plan.find(
      (e) =>
        e.target_dimension &&
        dimKey.toLowerCase().includes(e.target_dimension.toLowerCase().split(" ")[0])
    );
    if (exercise?.target_improvement)
      return Math.min(100, current + exercise.target_improvement);
    return Math.min(100, current + 10);
  }

  // Exercise names mapped to mockup style
  const exerciseNames = [
    "TONAL_STABILITY_CALIBRATION",
    "GESTURE_FLOW_SYNCHRONIZATION",
    "CONTENT_IMPACT_DRILL",
  ];

  // Practice instructions per exercise
  const practiceInstructions = [
    '"Sustain a single vowel sound at a steady pitch for 15 seconds. Repeat 5 times, focusing on eliminating pitch drift and maintaining diaphragmatic support."',
    '"Stand in front of a mirror. Deliver your opening 60 seconds using only open-palm gestures. Record and review for fluid transitions between gestures."',
    '"Take your weakest argument point and restructure it using the Problem-Agitation-Solution framework. Practice delivering it in under 90 seconds."',
  ];

  // Milestones
  const milestones = [
    { label: "STEADY_HAND_V5", value: Math.round(scores.body_language) },
    { label: "MONOTONE_MASTERY", value: Math.round(100 - scores.emotional_expressiveness) < 100 ? Math.round(100 - scores.emotional_expressiveness) : 22 },
    { label: "ZERO_PAUSE_DELIVERY", value: Math.round((scores.vocal_clarity + scores.content_structure) / 2 * 0.6) },
  ];

  // Total progress across exercises
  const totalProgress = Math.round(scores.overall);

  return (
    <div className="flex flex-1 min-h-0">
      {/* Main */}
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
        {/* Header */}
        <div className="flex items-center justify-between" style={{ marginBottom: "4px" }}>
          <div
            style={{
              fontSize: "14px",
              color: "#fd8b00",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "-0.02em",
            }}
            className="amber-glow"
          >
            EXERCISE_MODULE: F6_REPORT_PRACTICE
          </div>
          {/* TOTAL_PROGRESS LED bar on right */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "10px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700 }}>
              TOTAL_PROGRESS
            </span>
            <div style={{ width: "120px" }}>
              <LEDBar value={totalProgress} max={100} segments={12} height={10} />
            </div>
            <span style={{ fontSize: "11px", color: "#FFB000", fontWeight: 700, fontFamily: "monospace" }}>
              {totalProgress}%
            </span>
          </div>
        </div>
        <div style={{ fontSize: "10px", color: "#524533", fontFamily: "monospace", marginBottom: "16px" }}>
          SESSION_ID: 9942-XKB // STATUS: <span style={{ color: "#4ade80" }}>ACTIVE_MONITORING</span>
        </div>

        {/* Exercise blocks */}
        {practice_plan.map((exercise, i) => (
          <div
            key={i}
            style={{
              background: "#0e0e0e",
              borderTop: "1px solid #000",
              borderBottom: "1px solid #524533",
              marginBottom: "12px",
              overflow: "hidden",
            }}
          >
            {/* Orange header bar */}
            <div
              className="flex items-center justify-between"
              style={{
                background: "linear-gradient(to right, rgba(253,139,0,0.15), transparent)",
                borderBottom: "1px solid #524533",
                padding: "8px 12px",
              }}
            >
              <span
                style={{
                  color: "#fd8b00",
                  fontSize: "13px",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  fontFamily: "monospace",
                }}
              >
                {String(i + 1).padStart(2, "0")}. {exerciseNames[i] || exercise.title.toUpperCase().replace(/ /g, "_")}
              </span>
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  color: "#4ade80",
                  border: "1px solid #4ade80",
                  padding: "1px 8px",
                  letterSpacing: "0.08em",
                }}
              >
                READY
              </span>
            </div>

            {/* Description */}
            <div style={{ padding: "12px" }}>
              <p style={{ color: "#e2e2e2", fontSize: "14px", lineHeight: "1.6", margin: "0 0 12px 0" }}>
                {exercise.description}
              </p>

              {/* DO THIS block */}
              <div
                style={{
                  background: "#0a2a1a",
                  border: "1px solid rgba(34,197,94,0.2)",
                  padding: "10px 12px",
                }}
              >
                <div style={{ fontSize: "10px", color: "#4ade80", fontWeight: 700, letterSpacing: "0.1em", marginBottom: "6px" }}>
                  DO THIS:
                </div>
                <p
                  style={{
                    color: "#4ade80",
                    fontSize: "13px",
                    lineHeight: "1.6",
                    fontStyle: "italic",
                    margin: 0,
                    opacity: 0.9,
                  }}
                >
                  {practiceInstructions[i] || `"Practice ${exercise.title} as directed. Follow the exercise schedule: ${exercise.frequency}."`}
                </p>
              </div>

              {/* Meta row */}
              <div className="flex" style={{ gap: "24px", fontSize: "10px", marginTop: "8px" }}>
                <div>
                  <span style={{ color: "#524533" }}>SCHEDULE: </span>
                  <span style={{ color: "#06b6d4" }}>{exercise.frequency}</span>
                </div>
                {exercise.target_dimension && exercise.target_improvement && (
                  <div>
                    <span style={{ color: "#524533" }}>TARGET: </span>
                    <span style={{ color: "#4ade80" }}>
                      {exercise.target_dimension} +{exercise.target_improvement}pts
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </main>

      {/* Right Panel — 320px */}
      <aside
        className="shrink-0 overflow-y-auto no-scrollbar"
        style={{
          width: "320px",
          borderLeft: "1px solid #000",
          background: "#0e0e0e",
          padding: "16px",
        }}
      >
        {/* SCORES_TO_BEAT */}
        <h2
          className="amber-glow"
          style={{
            color: "#fd8b00",
            fontWeight: 700,
            fontSize: "12px",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            marginBottom: "12px",
            borderBottom: "1px solid #524533",
            paddingBottom: "4px",
          }}
        >
          SCORES_TO_BEAT
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: "2px", marginBottom: "24px" }}>
          {[
            { rank: "01", name: "OPERATOR_ALPHA", score: "98.42" },
            { rank: "02", name: "SYS_RELIANCE", score: "96.10" },
            { rank: "03", name: "USER_CURRENT", score: Math.round(scores.overall).toFixed(2) },
          ].map((entry) => (
            <div
              key={entry.rank}
              className="flex items-center"
              style={{
                padding: "4px 8px",
                background: entry.rank === "03" ? "rgba(255,176,0,0.08)" : "transparent",
                fontSize: "12px",
                fontFamily: "monospace",
                gap: "8px",
              }}
            >
              <span style={{ color: "#524533", fontWeight: 700 }}>{entry.rank}.</span>
              <span style={{ color: entry.rank === "03" ? "#FFB000" : "#9f8e78", flex: 1 }}>
                {entry.name}
              </span>
              <span
                className="font-bold"
                style={{
                  color: entry.rank === "03" ? "#FFB000" : "#e2e2e2",
                }}
              >
                {entry.score}
              </span>
            </div>
          ))}
        </div>

        {/* ACTIVE_MILESTONES */}
        <h2
          className="amber-glow"
          style={{
            color: "#fd8b00",
            fontWeight: 700,
            fontSize: "12px",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            marginBottom: "12px",
            borderBottom: "1px solid #524533",
            paddingBottom: "4px",
          }}
        >
          ACTIVE_MILESTONES
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "24px" }}>
          {milestones.map((ms) => (
            <div key={ms.label}>
              <div className="flex items-center justify-between" style={{ marginBottom: "4px" }}>
                <span style={{ color: "#e2e2e2", fontSize: "11px", fontFamily: "monospace", fontWeight: 700 }}>
                  {ms.label}
                </span>
                <span style={{ color: "#FFB000", fontSize: "11px", fontFamily: "monospace" }}>
                  {ms.value}%
                </span>
              </div>
              <LEDBar value={ms.value} max={100} segments={15} height={8} />
            </div>
          ))}
        </div>

        {/* SYSTEM_TELEMETRY */}
        <h2
          className="amber-glow"
          style={{
            color: "#fd8b00",
            fontWeight: 700,
            fontSize: "12px",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            marginBottom: "12px",
            borderBottom: "1px solid #524533",
            paddingBottom: "4px",
          }}
        >
          SYSTEM_TELEMETRY
        </h2>

        <div
          style={{
            background: "#0a0a0a",
            borderTop: "1px solid #000",
            borderBottom: "1px solid #524533",
            padding: "10px 12px",
          }}
        >
          {[
            { label: "Neural Link", status: "STABLE", color: "#4ade80" },
            { label: "Voice Sync", status: "CALIBRATED", color: "#06b6d4" },
          ].map((item) => (
            <div
              key={item.label}
              className="flex items-center justify-between"
              style={{ height: "24px", fontSize: "11px", fontFamily: "monospace" }}
            >
              <span style={{ color: "#524533" }}>{item.label}</span>
              <span style={{ color: item.color, fontWeight: 700 }}>{item.status}</span>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
