"use client";

import type { SpeechReport } from "@/lib/types";
import { formatDuration, heatmapChar, scoreColor } from "@/lib/utils";
import LEDBar from "./LEDBar";

export default function TimelineView({ report }: { report: SpeechReport }) {
  const { heatmap_channels, disruption_events, emotion_arc, duration_sec } = report;
  const duration = duration_sec;

  const heatWidth = 60;
  const secPerChar = duration / heatWidth;

  // Duration timer formatted as HH:MM:SS:FF
  const dHours = String(Math.floor(duration / 3600)).padStart(2, "0");
  const dMins = String(Math.floor((duration % 3600) / 60)).padStart(2, "0");
  const dSecs = String(Math.floor(duration % 60)).padStart(2, "0");
  const dFrames = "09";
  const durationTimer = `${dHours}:${dMins}:${dSecs}:${dFrames}`;

  function downsample(values: number[]): number[] {
    const result: number[] = [];
    for (let i = 0; i < heatWidth; i++) {
      const start = Math.floor(i * secPerChar);
      const end = Math.min(Math.floor((i + 1) * secPerChar), values.length);
      if (start >= values.length) {
        result.push(0.5);
        continue;
      }
      let sum = 0,
        count = 0;
      for (let j = start; j < end; j++) {
        sum += values[j];
        count++;
      }
      result.push(count > 0 ? sum / count : 0.5);
    }
    return result;
  }

  // Emotion arc SVG dimensions
  const svgWidth = 700;
  const svgHeight = 160;
  const svgPadL = 40;
  const svgPadR = 20;
  const svgPadT = 20;
  const svgPadB = 30;
  const plotW = svgWidth - svgPadL - svgPadR;
  const plotH = svgHeight - svgPadT - svgPadB;

  function arcToPoints(key: "voice_valence" | "face_valence" | "text_valence"): string {
    if (emotion_arc.length === 0) return "";
    const step = plotW / Math.max(1, emotion_arc.length - 1);
    return emotion_arc
      .map((pt, i) => {
        const x = svgPadL + i * step;
        const y = svgPadT + ((1 - pt[key]) / 2) * plotH;
        return `${x},${y}`;
      })
      .join(" ");
  }

  const voicePoints = arcToPoints("voice_valence");
  const facePoints = arcToPoints("face_valence");
  const textPoints = arcToPoints("text_valence");

  // Time labels for heatmap x-axis
  const timeLabels: string[] = [];
  const labelInterval = Math.ceil(duration / 10 / 30) * 30;
  for (let t = 0; t <= duration; t += labelInterval) timeLabels.push(formatDuration(t));

  // Heatmap row renderer
  function renderHeatRow(ch: { label: string; key: string; values: number[] }) {
    const sampled = downsample(ch.values);
    return (
      <div key={ch.key} className="flex items-center" style={{ height: "18px", lineHeight: "18px" }}>
        <span className="shrink-0 font-bold" style={{ width: "52px", color: "#9f8e78", fontSize: "11px" }}>
          {ch.label}
        </span>
        <span style={{ fontSize: "12px", letterSpacing: "-1px" }}>
          {sampled.map((v, i) => {
            const { char, className } = heatmapChar(v);
            return (
              <span key={i} className={className}>
                {char}
              </span>
            );
          })}
        </span>
      </div>
    );
  }

  // Status badge color
  function statusColor(status: string): string {
    if (status === "GOOD" || status === "RECOVERED") return "#4ade80";
    if (status === "SLOW") return "#ffb4ab";
    if (status === "MISMATCH") return "#fd8b00";
    return "#06b6d4";
  }

  function statusBg(status: string): string {
    if (status === "GOOD" || status === "RECOVERED") return "rgba(34,197,94,0.15)";
    if (status === "SLOW") return "rgba(255,180,171,0.1)";
    if (status === "MISMATCH") return "rgba(253,139,0,0.1)";
    return "rgba(6,182,212,0.1)";
  }

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
      {/* Header */}
      <div className="flex items-center justify-between" style={{ marginBottom: "16px" }}>
        <div
          style={{
            fontSize: "14px",
            color: "#fd8b00",
            textTransform: "uppercase",
            fontWeight: 700,
            letterSpacing: "-0.02em",
          }}
          className="amber-glow"
        >
          SESSION_REPORT: F2_TIMELINE
        </div>
        <div
          style={{
            fontSize: "18px",
            color: "#FFB000",
            fontWeight: 700,
            fontFamily: "monospace",
            letterSpacing: "0.05em",
          }}
          className="amber-glow"
        >
          {durationTimer}
        </div>
      </div>

      {/* VOICE_PROJECTION_HEATMAP */}
      <div
        style={{
          fontSize: "11px",
          color: "#fd8b00",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: "8px",
          borderBottom: "1px solid #524533",
          paddingBottom: "4px",
        }}
      >
        VOICE_PROJECTION_HEATMAP
      </div>
      <div style={{ marginBottom: "4px" }}>
        {heatmap_channels.slice(0, 4).map(renderHeatRow)}
      </div>
      <div className="flex" style={{ paddingLeft: "52px", fontSize: "10px", color: "#524533", marginBottom: "16px" }}>
        {timeLabels.map((l, i) => (
          <span key={i} style={{ width: `${100 / timeLabels.length}%` }}>
            {l}
          </span>
        ))}
      </div>

      {/* BODY_LANGUAGE_KINETICS */}
      <div
        style={{
          fontSize: "11px",
          color: "#fd8b00",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: "8px",
          borderBottom: "1px solid #524533",
          paddingBottom: "4px",
        }}
      >
        BODY_LANGUAGE_KINETICS
      </div>
      <div style={{ marginBottom: "4px" }}>
        {heatmap_channels.slice(4, 8).map(renderHeatRow)}
      </div>
      <div className="flex" style={{ paddingLeft: "52px", fontSize: "10px", color: "#524533", marginBottom: "4px" }}>
        {timeLabels.map((l, i) => (
          <span key={i} style={{ width: `${100 / timeLabels.length}%` }}>
            {l}
          </span>
        ))}
      </div>
      {/* Heatmap legend */}
      <div className="flex" style={{ paddingLeft: "52px", fontSize: "10px", gap: "8px", marginBottom: "20px" }}>
        <span className="heatmap-good">{"\u2588"} GOOD</span>
        <span className="heatmap-ok">{"\u2593"} OK</span>
        <span className="heatmap-poor">{"\u2592"} POOR</span>
        <span className="heatmap-critical">{"\u2591"} CRIT</span>
      </div>

      {/* EMOTION_ARC — SVG line chart */}
      <div
        style={{
          fontSize: "11px",
          color: "#fd8b00",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: "8px",
          borderBottom: "1px solid #524533",
          paddingBottom: "4px",
        }}
      >
        EMOTION_ARC
      </div>
      <div
        style={{
          background: "#0e0e0e",
          borderTop: "1px solid #000",
          borderBottom: "1px solid #524533",
          padding: "8px",
          marginBottom: "8px",
        }}
      >
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: "100%", height: "auto" }}
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Grid lines */}
          {[1, 0.5, 0, -0.5, -1].map((val) => {
            const y = svgPadT + ((1 - val) / 2) * plotH;
            return (
              <g key={val}>
                <line
                  x1={svgPadL}
                  y1={y}
                  x2={svgWidth - svgPadR}
                  y2={y}
                  stroke={val === 0 ? "#524533" : "#1b1b1b"}
                  strokeWidth={val === 0 ? 1 : 0.5}
                />
                <text
                  x={svgPadL - 6}
                  y={y + 3}
                  textAnchor="end"
                  fill="#524533"
                  fontSize="9"
                  fontFamily="monospace"
                >
                  {val > 0 ? `+${val.toFixed(1)}` : val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Time axis labels */}
          {timeLabels.map((l, i) => {
            const x = svgPadL + (i / Math.max(1, timeLabels.length - 1)) * plotW;
            return (
              <text
                key={i}
                x={x}
                y={svgHeight - 6}
                textAnchor="middle"
                fill="#524533"
                fontSize="9"
                fontFamily="monospace"
              >
                {l}
              </text>
            );
          })}

          {/* Voice valence line (red) */}
          {voicePoints && (
            <polyline
              points={voicePoints}
              fill="none"
              stroke="#ef4444"
              strokeWidth="1.5"
              opacity="0.8"
            />
          )}

          {/* Face valence line (blue) */}
          {facePoints && (
            <polyline
              points={facePoints}
              fill="none"
              stroke="#60a5fa"
              strokeWidth="1.5"
              opacity="0.8"
            />
          )}

          {/* Text valence line (green) */}
          {textPoints && (
            <polyline
              points={textPoints}
              fill="none"
              stroke="#22c55e"
              strokeWidth="1.5"
              opacity="0.8"
            />
          )}
        </svg>
      </div>

      {/* Emotion arc legend */}
      <div className="flex" style={{ fontSize: "10px", gap: "16px", marginBottom: "20px" }}>
        <div className="flex items-center" style={{ gap: "4px" }}>
          <div style={{ width: "16px", height: "2px", background: "#ef4444" }} />
          <span style={{ color: "#ef4444" }}>VOICE</span>
        </div>
        <div className="flex items-center" style={{ gap: "4px" }}>
          <div style={{ width: "16px", height: "2px", background: "#60a5fa" }} />
          <span style={{ color: "#60a5fa" }}>FACE</span>
        </div>
        <div className="flex items-center" style={{ gap: "4px" }}>
          <div style={{ width: "16px", height: "2px", background: "#22c55e" }} />
          <span style={{ color: "#22c55e" }}>TEXT</span>
        </div>
      </div>

      {/* CRITICAL_DISRUPTION_LOG table */}
      <div
        style={{
          fontSize: "11px",
          color: "#fd8b00",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: "8px",
          borderBottom: "1px solid #524533",
          paddingBottom: "4px",
        }}
      >
        CRITICAL_DISRUPTION_LOG ({disruption_events.length})
      </div>

      {/* Table header */}
      <div
        className="flex font-bold"
        style={{
          fontSize: "10px",
          color: "#524533",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          borderBottom: "1px solid #524533",
          padding: "6px 8px",
          background: "#0a0a0a",
        }}
      >
        <span style={{ width: "80px" }}>TIMESTAMP</span>
        <span style={{ flex: 1 }}>EVENT_TYPE</span>
        <span style={{ width: "80px", textAlign: "center" }}>INTENSITY</span>
        <span style={{ width: "90px", textAlign: "center" }}>RECOVERY_MS</span>
        <span style={{ width: "90px", textAlign: "right" }}>STATUS</span>
      </div>

      {/* Table rows */}
      {disruption_events.map((ev, i) => (
        <div
          key={i}
          className="flex items-center"
          style={{
            fontSize: "12px",
            fontFamily: "monospace",
            padding: "6px 8px",
            height: "32px",
            background: i % 2 === 0 ? "#000" : "#0a0a0a",
            borderBottom: "1px solid rgba(255,255,255,0.03)",
          }}
        >
          <span style={{ width: "80px", color: "#9f8e78" }}>{ev.timestamp}</span>
          <span style={{ flex: 1, color: "#FFB000" }}>{ev.label}</span>
          <span
            style={{
              width: "80px",
              textAlign: "center",
              color: ev.composure !== null ? scoreColor(ev.composure) : "#353535",
            }}
          >
            {ev.composure !== null ? `${ev.composure}%` : "---"}
          </span>
          <span style={{ width: "90px", textAlign: "center", color: "#9f8e78" }}>
            {ev.recovery_sec !== null ? `${Math.round(ev.recovery_sec * 1000)}ms` : "---"}
          </span>
          <span style={{ width: "90px", textAlign: "right" }}>
            <span
              style={{
                display: "inline-block",
                fontSize: "10px",
                fontWeight: 700,
                color: statusColor(ev.status),
                background: statusBg(ev.status),
                border: `1px solid ${statusColor(ev.status)}`,
                padding: "1px 8px",
                letterSpacing: "0.05em",
              }}
            >
              {ev.status}
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}
