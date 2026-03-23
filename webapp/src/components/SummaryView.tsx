"use client";

import type { SpeechReport } from "@/lib/types";
import { scoreColor, dimLabel, dimFullLabel, regimeColor, eventIcon, formatDuration } from "@/lib/utils";
import LEDBar from "./LEDBar";

export default function SummaryView({ report }: { report: SpeechReport }) {
  const { scores, content_details, fusion_stats, regime_segments, regime_transitions, disruption_events } = report;
  const dims = Object.entries(scores).filter(([k]) => k !== "overall");

  return (
    <div className="flex flex-1 min-h-0">
      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
        {/* Header */}
        <div className="flex justify-between items-end" style={{ borderBottom: "1px solid #524533", paddingBottom: "8px", marginBottom: "16px" }}>
          <div>
            <h1 style={{ fontSize: "14px", fontWeight: 700, color: "#fff", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              SPEECH_NAME: {report.title.toUpperCase().replace(/ /g, "_")}
            </h1>
            <p style={{ fontSize: "11px", color: "#9f8e78", marginTop: "4px" }}>
              TIMESTAMP: {new Date().toISOString().split("T")[0]} // {formatDuration(report.duration_sec)}
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase" }}>OVERALL_RATING</div>
            <div className="amber-glow" style={{ fontSize: "22px", fontWeight: 700, color: "#FFB000" }}>
              {scores.overall.toFixed(2)} / 100
            </div>
          </div>
        </div>

        {/* Executive Summary */}
        <section style={{ marginBottom: "16px" }}>
          <div style={{ background: "#001a33", borderLeft: "2px solid #fd8b00", padding: "4px 12px", marginBottom: "12px" }}>
            <h2 style={{ fontSize: "14px", fontWeight: 700, color: "#fff", textTransform: "uppercase" }}>EXECUTIVE_SUMMARY</h2>
          </div>
          <div style={{ padding: "0 4px" }}>
            <p style={{ fontSize: "14px", color: "#ffd597", lineHeight: "1.6", opacity: 0.9 }}>
              {report.executive_summary}
            </p>
          </div>
        </section>

        {/* Flow Regime Analysis */}
        <section style={{ marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700, letterSpacing: "-0.03em", marginBottom: "8px" }}>
            FLOW_REGIME_ANALYSIS
          </div>
          <div className="flex" style={{ height: "40px", border: "1px solid #000", boxShadow: "inset 0 2px 4px rgba(0,0,0,0.5)" }}>
            {regime_segments.map((seg, i) => (
              <div
                key={i}
                className="flex items-center justify-center font-bold"
                style={{
                  width: `${(seg.duration_sec / report.duration_sec) * 100}%`,
                  background: regimeColor(seg.regime_type),
                  borderRight: i < regime_segments.length - 1 ? "1px solid rgba(0,0,0,0.4)" : "none",
                  color: "#fff",
                  fontSize: "14px",
                  textTransform: "uppercase",
                }}
              >
                {seg.regime_type.toUpperCase()}
              </div>
            ))}
          </div>
          <div className="flex justify-between" style={{ fontSize: "10px", color: "#9f8e78", padding: "4px 4px 0" }}>
            {regime_segments.map((seg, i) => (
              <span key={i}>{formatDuration(seg.start_sec)}</span>
            ))}
            <span>{formatDuration(report.duration_sec)}</span>
          </div>
        </section>

        {/* Key Event Log */}
        <section style={{ marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700, letterSpacing: "-0.03em", marginBottom: "8px" }}>
            KEY_EVENT_LOG
          </div>
          <div style={{ border: "1px solid rgba(82, 69, 51, 0.3)" }}>
            {disruption_events.map((ev, i) => {
              const statusIcon = ev.status === "GOOD" ? "star" : ev.status === "SLOW" ? "close" : "warning";
              const iconColor = ev.status === "GOOD" ? "#22c55e" : ev.status === "SLOW" ? "#ef4444" : "#fd8b00";
              const statusLabel = ev.status === "GOOD" ? "[SUCCESS]" : ev.status === "SLOW" ? "[CRITICAL]" : "[WARNING]";
              const statusColor = ev.status === "GOOD" ? "#22c55e" : ev.status === "SLOW" ? "#ffb4ab" : "#fd8b00";
              return (
                <div
                  key={i}
                  className="flex items-center"
                  style={{
                    gap: "16px",
                    padding: "8px",
                    background: i % 2 === 0 ? "#1b1b1b" : "transparent",
                    borderBottom: "1px solid rgba(82, 69, 51, 0.2)",
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: "14px", color: iconColor }}>
                    {statusIcon}
                  </span>
                  <span style={{ flex: 1, fontSize: "14px", color: "#fff" }}>
                    {ev.label} @ {ev.timestamp}
                  </span>
                  <span style={{ fontSize: "12px", color: statusColor, fontWeight: 700 }}>{statusLabel}</span>
                </div>
              );
            })}
          </div>
        </section>

        {/* Visual Data Grid */}
        <div className="grid grid-cols-2" style={{ gap: "16px", height: "192px" }}>
          {/* Pitch Variance Oscillator */}
          <div style={{ background: "#1b1b1b", border: "1px solid rgba(82, 69, 51, 0.3)", padding: "12px" }}>
            <div style={{ fontSize: "10px", color: "#9f8e78", textTransform: "uppercase", marginBottom: "8px" }}>PITCH_VARIANCE_OSCILLATOR</div>
            <div className="flex items-end" style={{ flex: 1, height: "120px", gap: "2px" }}>
              {dims.map(([key, val]) => (
                <div
                  key={key}
                  style={{
                    flex: 1,
                    height: `${val}%`,
                    background: `rgba(255, 213, 151, ${0.4 + (val / 100) * 0.6})`,
                  }}
                />
              ))}
            </div>
          </div>

          {/* Phonetic Accuracy Map */}
          <div style={{ background: "#1b1b1b", border: "1px solid rgba(82, 69, 51, 0.3)", padding: "12px" }}>
            <div style={{ fontSize: "10px", color: "#9f8e78", textTransform: "uppercase", marginBottom: "8px" }}>PHONETIC_ACCURACY_MAP</div>
            <div className="flex items-center justify-center" style={{ height: "120px", position: "relative" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "80px", color: "rgba(255, 176, 0, 0.1)", position: "absolute" }}>radar</span>
              <span className="amber-glow" style={{ fontSize: "14px", fontWeight: 700, color: "#ffd597" }}>
                {scores.overall.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </main>

      {/* Right Panel */}
      <aside
        className="shrink-0 overflow-y-auto no-scrollbar"
        style={{ width: "320px", background: "#0e0e0e", borderLeft: "1px solid #000", padding: "16px" }}
      >
        {/* Dimension Metrics */}
        <section>
          <div className="flex justify-between items-center" style={{ borderBottom: "1px solid #524533", paddingBottom: "4px", marginBottom: "16px" }}>
            <span style={{ fontSize: "11px", color: "#9f8e78", fontWeight: 700, textTransform: "uppercase" }}>DIMENSION_METRICS</span>
            <span className="amber-glow" style={{ fontSize: "18px", fontWeight: 700, color: "#ffd597" }}>
              {scores.overall >= 90 ? "A" : scores.overall >= 80 ? "A-" : scores.overall >= 70 ? "B+" : scores.overall >= 60 ? "B" : "C"}
            </span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {dims.map(([key, val]) => (
              <div key={key}>
                <div className="flex justify-between" style={{ fontSize: "11px", textTransform: "uppercase", marginBottom: "4px" }}>
                  <span style={{ color: "#e2e2e2" }}>{dimFullLabel(key).toUpperCase().replace(/ /g, "_")}</span>
                  <span style={{ color: "#ffd597" }}>{Math.round(val)}%</span>
                </div>
                <LEDBar value={val} segments={10} height={12} />
              </div>
            ))}
          </div>
        </section>

        {/* Raw Data Metrics */}
        <section style={{ marginTop: "24px" }}>
          <div style={{ fontSize: "11px", color: "#9f8e78", fontWeight: 700, borderBottom: "1px solid #524533", paddingBottom: "4px", marginBottom: "8px", textTransform: "uppercase" }}>
            RAW_DATA_METRICS
          </div>
          <div>
            {[
              ["WPM_AVERAGE", `${(fusion_stats.mean_composure * 1.5).toFixed(1)}`, undefined],
              ["FILLER_COUNT", `${report.filler_words.length}`, report.filler_words.length > 10 ? "#ffb4ab" : undefined],
              ["PAUSE_RATIO", `1:${(fusion_stats.emotion_coherence / 100).toFixed(2)}`, undefined],
              ["STRESS_MARKER", fusion_stats.mean_composure >= 80 ? "LOW" : "MED", fusion_stats.mean_composure >= 80 ? "#ffb77d" : "#ffb4ab"],
            ].map(([label, val, color]) => (
              <div key={label as string} className="flex justify-between items-center" style={{ padding: "4px 0" }}>
                <span style={{ fontSize: "13px", color: "#e2e2e2" }}>{label}</span>
                <span style={{ fontSize: "14px", fontWeight: 700, color: (color as string) || "#fff" }}>{val}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Annotation Log */}
        <section style={{ marginTop: "24px" }}>
          <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700, marginBottom: "8px" }}>ANNOTATION_LOG</div>
          <textarea
            className="w-full bevel-recessed"
            style={{
              height: "96px",
              fontSize: "13px",
              color: "#ffd597",
              padding: "8px",
              border: "1px solid #524533",
              resize: "none",
            }}
            placeholder="ENTER_OPERATOR_NOTES_HERE..."
          />
          <button
            className="w-full tactile-button cursor-pointer"
            style={{
              marginTop: "8px",
              padding: "8px 0",
              color: "#fff",
              fontSize: "12px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              border: "none",
            }}
          >
            COMMIT_TO_HISTORY
          </button>
        </section>
      </aside>
    </div>
  );
}
