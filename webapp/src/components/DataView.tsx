"use client";

import { useMemo } from "react";
import type { SpeechReport } from "@/lib/types";
import { scoreColor } from "@/lib/utils";
import LEDBar from "./LEDBar";

interface MetricRow {
  displayName: string;
  scoreKey: string;
  score: number;
  delta: number;
}

const DIMENSION_MAP: Record<string, string> = {
  vocal_clarity: "LEXICAL_DENSITY",
  emotional_expressiveness: "PITCH_VARIANCE",
  body_language: "VOCAL_FRY_INDEX",
  content_structure: "CADENCE_STABILITY",
  regime_adaptability: "SENTIMENT_ALIGN",
};

const SIMULATED_DELTAS: Record<string, number> = {
  vocal_clarity: 1.2,
  emotional_expressiveness: -4.5,
  body_language: 0.0,
  content_structure: 8.2,
  regime_adaptability: -0.2,
};

export default function DataView({ report }: { report: SpeechReport }) {
  const { scores, filler_words, transcript } = report;

  const metrics: MetricRow[] = useMemo(() => {
    const keys = ["vocal_clarity", "emotional_expressiveness", "body_language", "content_structure", "regime_adaptability"];
    return keys.map((key) => ({
      displayName: DIMENSION_MAP[key] || key.toUpperCase(),
      scoreKey: key,
      score: scores[key as keyof typeof scores] ?? 0,
      delta: SIMULATED_DELTAS[key] ?? 0,
    }));
  }, [scores]);

  // Build highlighted transcript segments
  const fillerWordsSet = useMemo(() => {
    const words = new Set<string>();
    filler_words.forEach((f) => words.add(f.word.toLowerCase()));
    return words;
  }, [filler_words]);

  const goodPhrases = ["well-structured", "clear", "confident", "strong", "excellent", "effective", "engaging"];

  function renderTranscript(text: string) {
    // Split transcript into lines by newline, then process each line
    const lines = text.split("\n");
    return lines.map((line, lineIdx) => {
      // Check if line starts with a timestamp like [0:00] or [1:23]
      const tsMatch = line.match(/^(\[\d+:\d{2}\])\s*(.*)/);
      if (tsMatch) {
        const timestamp = tsMatch[1];
        const rest = tsMatch[2];
        return (
          <div key={lineIdx} style={{ marginBottom: "8px" }}>
            <span style={{ color: "#FFB000", fontWeight: 700, fontSize: "12px" }}>{timestamp} </span>
            {renderHighlightedText(rest)}
          </div>
        );
      }
      if (line.trim() === "") return <div key={lineIdx} style={{ height: "8px" }} />;
      return (
        <div key={lineIdx} style={{ marginBottom: "4px" }}>
          {renderHighlightedText(line)}
        </div>
      );
    });
  }

  function renderHighlightedText(text: string) {
    const words = text.split(/(\s+)/);
    return words.map((word, i) => {
      const cleaned = word.toLowerCase().replace(/[.,!?;:'"]/g, "");
      if (fillerWordsSet.has(cleaned)) {
        return (
          <span
            key={i}
            style={{
              background: "rgba(127, 29, 29, 0.4)",
              borderBottom: "1px solid rgba(239, 68, 68, 0.5)",
              padding: "0 2px",
              color: "#fca5a5",
            }}
          >
            {word}
          </span>
        );
      }
      if (goodPhrases.some((p) => cleaned.includes(p))) {
        return (
          <span key={i} style={{ color: "#4ade80" }}>
            {word}
          </span>
        );
      }
      return <span key={i}>{word}</span>;
    });
  }

  // Count filler words
  const fillerCount = filler_words.length;

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
      {/* Header */}
      <div className="flex justify-between items-start" style={{ marginBottom: "24px" }}>
        <div>
          <h1 style={{ color: "#FFB000", fontWeight: 700, fontSize: "18px", margin: 0, letterSpacing: "0.05em" }}>
            SESSION_REPORT_F4
          </h1>
          <div style={{ color: "#9f8e78", fontSize: "12px", marginTop: "4px" }}>
            {report.title.toUpperCase().replace(/ /g, "_")} // {report.duration_sec}s // PROCESSED_{report.processing_time_sec}s
          </div>
        </div>
        <div className="flex" style={{ gap: "8px" }}>
          <button
            className="bevel-raised cursor-pointer"
            style={{
              background: "#1a1a2e",
              color: "#60a5fa",
              border: "none",
              padding: "6px 14px",
              fontSize: "11px",
              fontWeight: 700,
              letterSpacing: "0.05em",
            }}
          >
            DOWNLOAD_RAW
          </button>
          <button
            className="bevel-raised cursor-pointer"
            style={{
              background: "#1a1a2e",
              color: "#60a5fa",
              border: "none",
              padding: "6px 14px",
              fontSize: "11px",
              fontWeight: 700,
              letterSpacing: "0.05em",
            }}
          >
            RE-ANALYZE
          </button>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid" style={{ gridTemplateColumns: "2fr 1fr", gap: "16px" }}>
        {/* Left Column (col-span-8 equivalent) */}
        <div>
          {/* Metrics Table */}
          <table className="w-full" style={{ borderCollapse: "collapse", fontSize: "13px", marginBottom: "24px" }}>
            <thead>
              <tr
                style={{
                  background: "#002244",
                  borderBottom: "1px solid rgba(96, 165, 250, 0.5)",
                }}
              >
                <th
                  className="text-left"
                  style={{ padding: "8px 12px", color: "#FFB000", fontWeight: 700, fontSize: "11px", letterSpacing: "0.05em" }}
                >
                  DIMENSION
                </th>
                <th
                  style={{ padding: "8px 12px", color: "#FFB000", fontWeight: 700, fontSize: "11px", width: "70px", textAlign: "right", letterSpacing: "0.05em" }}
                >
                  SCORE
                </th>
                <th
                  style={{ padding: "8px 16px", color: "#FFB000", fontWeight: 700, fontSize: "11px", width: "160px", letterSpacing: "0.05em" }}
                >
                  VISUALIZER
                </th>
                <th
                  style={{ padding: "8px 12px", color: "#FFB000", fontWeight: 700, fontSize: "11px", width: "80px", textAlign: "right", letterSpacing: "0.05em" }}
                >
                  DELTA
                </th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((row, i) => {
                const deltaColor = row.delta > 0 ? "#22c55e" : row.delta < 0 ? "#ef4444" : "#9f8e78";
                const deltaStr = row.delta > 0 ? `+${row.delta.toFixed(1)}` : row.delta < 0 ? row.delta.toFixed(1) : "0.0";

                return (
                  <tr
                    key={row.scoreKey}
                    style={{
                      background: i % 2 === 0 ? "#0a0a0a" : "#000",
                      borderBottom: "1px solid rgba(255,255,255,0.05)",
                    }}
                  >
                    <td style={{ padding: "8px 12px", color: "#e2e2e2", fontWeight: 600 }}>{row.displayName}</td>
                    <td style={{ padding: "8px 12px", textAlign: "right", color: scoreColor(row.score), fontWeight: 700 }}>
                      {row.score.toFixed(1)}
                    </td>
                    <td style={{ padding: "8px 16px" }}>
                      <LEDBar value={row.score} segments={10} height={12} />
                    </td>
                    <td style={{ padding: "8px 12px", textAlign: "right", color: deltaColor, fontWeight: 700, fontSize: "12px" }}>
                      {deltaStr}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Waveform Bar Chart Placeholder */}
          <div
            style={{
              background: "#0a0a0a",
              border: "1px solid #1a1a1a",
              padding: "16px",
              position: "relative",
              height: "180px",
              overflow: "hidden",
            }}
          >
            {/* Grid overlay */}
            <div
              style={{
                position: "absolute",
                inset: "16px",
                display: "grid",
                gridTemplateColumns: "repeat(20, 1fr)",
                gridTemplateRows: "repeat(4, 1fr)",
                gap: 0,
              }}
            >
              {Array.from({ length: 80 }, (_, i) => (
                <div key={i} style={{ border: "1px solid rgba(255, 176, 0, 0.06)" }} />
              ))}
            </div>
            {/* Simulated waveform bars */}
            <div
              style={{
                position: "absolute",
                bottom: "16px",
                left: "16px",
                right: "16px",
                height: "120px",
                display: "flex",
                alignItems: "flex-end",
                gap: "2px",
              }}
            >
              {Array.from({ length: 40 }, (_, i) => {
                // Generate a pseudo-waveform from timeline data if available
                const timeIdx = Math.floor((i / 40) * Math.max(1, report.timeline.length));
                const tp = report.timeline[timeIdx];
                const h = tp
                  ? Math.max(8, tp.voice.syllable_rate * 12 + Math.sin(i * 0.7) * 20)
                  : 20 + Math.sin(i * 0.5) * 15 + Math.random() * 10;
                const barHeight = Math.min(120, Math.max(4, h));
                return (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      height: `${barHeight}px`,
                      background: "linear-gradient(to top, rgba(0, 34, 68, 0.8), rgba(96, 165, 250, 0.4))",
                    }}
                  />
                );
              })}
            </div>
            {/* Label */}
            <div style={{ position: "absolute", top: "4px", left: "12px", fontSize: "10px", color: "#9f8e78", fontWeight: 700, letterSpacing: "0.05em" }}>
              WAVEFORM_ANALYSIS
            </div>
          </div>
        </div>

        {/* Right Column (col-span-4 equivalent) */}
        <div>
          {/* Transcript Analysis Header */}
          <div
            style={{
              background: "#002244",
              padding: "8px 12px",
              marginBottom: "0",
              borderBottom: "1px solid rgba(96, 165, 250, 0.5)",
            }}
          >
            <span style={{ color: "#FFB000", fontWeight: 700, fontSize: "11px", letterSpacing: "0.05em" }}>
              FULL_TRANSCRIPT_ANALYSIS
            </span>
          </div>

          {/* Transcript Body */}
          <div
            style={{
              background: "#0a0a0a",
              padding: "12px",
              fontSize: "12px",
              lineHeight: "1.7",
              color: "#9f8e78",
              maxHeight: "360px",
              overflowY: "auto",
              borderLeft: "1px solid #1a1a1a",
              borderRight: "1px solid #1a1a1a",
            }}
            className="no-scrollbar"
          >
            {renderTranscript(transcript)}
          </div>

          {/* Key Insights */}
          <div
            style={{
              background: "#0e0e0e",
              padding: "12px",
              borderTop: "1px solid #1a1a1a",
              borderLeft: "1px solid #1a1a1a",
              borderRight: "1px solid #1a1a1a",
            }}
          >
            <div style={{ fontSize: "11px", color: "#FFB000", fontWeight: 700, marginBottom: "8px", letterSpacing: "0.05em" }}>
              KEY_INSIGHTS
            </div>
            {fillerCount > 0 && (
              <div className="flex items-center" style={{ gap: "8px", marginBottom: "6px", fontSize: "12px" }}>
                <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#ef4444" }}>warning</span>
                <span style={{ color: "#fca5a5" }}>
                  {fillerCount} filler word{fillerCount !== 1 ? "s" : ""} detected — consider reducing verbal crutches
                </span>
              </div>
            )}
            <div className="flex items-center" style={{ gap: "8px", marginBottom: "6px", fontSize: "12px" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#22c55e" }}>check_circle</span>
              <span style={{ color: "#4ade80" }}>
                {scores.vocal_clarity >= 70 ? "Good vocal clarity and tone consistency" : "Vocal tone shows room for improvement"}
              </span>
            </div>
            <div className="flex items-center" style={{ gap: "8px", fontSize: "12px" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#22c55e" }}>check_circle</span>
              <span style={{ color: "#4ade80" }}>
                {scores.content_structure >= 70 ? "Strong content structure and cadence" : "Content structure could be tightened"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
