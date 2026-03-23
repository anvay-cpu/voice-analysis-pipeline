"use client";

import type { SpeechReport } from "@/lib/types";
import { scoreColor } from "@/lib/utils";
import LEDBar from "./LEDBar";

/* ── Dimension mapping: report keys -> display labels (0-10 scale) ── */
const DIMENSION_MAP: { key: keyof SpeechReport["scores"]; label: string }[] = [
  { key: "vocal_clarity", label: "VOCAL" },
  { key: "content_structure", label: "PACE" },
  { key: "emotional_expressiveness", label: "TONE" },
  { key: "body_language", label: "BODY" },
  { key: "audience_engagement", label: "ENERGY" },
  { key: "regime_adaptability", label: "FLOW" },
];

/* ── Static recent speeches table data ── */
const RECENT_SPEECHES = [
  { id: "SPX-0847", ts: "2026-03-22 14:32", topic: "Q3_EARNINGS_REVIEW", dur: "18:42", score: 8.4, status: "COMPLETE" },
  { id: "SPX-0846", ts: "2026-03-21 09:15", topic: "TEAM_STANDUP", dur: "06:11", score: 7.1, status: "COMPLETE" },
  { id: "SPX-0845", ts: "2026-03-20 16:50", topic: "PRODUCT_LAUNCH", dur: "24:03", score: 9.2, status: "COMPLETE" },
  { id: "SPX-0844", ts: "2026-03-19 11:00", topic: "INVESTOR_PITCH", dur: "12:37", score: 6.8, status: "REVIEW" },
  { id: "SPX-0843", ts: "2026-03-18 08:45", topic: "ALL_HANDS", dur: "31:22", score: 7.9, status: "COMPLETE" },
  { id: "SPX-0842", ts: "2026-03-17 14:10", topic: "TRAINING_SESSION", dur: "45:08", score: 5.3, status: "FLAGGED" },
  { id: "SPX-0841", ts: "2026-03-16 10:20", topic: "BOARD_BRIEFING", dur: "22:15", score: 8.1, status: "COMPLETE" },
  { id: "SPX-0840", ts: "2026-03-15 15:30", topic: "KEYNOTE_DRAFT", dur: "34:50", score: 7.6, status: "COMPLETE" },
];

/* ── Sparkline SVG renderer ── */
function SparklineSVG({ data, color }: { data: number[]; color: string }) {
  const w = 140, h = 32;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - 4 - ((v - min) / range) * (h - 8)}`)
    .join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

/* ── Generate simulated sparkline data ── */
function makeSparkData(current: number): number[] {
  const pts: number[] = [];
  let v = current - 1.5 + Math.random();
  for (let i = 0; i < 12; i++) {
    v += (Math.random() - 0.4) * 0.8;
    v = Math.max(1, Math.min(9.5, v));
    pts.push(v);
  }
  pts.push(current);
  return pts;
}

/* ── Status badge color ── */
function statusStyle(status: string): { color: string; bg: string } {
  if (status === "COMPLETE") return { color: "#4ade80", bg: "rgba(74,222,128,0.1)" };
  if (status === "REVIEW") return { color: "#FFB000", bg: "rgba(255,176,0,0.1)" };
  if (status === "FLAGGED") return { color: "#ef4444", bg: "rgba(239,68,68,0.1)" };
  return { color: "#888", bg: "rgba(136,136,136,0.1)" };
}

/* ══════════════════════════════════════════════════════════════════════ */

export default function DashboardView({ report }: { report: SpeechReport }) {
  const { scores } = report;

  /* Map dimension scores to 0-10 scale */
  const dims = DIMENSION_MAP.map((d) => ({
    label: d.label,
    raw: scores[d.key] as number,
    score10: parseFloat(((scores[d.key] as number) / 10).toFixed(1)),
  }));

  /* Sparkline trend data */
  const trendSeries = [
    { label: "Vocal Stability", value: dims[0].score10, color: "#FFB000" },
    { label: "Energy Arc", value: dims[4].score10, color: "#4ade80" },
    { label: "Pace Variance", value: dims[1].score10, color: "#60a5fa" },
  ];

  /* Heatmap rows — M, T, W with 12 cells each */
  const heatRows = ["M", "T", "W"];
  const heatColors = ["bg-[#FFB000]", "bg-[#353535]", "bg-[#fd8b00]"];

  /* Seeded pseudo-random heatmap pattern */
  const heatmapGrid = heatRows.map((_, ri) =>
    Array.from({ length: 12 }, (_, ci) => {
      const idx = (ri * 12 + ci) % 3;
      // Cycle through the three color classes for visual variety
      const pick = (ri + ci) % 5;
      if (pick === 0 || pick === 3) return heatColors[0]; // primary-container / amber
      if (pick === 1 || pick === 4) return heatColors[1]; // surface-variant / #353535
      return heatColors[2]; // secondary-container / #fd8b00
    })
  );

  return (
    <div className="flex flex-1 min-h-0">
      {/* ── MAIN CONTENT ── */}
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
        <div className="grid grid-cols-12 gap-4">

          {/* ════════════════════════════════════════════════════════ */}
          {/* col-span-8: PERFORMANCE_SUMMARY_MATRIX                 */}
          {/* ════════════════════════════════════════════════════════ */}
          <div className="col-span-8">
            <div className="flex items-center justify-between" style={{ marginBottom: "4px" }}>
              <span
                className="font-bold tracking-widest uppercase"
                style={{ fontSize: "11px", color: "#FF8C00", letterSpacing: "0.08em" }}
              >
                PERFORMANCE_SUMMARY_MATRIX
              </span>
              <span style={{ fontSize: "10px", color: "#9f8e78" }}>
                [実時監控] REAL-TIME FEED
              </span>
            </div>

            <div className="bevel-recessed" style={{ padding: "12px" }}>
              {dims.map((d) => {
                const filled = Math.round(d.score10);
                return (
                  <div
                    key={d.label}
                    className="flex items-center"
                    style={{ height: "28px", gap: "12px" }}
                  >
                    {/* Label */}
                    <span
                      className="font-bold uppercase"
                      style={{
                        width: "56px",
                        fontSize: "11px",
                        color: "#9f8e78",
                        letterSpacing: "0.04em",
                        flexShrink: 0,
                      }}
                    >
                      {d.label}
                    </span>

                    {/* 10-segment LED bar */}
                    <div className="flex" style={{ gap: "4px", flex: 1 }}>
                      {Array.from({ length: 10 }, (_, i) => (
                        <div
                          key={i}
                          className={i < filled ? "segmented-bar-fill" : ""}
                          style={{
                            flex: 1,
                            height: "12px",
                            ...(i < filled
                              ? {}
                              : { opacity: 0.2, backgroundColor: "#353535" }),
                          }}
                        />
                      ))}
                    </div>

                    {/* Score */}
                    <span
                      className="amber-glow font-bold"
                      style={{
                        width: "40px",
                        textAlign: "right",
                        fontSize: "14px",
                        color: "#FFB000",
                        flexShrink: 0,
                      }}
                    >
                      {d.score10.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ════════════════════════════════════════════════════════ */}
          {/* col-span-4: TREND_VECTORS                              */}
          {/* ════════════════════════════════════════════════════════ */}
          <div className="col-span-4">
            <span
              className="font-bold tracking-widest uppercase"
              style={{ fontSize: "11px", color: "#FF8C00", letterSpacing: "0.08em", display: "block", marginBottom: "4px" }}
            >
              TREND_VECTORS
            </span>

            <div className="bevel-recessed" style={{ padding: "12px" }}>
              {trendSeries.map((t) => {
                const data = makeSparkData(t.value);
                return (
                  <div key={t.label} style={{ marginBottom: "10px" }}>
                    <div
                      style={{
                        fontSize: "10px",
                        color: "#9f8e78",
                        fontWeight: 700,
                        marginBottom: "2px",
                        textTransform: "uppercase",
                      }}
                    >
                      {t.label}
                    </div>
                    <SparklineSVG data={data} color={t.color} />
                  </div>
                );
              })}
            </div>
          </div>

          {/* ════════════════════════════════════════════════════════ */}
          {/* col-span-12: RECENT SPEECHES TABLE                     */}
          {/* ════════════════════════════════════════════════════════ */}
          <div className="col-span-12">
            <span
              className="font-bold tracking-widest uppercase"
              style={{ fontSize: "11px", color: "#FF8C00", letterSpacing: "0.08em", display: "block", marginBottom: "4px" }}
            >
              RECENT_SPEECHES
            </span>

            <table className="w-full" style={{ borderCollapse: "collapse", fontSize: "12px" }}>
              <thead>
                <tr style={{ background: "#2a2a2a", height: "28px" }}>
                  <th className="text-left" style={{ padding: "0 8px", color: "#FF8C00", fontSize: "10px", fontWeight: 700 }}>ID_ENTRY</th>
                  <th className="text-left" style={{ padding: "0 8px", color: "#FF8C00", fontSize: "10px", fontWeight: 700 }}>TIMESTAMP</th>
                  <th className="text-left" style={{ padding: "0 8px", color: "#FF8C00", fontSize: "10px", fontWeight: 700 }}>TOPIC_TAG</th>
                  <th className="text-right" style={{ padding: "0 8px", color: "#FF8C00", fontSize: "10px", fontWeight: 700 }}>DURATION</th>
                  <th className="text-right" style={{ padding: "0 8px", color: "#FF8C00", fontSize: "10px", fontWeight: 700 }}>SCORE</th>
                  <th className="text-center" style={{ padding: "0 8px", color: "#FF8C00", fontSize: "10px", fontWeight: 700 }}>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {RECENT_SPEECHES.map((row, i) => {
                  const ss = statusStyle(row.status);
                  return (
                    <tr
                      key={row.id}
                      className="event-item"
                      style={{
                        background: i % 2 === 0 ? "#000" : "#0A0A0A",
                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                        height: "30px",
                        cursor: "pointer",
                      }}
                    >
                      <td style={{ padding: "0 8px", color: "#60a5fa" }}>{row.id}</td>
                      <td style={{ padding: "0 8px", color: "#9f8e78" }}>{row.ts}</td>
                      <td style={{ padding: "0 8px", color: "#FFB000" }}>{row.topic}</td>
                      <td style={{ padding: "0 8px", color: "#9f8e78", textAlign: "right" }}>{row.dur}</td>
                      <td className="amber-glow font-bold" style={{ padding: "0 8px", color: "#FFB000", textAlign: "right" }}>{row.score.toFixed(1)}</td>
                      <td style={{ padding: "0 8px", textAlign: "center" }}>
                        <span style={{ fontSize: "10px", fontWeight: 700, color: ss.color, background: ss.bg, padding: "1px 6px" }}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ════════════════════════════════════════════════════════ */}
          {/* col-span-12: TEMPORAL_INTENSITY_MAP                    */}
          {/* ════════════════════════════════════════════════════════ */}
          <div className="col-span-12">
            <span
              className="font-bold tracking-widest uppercase"
              style={{ fontSize: "11px", color: "#FF8C00", letterSpacing: "0.08em", display: "block", marginBottom: "4px" }}
            >
              TEMPORAL_INTENSITY_MAP
            </span>

            <div className="bevel-recessed" style={{ padding: "8px" }}>
              {heatmapGrid.map((row, ri) => (
                <div key={ri} className="flex items-center" style={{ gap: "4px", marginBottom: ri < 2 ? "4px" : "0" }}>
                  {/* Row label */}
                  <span style={{ width: "16px", fontSize: "10px", color: "#9f8e78", fontWeight: 700, textAlign: "center", flexShrink: 0 }}>
                    {heatRows[ri]}
                  </span>
                  {/* 12 cells */}
                  <div className="flex" style={{ gap: "3px", flex: 1 }}>
                    {row.map((bgClass, ci) => (
                      <div
                        key={ci}
                        className={bgClass}
                        style={{ flex: 1, height: "18px" }}
                      />
                    ))}
                  </div>
                </div>
              ))}

              {/* Time axis */}
              <div className="flex justify-between" style={{ marginTop: "4px", marginLeft: "20px", fontSize: "9px", color: "#9f8e78" }}>
                {Array.from({ length: 7 }, (_, i) => (
                  <span key={i}>{String(6 + i * 2).padStart(2, "0")}:00</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── RIGHT PANEL (320px) ── */}
      <aside
        className="shrink-0 overflow-y-auto no-scrollbar"
        style={{ width: "320px", borderLeft: "1px solid #000", background: "#0e0e0e", padding: "16px" }}
      >
        {/* ═══ GLOBAL_METRICS ═══ */}
        <h2
          className="font-bold"
          style={{
            color: "#FF8C00",
            fontSize: "11px",
            textTransform: "uppercase",
            marginBottom: "16px",
            borderBottom: "1px solid rgba(82,69,51,0.3)",
            paddingBottom: "4px",
            letterSpacing: "0.08em",
          }}
        >
          GLOBAL_METRICS
        </h2>

        <div className="bevel-recessed" style={{ padding: "10px 12px", marginBottom: "24px" }}>
          {[
            { label: "Total Sessions", value: "142" },
            { label: "Average Score", value: "7.84" },
            { label: "Peak Performance", value: "9.52" },
            { label: "Fluency Index", value: "92%" },
          ].map((m) => (
            <div key={m.label} className="flex justify-between items-center" style={{ height: "26px" }}>
              <span style={{ color: "#9f8e78", fontSize: "11px" }}>{m.label}</span>
              <span className="amber-glow font-bold" style={{ color: "#FFB000", fontSize: "13px" }}>{m.value}</span>
            </div>
          ))}
        </div>

        {/* ═══ AI_GROWTH_SUGGESTIONS ═══ */}
        <h2
          className="font-bold"
          style={{
            color: "#FF8C00",
            fontSize: "11px",
            textTransform: "uppercase",
            marginBottom: "16px",
            borderBottom: "1px solid rgba(82,69,51,0.3)",
            paddingBottom: "4px",
            letterSpacing: "0.08em",
          }}
        >
          AI_GROWTH_SUGGESTIONS
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginBottom: "24px" }}>
          {[
            { num: "01", text: "Increase pause duration between key transitions to improve FLOW score above 5.0 threshold." },
            { num: "02", text: "Vocal pitch variance is low during data-heavy segments — modulate TONE to maintain engagement." },
            { num: "03", text: "ENERGY peaks in opening 2 minutes then drops — sustain intensity through middle sections." },
          ].map((sug) => (
            <div key={sug.num} className="bevel-recessed" style={{ padding: "8px 10px" }}>
              <div className="flex" style={{ gap: "8px" }}>
                <span
                  className="amber-glow font-bold"
                  style={{ color: "#FFB000", fontSize: "14px", flexShrink: 0, width: "22px" }}
                >
                  {sug.num}
                </span>
                <span style={{ fontSize: "11px", color: "#ffd597", lineHeight: "1.5", opacity: 0.85 }}>
                  {sug.text}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* ═══ SYSTEM_ALERTS ═══ */}
        <h2
          className="font-bold"
          style={{
            color: "#FF8C00",
            fontSize: "11px",
            textTransform: "uppercase",
            marginBottom: "16px",
            borderBottom: "1px solid rgba(82,69,51,0.3)",
            paddingBottom: "4px",
            letterSpacing: "0.08em",
          }}
        >
          SYSTEM_ALERTS
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          {/* Red alert — CADENCE_WARNING */}
          <div
            style={{
              border: "1px solid #ef4444",
              background: "rgba(239,68,68,0.1)",
              padding: "6px 10px",
              fontSize: "11px",
            }}
          >
            <span className="font-bold" style={{ color: "#ef4444" }}>CADENCE_WARNING</span>
            <span style={{ color: "#9f8e78", marginLeft: "8px" }}>Syllable rate exceeded 5.2/s in 3 segments</span>
          </div>

          {/* Amber alert — Update */}
          <div
            style={{
              border: "1px solid #FFB000",
              background: "rgba(255,176,0,0.1)",
              padding: "6px 10px",
              fontSize: "11px",
            }}
          >
            <span className="font-bold" style={{ color: "#FFB000" }}>UPDATE</span>
            <span style={{ color: "#9f8e78", marginLeft: "8px" }}>Pipeline v2.5 available — improved TONE analysis</span>
          </div>

          {/* Gray alert — Backup */}
          <div
            style={{
              border: "1px solid #6b7280",
              background: "rgba(107,114,128,0.1)",
              padding: "6px 10px",
              fontSize: "11px",
            }}
          >
            <span className="font-bold" style={{ color: "#6b7280" }}>BACKUP</span>
            <span style={{ color: "#9f8e78", marginLeft: "8px" }}>Session data archived — 142 records synced</span>
          </div>
        </div>
      </aside>
    </div>
  );
}
