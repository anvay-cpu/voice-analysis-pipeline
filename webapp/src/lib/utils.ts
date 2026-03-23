export function scoreColor(score: number): string {
  if (score >= 80) return "#22c55e";
  if (score >= 60) return "#ffb000";
  return "#ef4444";
}

export function scoreGrade(score: number): string {
  if (score >= 80) return "GOOD";
  if (score >= 60) return "WARN";
  return "POOR";
}

export function scoreBarChars(score: number, total = 10): string {
  const filled = Math.round((score / 100) * total);
  return "\u2588".repeat(filled) + "\u2591".repeat(total - filled);
}

export function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function heatmapChar(value: number): { char: string; className: string } {
  if (value >= 0.75) return { char: "\u2588", className: "heatmap-good" };
  if (value >= 0.50) return { char: "\u2593", className: "heatmap-ok" };
  if (value >= 0.25) return { char: "\u2592", className: "heatmap-poor" };
  return { char: "\u2591", className: "heatmap-critical" };
}

export function eventIcon(event: { type: string; status: string }): { icon: string; color: string } {
  if (event.type === "transition") return { icon: "\u2500\u2500", color: "#60a5fa" };
  if (event.status === "MISMATCH") return { icon: "\u26A0", color: "#ffb000" };
  if (event.status === "GOOD") return { icon: "\u2713", color: "#22c55e" };
  if (event.status === "SLOW") return { icon: "\u2717", color: "#ef4444" };
  return { icon: "\u2022", color: "#6b7280" };
}

export function dimLabel(key: string): string {
  const labels: Record<string, string> = {
    vocal_clarity: "VOCAL",
    body_language: "BODY",
    content_structure: "CONTENT",
    audience_engagement: "ENGAGE",
    emotional_expressiveness: "EXPRESS",
    regime_adaptability: "ADAPT",
  };
  return labels[key] || key.toUpperCase();
}

export function dimFullLabel(key: string): string {
  const labels: Record<string, string> = {
    vocal_clarity: "Vocal Clarity",
    body_language: "Body Language",
    content_structure: "Content Structure",
    audience_engagement: "Audience Engagement",
    emotional_expressiveness: "Emotional Express.",
    regime_adaptability: "Regime Adaptability",
  };
  return labels[key] || key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function regimeColor(regime: string): string {
  const colors: Record<string, string> = {
    Introduction: "#1e3a5f",
    "Main Argument": "#1a4731",
    "Data Presentation": "#78350f",
    Anecdote: "#5b2d0c",
    "Call to Action": "#7f1d1d",
    Conclusion: "#1e3a5f",
    "Q&A": "#4b5563",
  };
  return colors[regime] || "#4b5563";
}

export function dimIcon(key: string): string {
  const icons: Record<string, string> = {
    vocal_clarity: "mic",
    body_language: "accessibility_new",
    content_structure: "article",
    audience_engagement: "groups",
    emotional_expressiveness: "mood",
    regime_adaptability: "swap_horiz",
  };
  return icons[key] || "analytics";
}
