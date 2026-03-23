"use client";

import type { ViewType } from "@/lib/types";

const TABS: { key: ViewType; icon: string; label: string; fkey: string }[] = [
  { key: "dashboard", icon: "dashboard", label: "DASH", fkey: "F1" },
  { key: "summary", icon: "mic", label: "VOICE", fkey: "F2" },
  { key: "coaching", icon: "volume_up", label: "COACH", fkey: "F3" },
  { key: "data", icon: "query_stats", label: "DATA", fkey: "F4" },
  { key: "video", icon: "videocam", label: "VIDEO", fkey: "F5" },
  { key: "practice", icon: "fitness_center", label: "TRAIN", fkey: "F6" },
];

export default function FunctionKeyBar({
  active,
  onSwitch,
}: {
  active: ViewType;
  onSwitch: (v: ViewType) => void;
}) {
  return (
    <footer
      className="no-print flex justify-center items-center shrink-0"
      style={{
        height: "32px",
        background: "#002244",
        borderTop: "1px solid #000000",
        gap: "4px",
        padding: "0 4px",
        boxShadow: "0 -1px 0 rgba(255,255,255,0.1)",
      }}
    >
      {TABS.map((tab) => {
        const isActive = active === tab.key;
        return (
          <button
            key={tab.key}
            onClick={() => onSwitch(tab.key)}
            className="cursor-pointer flex items-center justify-center"
            style={{
              flex: "1 1 0",
              maxWidth: "120px",
              height: "100%",
              gap: "4px",
              border: "none",
              fontSize: "11px",
              fontWeight: 700,
              fontFamily: "'B612 Mono', monospace",
              color: isActive ? "#000000" : "#93c5fd",
              background: isActive ? "#FFB000" : "transparent",
              borderTop: isActive ? "1px solid rgba(255,255,255,0.4)" : "1px solid transparent",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>{tab.icon}</span>
            <span>[{tab.fkey}] {tab.label}</span>
          </button>
        );
      })}
    </footer>
  );
}
