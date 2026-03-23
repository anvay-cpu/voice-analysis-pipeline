"use client";

import type { ViewType, TopNavMode } from "@/lib/types";

const NAV_ITEMS: { key: ViewType; icon: string; label: string }[] = [
  { key: "dashboard", icon: "dashboard", label: "DASH" },
  { key: "summary", icon: "mic", label: "VOICE" },
  { key: "coaching", icon: "graphic_eq", label: "TONE" },
  { key: "data", icon: "speed", label: "PACE" },
  { key: "video", icon: "waves", label: "CADENCE" },
  { key: "practice", icon: "save_alt", label: "EXPORT" },
];

export default function NavPanel({
  active,
  onSwitch,
  topNav,
  onStartSession,
}: {
  active: ViewType;
  onSwitch: (v: ViewType) => void;
  topNav?: TopNavMode;
  onStartSession?: () => void;
}) {
  return (
    <aside
      className="shrink-0 flex flex-col"
      style={{
        width: "160px",
        background: "#1b1b1b",
        borderRight: "1px solid #000000",
        fontSize: "14px",
        textTransform: "uppercase",
        letterSpacing: "-0.03em",
      }}
    >
      {/* Branding */}
      <div style={{ padding: "16px", borderBottom: "1px solid #000" }}>
        <div style={{ color: "#FF8C00", fontWeight: 700, fontSize: "18px", marginBottom: "4px" }}>
          COACH_OS
        </div>
        <div style={{ fontSize: "10px", opacity: 0.6, color: "#e2e2e2" }}>V_2.4.0</div>
        <div className="flex items-center" style={{ gap: "8px", marginTop: "16px" }}>
          <div
            className="flex items-center justify-center"
            style={{
              width: "32px",
              height: "32px",
              background: "#353535",
              border: "1px solid #524533",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#e2e2e2" }}>person</span>
          </div>
          <div style={{ fontSize: "9px", lineHeight: "1.3", color: "#e2e2e2" }}>
            OPERATOR_ID<br />01_J_DOE
          </div>
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex-1" style={{ marginTop: "8px" }}>
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onSwitch(item.key)}
              className="flex items-center w-full text-left cursor-pointer"
              style={{
                gap: "12px",
                padding: "8px 16px",
                fontSize: "14px",
                fontWeight: isActive ? 400 : 400,
                color: isActive ? "#60a5fa" : "#FFB000",
                opacity: isActive ? 1 : 0.8,
                background: isActive ? "rgba(30, 58, 138, 0.4)" : "transparent",
                borderTop: isActive ? "1px solid rgba(96, 165, 250, 0.3)" : "1px solid transparent",
                borderBottom: isActive ? "1px solid #000" : "1px solid transparent",
                borderLeft: "none",
                borderRight: "none",
                boxShadow: isActive ? "inset 0 1px 0 rgba(255,255,255,0.1)" : "none",
                textTransform: "uppercase",
                letterSpacing: "-0.03em",
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>{item.icon}</span>
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div style={{ padding: "8px" }}>
        <button
          onClick={onStartSession}
          className="w-full cursor-pointer bevel-raised"
          style={{
            padding: "8px 0",
            background: "#FF8C00",
            color: "#000",
            fontWeight: 700,
            fontSize: "12px",
            border: "none",
            textTransform: "uppercase",
          }}
        >
          START_SESSION
        </button>
      </div>
      <div style={{ borderTop: "1px solid #000", padding: "8px", background: "#1b1b1b" }}>
        <div className="flex items-center cursor-pointer" style={{ gap: "8px", padding: "4px 8px", fontSize: "11px", color: "#FFB000", opacity: 0.6 }}>
          <span className="material-symbols-outlined" style={{ fontSize: "12px" }}>warning</span> ALERTS
        </div>
        <div className="flex items-center cursor-pointer" style={{ gap: "8px", padding: "4px 8px", fontSize: "11px", color: "#FFB000", opacity: 0.6 }}>
          <span className="material-symbols-outlined" style={{ fontSize: "12px" }}>receipt_long</span> LOGS
        </div>
      </div>
    </aside>
  );
}
