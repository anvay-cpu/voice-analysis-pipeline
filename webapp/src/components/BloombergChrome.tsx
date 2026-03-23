"use client";

import { useEffect, useState, useCallback } from "react";
import type { SpeechReport, ViewType, TopNavMode } from "@/lib/types";
import { getReport, enrichReport } from "@/lib/api";
import NavPanel from "./NavPanel";
import FunctionKeyBar from "./FunctionKeyBar";
import DashboardView from "./DashboardView";
import SummaryView from "./SummaryView";
import TimelineView from "./TimelineView";
import CoachingView from "./CoachingView";
import DataView from "./DataView";
import VideoView from "./VideoView";
import PracticeView from "./PracticeView";
import SettingsView from "./SettingsView";
import HistoryView from "./HistoryView";
import UploadView from "./UploadView";
import ProcessingView from "./ProcessingView";

const TOP_NAV: TopNavMode[] = ["LIVE_FEED", "ANALYSIS", "HISTORY", "SETTINGS"];

interface Props {
  report: SpeechReport | null;
}

export default function BloombergChrome({ report: initialReport }: Props) {
  const [report, setReport] = useState<SpeechReport | null>(initialReport);
  const [activeView, setActiveView] = useState<ViewType>("dashboard");
  const [activeTopNav, setActiveTopNav] = useState<TopNavMode>(
    initialReport ? "ANALYSIS" : "LIVE_FEED"
  );

  // Processing state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    const keyMap: Record<string, ViewType> = {
      F1: "dashboard",
      F2: "summary",
      F3: "coaching",
      F4: "data",
      F5: "video",
      F6: "practice",
    };
    if (keyMap[e.key]) {
      e.preventDefault();
      setActiveView(keyMap[e.key]);
      setActiveTopNav("ANALYSIS");
    }
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  function handleTopNavSwitch(mode: TopNavMode) {
    setActiveTopNav(mode);
    if (mode === "ANALYSIS") setActiveView("dashboard");
  }

  function handleUploadStarted(jobId: string, sessionId: string) {
    setActiveJobId(jobId);
    setActiveSessionId(sessionId);
    setIsProcessing(true);
    setActiveTopNav("LIVE_FEED");
  }

  async function handleProcessingComplete(sessionId: string) {
    setIsProcessing(false);
    setActiveJobId(null);
    try {
      const newReport = await getReport(sessionId);
      setReport(newReport);
      setActiveTopNav("ANALYSIS");
      setActiveView("dashboard");
      // Enrich coaching in background via local claude -p (Max subscription)
      enrichReport(newReport).then((enriched) => {
        if (enriched !== newReport) setReport(enriched);
      });
    } catch {
      setActiveTopNav("HISTORY");
    }
  }

  function handleSessionSelect(sessionId: string) {
    getReport(sessionId).then((r) => {
      setReport(r);
      setActiveTopNav("ANALYSIS");
      setActiveView("dashboard");
      // Enrich coaching in background via local claude -p
      enrichReport(r).then((enriched) => {
        if (enriched !== r) setReport(enriched);
      });
    }).catch(() => {});
  }

  const isReportMode = activeTopNav === "ANALYSIS";
  const isLiveFeed = activeTopNav === "LIVE_FEED";

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: "#000" }}>
      {/* TopAppBar */}
      <header
        className="no-print flex items-center justify-between shrink-0"
        style={{
          background: "#131313",
          borderBottom: "1px solid rgba(159, 142, 120, 0.3)",
          height: "22px",
          padding: "0 8px",
          fontSize: "13px",
          lineHeight: "22px",
          fontWeight: 700,
          letterSpacing: "-0.02em",
        }}
      >
        <div className="flex items-center" style={{ gap: "16px" }}>
          <span style={{ color: "#FF8C00", fontWeight: 900, textTransform: "uppercase", fontSize: "13px" }}>
            SPEECH_COACH_V1.0
          </span>
          <nav className="flex" style={{ gap: "16px" }}>
            {TOP_NAV.map((item) => (
              <span
                key={item}
                onClick={() => handleTopNavSwitch(item)}
                className="cursor-pointer"
                style={{
                  color: activeTopNav === item ? "#FFB000" : "#888888",
                  borderBottom: activeTopNav === item ? "2px solid #FFB000" : "2px solid transparent",
                  fontSize: "13px",
                }}
              >
                {item}
              </span>
            ))}
          </nav>
        </div>
        <div className="flex items-center" style={{ gap: "8px" }}>
          {isProcessing && (
            <span style={{ color: "#FFB000", fontSize: "11px", marginRight: "8px" }}>
              PROCESSING...
            </span>
          )}
          <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#888888", cursor: "pointer" }}>settings_input_component</span>
          <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#888888", cursor: "pointer" }}>terminal</span>
          <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "#888888", cursor: "pointer" }}>account_circle</span>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 min-h-0">
        <NavPanel
          active={activeView}
          onSwitch={(v) => { setActiveView(v); setActiveTopNav("ANALYSIS"); }}
          topNav={activeTopNav}
          onStartSession={() => { setActiveTopNav("LIVE_FEED"); setIsProcessing(false); setActiveJobId(null); }}
        />
        <div className="flex flex-col flex-1 min-h-0">
          {isReportMode && report ? (
            <>
              {activeView === "dashboard" && <DashboardView report={report} />}
              {activeView === "summary" && <SummaryView report={report} />}
              {activeView === "timeline" && <TimelineView report={report} />}
              {activeView === "coaching" && <CoachingView report={report} />}
              {activeView === "data" && <DataView report={report} />}
              {activeView === "video" && <VideoView report={report} />}
              {activeView === "practice" && <PracticeView report={report} />}
            </>
          ) : isReportMode && !report ? (
            <div className="flex-1 flex items-center justify-center" style={{ background: "#000" }}>
              <div className="text-center">
                <div style={{ fontSize: "14px", color: "#9f8e78", marginBottom: "16px" }}>NO_REPORT_LOADED</div>
                <button
                  onClick={() => setActiveTopNav("LIVE_FEED")}
                  style={{
                    padding: "8px 24px",
                    background: "linear-gradient(to bottom, #FF8C00, #93000a)",
                    color: "#fff",
                    fontWeight: 700,
                    fontSize: "13px",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  UPLOAD_NEW_SESSION
                </button>
              </div>
            </div>
          ) : isLiveFeed ? (
            isProcessing && activeJobId ? (
              <ProcessingView
                jobId={activeJobId}
                sessionId={activeSessionId || ""}
                onComplete={handleProcessingComplete}
              />
            ) : (
              <UploadView onUploadStarted={handleUploadStarted} />
            )
          ) : activeTopNav === "HISTORY" ? (
            <HistoryView onSessionSelect={handleSessionSelect} />
          ) : activeTopNav === "SETTINGS" ? (
            <SettingsView />
          ) : null}
        </div>
      </div>

      <FunctionKeyBar active={activeView} onSwitch={(v) => { setActiveView(v); setActiveTopNav("ANALYSIS"); }} />
    </div>
  );
}
