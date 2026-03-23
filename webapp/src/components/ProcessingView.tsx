"use client";

import { useState, useEffect, useRef } from "react";
import { getJobStatus } from "@/lib/api";
import type { JobStatus } from "@/lib/types";

interface Props {
  jobId: string;
  sessionId: string;
  onComplete: (sessionId: string) => void;
}

const PIPELINE_STEP_NAMES: Record<string, string> = {
  QUEUED: "INITIALIZATION",
  VOICE_ANALYSIS: "VOICE_PIPELINE_M1",
  BODY_LANGUAGE_ANALYSIS: "BODY_LANGUAGE_PIPELINE_M2",
  CONTENT_ANALYSIS: "CONTENT_ANALYSIS_M3",
  MULTIMODAL_FUSION: "MULTIMODAL_FUSION",
  DIMENSION_SCORING: "DIMENSION_SCORING",
  COACHING_GENERATION: "COACHING_GENERATION",
  CHART_GENERATION: "CHART_GENERATION",
  REPORT_ASSEMBLY: "REPORT_ASSEMBLY",
  COMPLETE: "COMPLETE",
};

const ALL_STEPS = [
  "VOICE_ANALYSIS",
  "BODY_LANGUAGE_ANALYSIS",
  "CONTENT_ANALYSIS",
  "MULTIMODAL_FUSION",
  "DIMENSION_SCORING",
  "COACHING_GENERATION",
  "CHART_GENERATION",
  "REPORT_ASSEMBLY",
];

function stepStatusColor(status: "complete" | "running" | "pending"): string {
  if (status === "complete") return "#4ade80";
  if (status === "running") return "#FFB000";
  return "#353535";
}

export default function ProcessingView({ jobId, sessionId, onComplete }: Props) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [logs, setLogs] = useState<Array<{ time: string; level: string; msg: string }>>([]);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    function addLog(level: string, msg: string) {
      const now = new Date();
      const time = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;
      setLogs((prev) => [...prev.slice(-20), { time, level, msg }]);
    }

    addLog("INFO", `Job ${jobId} started`);
    addLog("SYS", `Session: ${sessionId}`);

    async function poll() {
      try {
        const status = await getJobStatus(jobId);
        setJob(status);

        if (status.step) {
          addLog("PROC", `Step: ${PIPELINE_STEP_NAMES[status.step] || status.step} (${status.progress.toFixed(1)}%)`);
        }

        if (status.status === "complete") {
          addLog("INFO", "Pipeline complete!");
          if (pollRef.current) clearInterval(pollRef.current);
          setTimeout(() => onComplete(sessionId), 1500);
        } else if (status.status === "failed") {
          addLog("ERROR", status.error || "Pipeline failed");
          setError(status.error || "Pipeline processing failed");
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        addLog("WARN", "Connection lost, retrying...");
      }
    }

    poll();
    pollRef.current = setInterval(poll, 2000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId, sessionId, onComplete]);

  const progress = job?.progress || 0;
  const currentStep = job?.step || "QUEUED";

  function getStepStatus(step: string): "complete" | "running" | "pending" {
    if (!job) return "pending";
    const stepIdx = ALL_STEPS.indexOf(step);
    const currentIdx = ALL_STEPS.indexOf(currentStep);
    if (currentStep === "COMPLETE") return "complete";
    if (stepIdx < currentIdx) return "complete";
    if (stepIdx === currentIdx) return "running";
    return "pending";
  }

  return (
    <div className="flex flex-1 min-h-0">
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
        <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700, letterSpacing: "-0.03em", marginBottom: "16px" }}>
          SYSTEM_PROCESSING_QUEUE
        </div>

        {/* Progress Bar */}
        <div style={{ marginBottom: "24px" }}>
          <div className="flex justify-between items-center" style={{ marginBottom: "4px" }}>
            <span style={{ fontSize: "11px", color: "#9f8e78" }}>PROGRESS</span>
            <span className="amber-glow font-bold" style={{ fontSize: "18px", color: "#FFB000" }}>{progress.toFixed(1)}%</span>
          </div>
          <div style={{ height: "24px", background: "#0e0e0e", border: "1px solid #524533", position: "relative" }}>
            <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${progress}%`, background: "linear-gradient(to right, #FF8C00, #FFB000)", transition: "width 0.5s ease" }} />
            <div className="flex" style={{ position: "absolute", inset: 0, gap: "2px", padding: "2px" }}>
              {Array.from({ length: 30 }, (_, i) => (
                <div key={i} style={{ flex: 1, background: i < Math.floor(progress / 3.33) ? "rgba(255,176,0,0.8)" : "rgba(53,53,53,0.3)" }} />
              ))}
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", marginBottom: "24px", fontSize: "13px", color: "#ef4444" }}>
            PIPELINE_ERROR: {error}
          </div>
        )}

        {/* Pipeline Steps */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1px", marginBottom: "24px" }}>
          {ALL_STEPS.map((step) => {
            const status = getStepStatus(step);
            return (
              <div
                key={step}
                className="flex items-center justify-between"
                style={{
                  padding: "12px 16px",
                  background: status === "running" ? "rgba(255,176,0,0.05)" : "#131313",
                  borderLeft: `3px solid ${stepStatusColor(status)}`,
                }}
              >
                <div className="flex items-center" style={{ gap: "12px" }}>
                  <span className="material-symbols-outlined" style={{ fontSize: "14px", color: stepStatusColor(status) }}>
                    {status === "complete" ? "check_circle" : status === "running" ? "pending" : "radio_button_unchecked"}
                  </span>
                  <span style={{ fontSize: "14px", color: status === "pending" ? "#9f8e78" : "#e2e2e2", fontWeight: 700, textTransform: "uppercase" }}>
                    {PIPELINE_STEP_NAMES[step] || step}
                  </span>
                </div>
                <span className="font-bold" style={{ fontSize: "11px", color: stepStatusColor(status) }}>
                  {status.toUpperCase()}
                </span>
              </div>
            );
          })}
        </div>

        {/* Bottom info */}
        <div className="flex items-center justify-between" style={{ borderTop: "1px solid #524533", paddingTop: "12px", fontSize: "11px" }}>
          <div>
            <span style={{ color: "#9f8e78" }}>SESSION_ID: </span>
            <span style={{ color: "#FFB000" }}>{sessionId}</span>
          </div>
          <div>
            <span style={{ color: "#9f8e78" }}>JOB_ID: </span>
            <span style={{ color: "#FFB000" }}>{jobId}</span>
          </div>
        </div>
      </main>

      {/* Right Panel */}
      <aside className="shrink-0 overflow-y-auto no-scrollbar" style={{ width: "320px", borderLeft: "1px solid #000", background: "#0e0e0e", padding: "16px" }}>
        <h2 className="amber-glow" style={{ color: "#fd8b00", fontWeight: 700, fontSize: "14px", textTransform: "uppercase", marginBottom: "16px" }}>
          LIVE_ENGINE_LOG
        </h2>
        <div className="bevel-recessed" style={{ padding: "8px", marginBottom: "24px", maxHeight: "300px", overflowY: "auto", fontSize: "11px" }}>
          {logs.map((entry, i) => (
            <div key={i} style={{ lineHeight: "20px" }}>
              <span style={{ color: "#9f8e78" }}>{entry.time}</span>{" "}
              <span style={{
                color: entry.level === "WARN" ? "#FFB000"
                  : entry.level === "ERROR" ? "#ef4444"
                  : entry.level === "PROC" ? "#60a5fa"
                  : "#4ade80",
                fontWeight: 700
              }}>
                [{entry.level}]
              </span>{" "}
              <span style={{ color: "#e2e2e2" }}>{entry.msg}</span>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <div className="flex justify-between" style={{ fontSize: "11px", marginBottom: "4px" }}>
              <span style={{ color: "#9f8e78" }}>PIPELINE_PROGRESS</span>
              <span className="font-bold" style={{ color: "#FFB000" }}>{progress.toFixed(0)}%</span>
            </div>
            <div style={{ height: "8px", background: "#2a2a2a" }}>
              <div style={{ width: `${progress}%`, height: "100%", background: "#FFB000", transition: "width 0.5s ease" }} />
            </div>
          </div>

          <div>
            <div className="flex justify-between" style={{ fontSize: "11px", marginBottom: "4px" }}>
              <span style={{ color: "#9f8e78" }}>CURRENT_STEP</span>
              <span className="font-bold" style={{ color: "#FFB000" }}>{job?.step_index || 0}/{job?.total_steps || 8}</span>
            </div>
            <div style={{ height: "8px", background: "#2a2a2a" }}>
              <div style={{ width: `${((job?.step_index || 0) / (job?.total_steps || 8)) * 100}%`, height: "100%", background: "#4ade80", transition: "width 0.5s ease" }} />
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
