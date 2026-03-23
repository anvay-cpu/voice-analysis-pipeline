"use client";

import { useState, useRef } from "react";
import { uploadVideo } from "@/lib/api";

interface Props {
  onUploadStarted: (jobId: string, sessionId: string) => void;
}

export default function UploadView({ onUploadStarted }: Props) {
  const [sessionName, setSessionName] = useState("");
  const [analysisProfile, setAnalysisProfile] = useState("EXECUTIVE_PRESENTATION_V2");
  const [metadata, setMetadata] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileSelect(files: FileList | null) {
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
      setError(null);
      if (!sessionName) {
        setSessionName(files[0].name.replace(/\.[^/.]+$/, ""));
      }
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setError("SELECT_A_FILE_FIRST");
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const result = await uploadVideo(
        selectedFile,
        sessionName || selectedFile.name,
        analysisProfile
      );
      onUploadStarted(result.job_id, result.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "UPLOAD_FAILED");
      setUploading(false);
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }

  return (
    <div className="flex flex-1 min-h-0">
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
        <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700, letterSpacing: "-0.03em", marginBottom: "8px" }}>
          SYSTEM_INPUT_MODULE &gt; UPLOAD_PIPELINE
        </div>

        <div style={{ maxWidth: "600px", margin: "0 auto", marginTop: "32px" }}>
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,audio/*,.mp4,.webm,.avi,.mov,.mkv,.wav,.mp3,.flac,.ogg"
            onChange={(e) => handleFileSelect(e.target.files)}
            style={{ display: "none" }}
          />

          {/* Drag & Drop Zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="bevel-recessed flex flex-col items-center justify-center cursor-pointer"
            style={{
              padding: "48px",
              marginBottom: "24px",
              border: isDragging ? "1px solid #FFB000" : selectedFile ? "1px solid #4ade80" : "1px solid #524533",
              background: isDragging ? "rgba(255,176,0,0.05)" : selectedFile ? "rgba(74,222,128,0.03)" : "#0e0e0e",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "48px", color: selectedFile ? "#4ade80" : "#FFB000", marginBottom: "16px" }}>
              {selectedFile ? "check_circle" : "cloud_upload"}
            </span>
            {selectedFile ? (
              <>
                <div className="font-bold" style={{ color: "#4ade80", fontSize: "14px", marginBottom: "4px" }}>
                  {selectedFile.name}
                </div>
                <div style={{ fontSize: "11px", color: "#9f8e78" }}>
                  {formatFileSize(selectedFile.size)} — CLICK_TO_CHANGE
                </div>
              </>
            ) : (
              <>
                <div className="amber-glow font-bold" style={{ color: "#FFB000", fontSize: "14px", marginBottom: "8px" }}>
                  DRAG_DROP_OR_SELECT_FILE
                </div>
                <div style={{ fontSize: "11px", color: "#9f8e78" }}>
                  SUPPORTED: .MP4, .WEBM, .AVI, .MOV, .WAV, .MP3, .FLAC, .OGG
                </div>
              </>
            )}
          </div>

          {/* Session Name */}
          <div style={{ marginBottom: "16px" }}>
            <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", display: "block", marginBottom: "4px" }}>SESSION_NAME</label>
            <input
              type="text"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder="FILE_RECORDING_001..."
              style={{ width: "100%", background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "14px", padding: "8px", fontWeight: 700 }}
            />
          </div>

          {/* Analysis Profile */}
          <div style={{ marginBottom: "16px" }}>
            <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", display: "block", marginBottom: "4px" }}>ANALYSIS_PROFILE</label>
            <select
              value={analysisProfile}
              onChange={(e) => setAnalysisProfile(e.target.value)}
              style={{ width: "100%", background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "14px", padding: "8px", fontWeight: 700 }}
            >
              <option>EXECUTIVE_PRESENTATION_V2</option>
              <option>TECHNICAL_BRIEFING</option>
              <option>CASUAL_CONVERSATION</option>
              <option>DEBATE_FORMAT</option>
            </select>
          </div>

          {/* Context Metadata */}
          <div style={{ marginBottom: "24px" }}>
            <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", display: "block", marginBottom: "4px" }}>CONTEXT_AND_METADATA</label>
            <textarea
              value={metadata}
              onChange={(e) => setMetadata(e.target.value)}
              placeholder="ENTER_SPEAKER_CONTEXT_FOR_AI_CALIBRATION..."
              style={{ width: "100%", height: "80px", background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "13px", padding: "8px", resize: "none" }}
            />
          </div>

          {/* Error */}
          {error && (
            <div style={{ padding: "8px 12px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", marginBottom: "16px", fontSize: "12px", color: "#ef4444" }}>
              ERROR: {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex" style={{ gap: "8px" }}>
            <button
              onClick={handleAnalyze}
              disabled={uploading || !selectedFile}
              className="flex-1 flex items-center justify-center cursor-pointer"
              style={{
                height: "44px",
                background: uploading || !selectedFile
                  ? "linear-gradient(to bottom, #555, #333)"
                  : "linear-gradient(to bottom, #FF8C00, #93000a)",
                color: "#fff",
                fontWeight: 700,
                fontSize: "14px",
                border: "none",
                borderTop: "1px solid rgba(255,180,171,0.2)",
                gap: "8px",
                opacity: uploading || !selectedFile ? 0.5 : 1,
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                {uploading ? "hourglass_empty" : "play_arrow"}
              </span>
              {uploading ? "UPLOADING..." : "ANALYZE_DATA"}
            </button>
            <button
              onClick={() => { setSelectedFile(null); setSessionName(""); setError(null); }}
              className="tactile-button cursor-pointer"
              style={{ padding: "0 32px", height: "44px", fontSize: "14px", color: "#e2e2e2", fontWeight: 700, border: "none" }}
            >
              CANCEL
            </button>
          </div>
        </div>
      </main>

      {/* Right Panel */}
      <aside className="shrink-0 overflow-y-auto no-scrollbar" style={{ width: "320px", borderLeft: "1px solid #000", background: "#0e0e0e", padding: "16px" }}>
        <h2 className="amber-glow" style={{ color: "#fd8b00", fontWeight: 700, fontSize: "14px", textTransform: "uppercase", marginBottom: "16px" }}>
          UPLOAD_STATISTICS
        </h2>
        <div className="bevel-recessed" style={{ padding: "8px 12px", marginBottom: "24px" }}>
          {[
            ["FILE_SIZE", selectedFile ? formatFileSize(selectedFile.size) : "—"],
            ["FILE_TYPE", selectedFile ? selectedFile.type || selectedFile.name.split(".").pop()?.toUpperCase() : "—"],
            ["PROFILE", analysisProfile.split("_").slice(0, 2).join("_")],
          ].map(([label, val]) => (
            <div key={label} className="flex justify-between items-center" style={{ height: "24px", fontSize: "13px" }}>
              <span style={{ color: "#9f8e78" }}>{label}</span>
              <span className="font-bold" style={{ color: "#FFB000" }}>{val}</span>
            </div>
          ))}
        </div>

        <h2 className="amber-glow" style={{ color: "#fd8b00", fontWeight: 700, fontSize: "14px", textTransform: "uppercase", marginBottom: "8px" }}>
          PIPELINE_INFO
        </h2>
        <div className="bevel-recessed" style={{ padding: "8px 12px", marginBottom: "24px" }}>
          {[
            ["MODALITIES", "3 (Voice + Body + Content)"],
            ["FUSION", "Multimodal Alignment"],
            ["SCORING", "6-Dimension (0-100)"],
          ].map(([label, val]) => (
            <div key={label} className="flex justify-between items-center" style={{ height: "24px", fontSize: "13px" }}>
              <span style={{ color: "#9f8e78" }}>{label}</span>
              <span className="font-bold" style={{ color: "#FFB000" }}>{val}</span>
            </div>
          ))}
        </div>

        <div style={{ padding: "12px", background: "#131313", border: "1px solid rgba(82,69,51,0.4)", marginTop: "24px" }}>
          <div style={{ fontSize: "11px", color: "#9f8e78", marginBottom: "8px", fontWeight: 700 }}>NOTICE</div>
          <p style={{ fontSize: "12px", color: "rgba(255,176,0,0.7)", lineHeight: "1.5" }}>
            Upload a speech video to receive AI-powered coaching across voice, body language, and content dimensions.
          </p>
        </div>
      </aside>
    </div>
  );
}
