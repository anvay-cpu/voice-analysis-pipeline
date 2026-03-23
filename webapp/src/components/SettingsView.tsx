"use client";

import { useState, useEffect } from "react";
import { setBackendUrl, getStoredBackendUrl, checkBackendHealth } from "@/lib/api";

export default function SettingsView() {
  const [colabUrl, setColabUrl] = useState("");
  const [backendStatus, setBackendStatus] = useState<"checking" | "connected" | "disconnected">("checking");
  const [backendInfo, setBackendInfo] = useState({ url: "", latency_ms: 0 });
  const [saved, setSaved] = useState(false);

  const [inputDevice, setInputDevice] = useState("MIC_ARRAY_SENN_HD4");
  const [noiseFloor, setNoiseFloor] = useState("-42.5_DB");
  const [sampleRate, setSampleRate] = useState("48000_HZ");
  const [bufferSize, setBufferSize] = useState("256_SAMPLES");
  const [sentimentEnabled, setSentimentEnabled] = useState(true);
  const [cadenceEnabled, setCadenceEnabled] = useState(true);
  const [debugEnabled, setDebugEnabled] = useState(false);

  useEffect(() => {
    setColabUrl(getStoredBackendUrl());
    checkConnection();
  }, []);

  async function checkConnection() {
    setBackendStatus("checking");
    const result = await checkBackendHealth();
    setBackendStatus(result.ok ? "connected" : "disconnected");
    setBackendInfo({ url: result.url, latency_ms: result.latency_ms });
  }

  function handleSaveColabUrl() {
    setBackendUrl(colabUrl);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    checkConnection();
  }

  function handleClearColabUrl() {
    setColabUrl("");
    setBackendUrl("");
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    checkConnection();
  }

  const statusColor = backendStatus === "connected" ? "#4ade80" : backendStatus === "disconnected" ? "#ef4444" : "#FFB000";
  const statusLabel = backendStatus === "connected" ? "CONNECTED" : backendStatus === "disconnected" ? "DISCONNECTED" : "CHECKING...";

  return (
    <div className="flex flex-1 min-h-0">
      <main className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "24px" }}>
        <div style={{ marginBottom: "32px", borderBottom: "1px solid rgba(82,69,51,0.3)", paddingBottom: "16px" }}>
          <h1 style={{ color: "#fd8b00", fontWeight: 700, fontSize: "24px", textTransform: "uppercase", letterSpacing: "-0.02em" }}>
            Terminal_Settings
          </h1>
          <p style={{ fontSize: "11px", color: "rgba(255,176,0,0.6)", marginTop: "4px" }}>
            CORE_CONFIGURATION_UTILITY // PID: 99281
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "48px" }}>
          {/* GPU Backend Configuration */}
          <div>
            <h2 className="amber-glow" style={{ fontSize: "14px", color: "#FFB000", fontWeight: 700, borderLeft: "2px solid #FFB000", paddingLeft: "8px", marginBottom: "16px", textTransform: "uppercase" }}>
              GPU_BACKEND_CONFIGURATION
            </h2>

            <div style={{ padding: "16px", background: "#0a0a0a", border: "1px solid rgba(82,69,51,0.3)", marginBottom: "16px" }}>
              <div style={{ fontSize: "11px", color: "#9f8e78", marginBottom: "12px" }}>
                Paste the ngrok URL from your Colab notebook to route processing to T4 GPU.
                Leave empty to use local backend (localhost:8000).
              </div>

              <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
                <input
                  type="text"
                  value={colabUrl}
                  onChange={(e) => setColabUrl(e.target.value)}
                  placeholder="https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app"
                  style={{
                    flex: 1,
                    background: "#0e0e0e",
                    border: "1px solid #524533",
                    boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)",
                    color: "#FFB000",
                    fontSize: "14px",
                    padding: "8px",
                    fontWeight: 700,
                  }}
                />
                <button
                  onClick={handleSaveColabUrl}
                  style={{
                    padding: "8px 20px",
                    background: "linear-gradient(to bottom, #FF8C00, #93000a)",
                    color: "#fff",
                    fontWeight: 700,
                    fontSize: "13px",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  {saved ? "SAVED ✓" : "CONNECT"}
                </button>
                {colabUrl && (
                  <button
                    onClick={handleClearColabUrl}
                    className="tactile-button"
                    style={{ padding: "8px 16px", fontSize: "13px", color: "#9f8e78", fontWeight: 700, border: "none", cursor: "pointer" }}
                  >
                    CLEAR
                  </button>
                )}
              </div>

              {/* Connection Status */}
              <div className="flex items-center justify-between" style={{ padding: "8px 12px", background: "#131313", border: "1px solid rgba(82,69,51,0.2)" }}>
                <div className="flex items-center" style={{ gap: "8px" }}>
                  <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: statusColor }} />
                  <span style={{ fontSize: "13px", color: statusColor, fontWeight: 700 }}>{statusLabel}</span>
                </div>
                <div style={{ fontSize: "11px", color: "#9f8e78" }}>
                  {backendInfo.url && (
                    <span>
                      {backendInfo.url.includes("ngrok") ? "COLAB_T4_GPU" : "LOCAL_CPU"} — {backendInfo.latency_ms}ms
                    </span>
                  )}
                </div>
                <button
                  onClick={checkConnection}
                  className="tactile-button"
                  style={{ padding: "4px 12px", fontSize: "11px", color: "#FFB000", fontWeight: 700, border: "none", cursor: "pointer" }}
                >
                  [TEST]
                </button>
              </div>
            </div>

            {/* Quick Setup Guide */}
            <div style={{ padding: "12px", background: "#131313", border: "1px solid rgba(82,69,51,0.4)", fontSize: "12px", color: "rgba(255,176,0,0.7)", lineHeight: "1.6" }}>
              <div style={{ fontWeight: 700, color: "#FFB000", marginBottom: "8px" }}>QUICK_SETUP:</div>
              1. Open <span style={{ color: "#4ade80" }}>notebooks/colab_gpu_server.ipynb</span> in Google Colab<br />
              2. Set runtime to <span style={{ color: "#4ade80" }}>T4 GPU</span><br />
              3. Run All cells (Ctrl+F9)<br />
              4. Copy the ngrok URL → paste above → click CONNECT<br />
              <div style={{ marginTop: "8px", fontSize: "11px", color: "#9f8e78" }}>
                Processing a 5-min video: ~60s on T4 GPU vs ~10+ min on local CPU
              </div>
            </div>
          </div>

          {/* Audio Hardware Parameters */}
          <div>
            <h2 className="amber-glow" style={{ fontSize: "14px", color: "#FFB000", fontWeight: 700, borderLeft: "2px solid #FFB000", paddingLeft: "8px", marginBottom: "16px", textTransform: "uppercase" }}>
              AUDIO_HARDWARE_PARAMETERS
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase" }}>Input_Device</label>
                  <select
                    value={inputDevice}
                    onChange={(e) => setInputDevice(e.target.value)}
                    style={{ background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "14px", padding: "8px", fontWeight: 700 }}
                  >
                    <option>MIC_ARRAY_SENN_HD4</option>
                    <option>SYSTEM_DEFAULT_AUDIO</option>
                    <option>VIRTUAL_LINE_IN_01</option>
                  </select>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase" }}>Sampling_Rate</label>
                  <input
                    type="text"
                    value={sampleRate}
                    onChange={(e) => setSampleRate(e.target.value)}
                    style={{ background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "14px", padding: "8px", fontWeight: 700 }}
                  />
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase" }}>Noise_Floor_Cutoff</label>
                  <input
                    type="text"
                    value={noiseFloor}
                    onChange={(e) => setNoiseFloor(e.target.value)}
                    style={{ background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "14px", padding: "8px", fontWeight: 700 }}
                  />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <label style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase" }}>Buffer_Size</label>
                  <input
                    type="text"
                    value={bufferSize}
                    onChange={(e) => setBufferSize(e.target.value)}
                    style={{ background: "#0e0e0e", border: "1px solid #524533", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.8)", color: "#FFB000", fontSize: "14px", padding: "8px", fontWeight: 700 }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* AI Inference Thresholds */}
          <div>
            <h2 className="amber-glow" style={{ fontSize: "14px", color: "#FFB000", fontWeight: 700, borderLeft: "2px solid #FFB000", paddingLeft: "8px", marginBottom: "16px", textTransform: "uppercase" }}>
              AI_INFERENCE_THRESHOLDS
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <ToggleRow label="ENABLE_SENTIMENT_ANALYSIS" checked={sentimentEnabled} onChange={setSentimentEnabled} status="[RUNNING]" statusColor="#9f8e78" />
              <ToggleRow label="CADENCE_DRIFT_DETECTION" checked={cadenceEnabled} onChange={setCadenceEnabled} status="[RUNNING]" statusColor="#9f8e78" />
              <ToggleRow label="DEBUG_VERBOSITY_LOGGING" checked={debugEnabled} onChange={setDebugEnabled} status="[DISABLED]" statusColor="#ffb4ab" />
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ paddingTop: "32px", borderTop: "1px solid rgba(82,69,51,0.3)", display: "flex", gap: "16px" }}>
            <button onClick={checkConnection} className="tactile-button amber-glow" style={{ padding: "8px 24px", fontSize: "14px", color: "#FFB000", fontWeight: 700, border: "none" }}>
              [F10] TEST_CONNECTION
            </button>
            <button className="tactile-button amber-glow" style={{ padding: "8px 24px", fontSize: "14px", color: "#FFB000", fontWeight: 700, border: "none" }}>
              [F11] EXPORT_CONFIG
            </button>
            <button className="tactile-button" style={{ padding: "8px 24px", fontSize: "14px", color: "#9c1209", fontWeight: 700, border: "none" }}>
              [DEL] PURGE_CACHE
            </button>
          </div>
        </div>
      </main>

      {/* Right Panel */}
      <aside className="shrink-0 overflow-y-auto no-scrollbar" style={{ width: "320px", borderLeft: "1px solid #000", background: "#0e0e0e", padding: "16px" }}>
        <div className="flex items-center justify-between" style={{ marginBottom: "24px", borderBottom: "1px solid rgba(82,69,51,0.2)", paddingBottom: "8px" }}>
          <span className="amber-glow" style={{ color: "#fd8b00", fontSize: "11px", fontWeight: 700 }}>BACKEND_STATUS</span>
          <span style={{ color: statusColor, fontSize: "11px" }}>● {statusLabel}</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          {/* Backend Info */}
          <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "13px" }}>
            {[
              ["BACKEND_TYPE", colabUrl ? "COLAB_T4_GPU" : "LOCAL_CPU"],
              ["ENDPOINT", backendInfo.url ? new URL(backendInfo.url).hostname : "—"],
              ["LATENCY", backendInfo.latency_ms ? `${backendInfo.latency_ms}ms` : "—"],
              ["STATUS", statusLabel],
            ].map(([label, val]) => (
              <div key={label} className="flex justify-between">
                <span style={{ color: "#9f8e78" }}>{label}</span>
                <span style={{ color: "#FFB000", fontWeight: 700 }}>{val}</span>
              </div>
            ))}
          </div>

          {/* GPU Capabilities */}
          <div>
            <div style={{ fontSize: "11px", color: "#9f8e78", marginBottom: "8px", fontWeight: 700 }}>COMPUTE_CAPABILITIES</div>
            <div className="bevel-recessed" style={{ padding: "8px 12px", fontSize: "12px" }}>
              {colabUrl ? (
                <>
                  <div style={{ color: "#4ade80", marginBottom: "4px" }}>T4 GPU — 15GB VRAM</div>
                  <div style={{ color: "#9f8e78" }}>CUDA 11.8 / cuDNN 8.6</div>
                  <div style={{ color: "#9f8e78" }}>~60s per 5-min video</div>
                </>
              ) : (
                <>
                  <div style={{ color: "#FFB000", marginBottom: "4px" }}>Apple M1 — 16GB Unified</div>
                  <div style={{ color: "#9f8e78" }}>MPS Backend</div>
                  <div style={{ color: "#ef4444" }}>⚠ May OOM on full pipeline</div>
                </>
              )}
            </div>
          </div>

          {/* Raw Feed */}
          <div style={{ marginTop: "auto", background: "#131313", padding: "12px", border: "1px solid rgba(82,69,51,0.4)" }}>
            <div style={{ fontSize: "10px", color: "#9f8e78", marginBottom: "8px" }}>SYSTEM_LOG</div>
            <div style={{ fontSize: "12px", color: "rgba(255,176,0,0.8)", lineHeight: "1.4", fontWeight: 700 }}>
              &gt; BACKEND: {colabUrl ? "REMOTE_GPU" : "LOCAL"}<br />
              &gt; HEALTH: {statusLabel}<br />
              &gt; SYNC_STATE: STABLE<br />
              &gt; HEARTBEAT_OK
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}

function ToggleRow({ label, checked, onChange, status, statusColor }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; status: string; statusColor: string;
}) {
  return (
    <div className="flex items-center justify-between" style={{ padding: "12px", background: "#1b1b1b", border: "1px solid rgba(82,69,51,0.2)" }}>
      <div className="flex items-center" style={{ gap: "16px" }}>
        <div
          onClick={() => onChange(!checked)}
          className="cursor-pointer flex items-center justify-center"
          style={{ width: "14px", height: "14px", border: "1px solid #FFB000", background: "#0e0e0e", fontSize: "10px", fontWeight: 700, color: "#FFB000" }}
        >
          {checked ? "X" : ""}
        </div>
        <span className="amber-glow" style={{ fontSize: "14px", textTransform: "uppercase" }}>{label}</span>
      </div>
      <span style={{ fontSize: "11px", color: statusColor }}>{status}</span>
    </div>
  );
}
