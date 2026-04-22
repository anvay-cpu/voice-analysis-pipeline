"use client";

import { useRef, useState, useEffect } from "react";
import type { SpeechReport } from "@/lib/types";
import { scoreColor, formatDuration, regimeColor } from "@/lib/utils";
import { getStoredBackendUrl } from "@/lib/api";

export default function VideoView({ report }: { report: SpeechReport }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(report.duration_sec);
  const [videoBlobUrl, setVideoBlobUrl] = useState<string | null>(null);

  const { disruption_events, coaching_feedback, regime_segments, scores } = report;

  // Fetch video as blob to bypass ngrok interstitial
  useEffect(() => {
    if (!report.video_url) return;
    const backendUrl = getStoredBackendUrl() || "http://localhost:8000";
    const fullUrl = `${backendUrl}${report.video_url}`;
    const headers: Record<string, string> = {};
    if (backendUrl.includes("ngrok")) {
      headers["ngrok-skip-browser-warning"] = "true";
    }
    fetch(fullUrl, { headers })
      .then((res) => res.blob())
      .then((blob) => setVideoBlobUrl(URL.createObjectURL(blob)))
      .catch(() => setVideoBlobUrl(null));
    return () => {
      if (videoBlobUrl) URL.revokeObjectURL(videoBlobUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report.video_url]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTime = () => setCurrentTime(video.currentTime);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onLoaded = () => {
      if (video.duration && isFinite(video.duration)) setDuration(video.duration);
    };
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("loadedmetadata", onLoaded);
    return () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("loadedmetadata", onLoaded);
    };
  }, []);

  function seekTo(timeSec: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = timeSec;
      setCurrentTime(timeSec);
    }
  }

  function togglePlay() {
    if (!videoRef.current) return;
    if (videoRef.current.paused) videoRef.current.play();
    else videoRef.current.pause();
  }

  function rewind() {
    if (videoRef.current) {
      const t = Math.max(0, videoRef.current.currentTime - 10);
      videoRef.current.currentTime = t;
      setCurrentTime(t);
    }
  }

  function fastForward() {
    if (videoRef.current) {
      const t = Math.min(duration, videoRef.current.currentTime + 10);
      videoRef.current.currentTime = t;
      setCurrentTime(t);
    }
  }

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Waveform bars (decorative visualization)
  const waveformBars = Array.from({ length: 60 }, (_, i) => {
    const t = (i / 60) * duration;
    const nearby = disruption_events.find((ev) => Math.abs(ev.time_sec - t) < duration / 30);
    const base = nearby ? 0.6 + Math.random() * 0.4 : 0.15 + Math.random() * 0.35;
    return base;
  });

  // Coaching diagnostics
  const rating = Math.round(scores.overall);
  const confidence = Math.round(
    (scores.vocal_clarity + scores.content_structure) / 2
  );
  const stressLvl = Math.round(
    100 - (scores.emotional_expressiveness + scores.regime_adaptability) / 2
  );
  const eyeContact = Math.round(
    (scores.audience_engagement + scores.body_language) / 2
  );

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar" style={{ background: "#000", padding: "16px" }}>
      {/* Header */}
      <div className="flex items-center justify-between" style={{ marginBottom: "12px" }}>
        <div style={{ fontSize: "11px", color: "#9f8e78", textTransform: "uppercase", fontWeight: 700, letterSpacing: "-0.03em" }}>
          REPORT_F5_VIDEO_SCREEN
        </div>
        <div style={{ fontSize: "10px", color: "#524533", fontFamily: "monospace" }}>
          CAM_ID: 08-ALPHA-9 // [LIVE_SYNC: <span style={{ color: "#4ade80" }}>OK</span>]
        </div>
      </div>

      {/* Video frame — recessed */}
      <div
        style={{
          position: "relative",
          borderTop: "1px solid #000",
          borderBottom: "1px solid #524533",
          background: "#0e0e0e",
          marginBottom: "8px",
        }}
      >
        <div className="aspect-video" style={{ position: "relative", overflow: "hidden" }}>
          {videoBlobUrl ? (
            <video
              ref={videoRef}
              src={videoBlobUrl}
              style={{ width: "100%", height: "100%", objectFit: "contain", background: "#000" }}
            />
          ) : (
            <div
              className="flex flex-col items-center justify-center"
              style={{ width: "100%", height: "100%", color: "#353535", background: "#0e0e0e" }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "48px", marginBottom: "8px" }}>
                videocam_off
              </span>
              <span style={{ fontSize: "11px", textTransform: "uppercase" }}>NO_SIGNAL</span>
              <span style={{ fontSize: "10px", color: "#1b1b1b", marginTop: "4px" }}>
                Awaiting video feed...
              </span>
            </div>
          )}

          {/* Grid overlay — rule of thirds */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
            }}
          >
            {/* Vertical lines */}
            <div style={{ position: "absolute", left: "33.33%", top: 0, bottom: 0, width: "1px", background: "rgba(255,176,0,0.12)" }} />
            <div style={{ position: "absolute", left: "66.66%", top: 0, bottom: 0, width: "1px", background: "rgba(255,176,0,0.12)" }} />
            {/* Horizontal lines */}
            <div style={{ position: "absolute", top: "33.33%", left: 0, right: 0, height: "1px", background: "rgba(255,176,0,0.12)" }} />
            <div style={{ position: "absolute", top: "66.66%", left: 0, right: 0, height: "1px", background: "rgba(255,176,0,0.12)" }} />
          </div>

          {/* Voice waveform overlay bar at bottom */}
          <div
            style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              height: "32px",
              display: "flex",
              alignItems: "flex-end",
              padding: "0 4px 4px",
              background: "linear-gradient(to top, rgba(0,0,0,0.7), transparent)",
              gap: "1px",
            }}
          >
            {waveformBars.map((h, i) => {
              const pos = i / waveformBars.length;
              const isPlayed = pos * 100 < progress;
              return (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    height: `${h * 24}px`,
                    background: isPlayed
                      ? "rgba(255,176,0,0.6)"
                      : "rgba(255,176,0,0.15)",
                    borderRadius: "1px",
                  }}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* Transport controls — timeline scrubber */}
      <div style={{ marginBottom: "8px" }}>
        <div
          style={{
            position: "relative",
            height: "28px",
            cursor: "pointer",
            background: "#0e0e0e",
            borderTop: "1px solid #000",
            borderBottom: "1px solid #524533",
          }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            seekTo(((e.clientX - rect.left) / rect.width) * duration);
          }}
        >
          {/* Regime colored segments */}
          {regime_segments.map((seg, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: `${(seg.start_sec / duration) * 100}%`,
                width: `${(seg.duration_sec / duration) * 100}%`,
                top: 0,
                bottom: 0,
                opacity: 0.2,
                backgroundColor: regimeColor(seg.regime_type),
              }}
            />
          ))}

          {/* Progress fill */}
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: `${progress}%`,
              background: "rgba(255,176,0,0.15)",
            }}
          />

          {/* Event markers: star / x / warning */}
          {disruption_events.map((ev, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: `${(ev.time_sec / duration) * 100}%`,
                top: "3px",
                fontSize: "12px",
                color:
                  ev.status === "GOOD"
                    ? "#4ade80"
                    : ev.status === "SLOW"
                    ? "#ffb4ab"
                    : "#fd8b00",
                transform: "translateX(-50%)",
                lineHeight: "22px",
              }}
              title={`${ev.timestamp} ${ev.label}`}
            >
              {ev.status === "GOOD" ? "\u2605" : ev.status === "SLOW" ? "\u2717" : "\u26A0"}
            </div>
          ))}

          {/* Progress pointer */}
          <div
            style={{
              position: "absolute",
              left: `${progress}%`,
              top: "-2px",
              bottom: "-2px",
              width: "2px",
              background: "#FFB000",
              boxShadow: "0 0 6px rgba(255,176,0,0.5)",
            }}
          />
          {/* Pointer triangle */}
          <div
            style={{
              position: "absolute",
              left: `${progress}%`,
              bottom: "-6px",
              transform: "translateX(-4px)",
              width: 0,
              height: 0,
              borderLeft: "4px solid transparent",
              borderRight: "4px solid transparent",
              borderTop: "6px solid #FFB000",
            }}
          />
        </div>
      </div>

      {/* Tactile transport buttons */}
      <div className="flex items-center" style={{ gap: "8px", marginBottom: "16px" }}>
        <button
          onClick={rewind}
          className="cursor-pointer"
          style={{
            color: "#e2e2e2",
            fontSize: "12px",
            fontWeight: 700,
            fontFamily: "monospace",
            background: "linear-gradient(to bottom, #353535, #131313)",
            border: "none",
            borderTop: "1px solid #4a4a4a",
            borderBottom: "1px solid #000",
            borderLeft: "1px solid #2a2a2a",
            borderRight: "1px solid #0a0a0a",
            padding: "6px 14px",
            letterSpacing: "0.05em",
          }}
        >
          [◄◄] REW
        </button>
        <button
          onClick={togglePlay}
          className="cursor-pointer"
          style={{
            color: "#FFB000",
            fontSize: "12px",
            fontWeight: 700,
            fontFamily: "monospace",
            background: "linear-gradient(to bottom, #353535, #131313)",
            border: "none",
            borderTop: "1px solid #4a4a4a",
            borderBottom: "1px solid #000",
            borderLeft: "1px solid #2a2a2a",
            borderRight: "1px solid #0a0a0a",
            padding: "6px 18px",
            letterSpacing: "0.05em",
          }}
        >
          {isPlaying ? "[❚❚] PAUSE" : "[▶] PLAY_ST"}
        </button>
        <button
          onClick={fastForward}
          className="cursor-pointer"
          style={{
            color: "#e2e2e2",
            fontSize: "12px",
            fontWeight: 700,
            fontFamily: "monospace",
            background: "linear-gradient(to bottom, #353535, #131313)",
            border: "none",
            borderTop: "1px solid #4a4a4a",
            borderBottom: "1px solid #000",
            borderLeft: "1px solid #2a2a2a",
            borderRight: "1px solid #0a0a0a",
            padding: "6px 14px",
            letterSpacing: "0.05em",
          }}
        >
          [►►] FFW
        </button>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px" }}>
          <span className="font-bold" style={{ color: "#FFB000", fontSize: "16px", fontFamily: "monospace" }}>
            {formatDuration(currentTime)}
          </span>
          <span style={{ color: "#353535" }}>/</span>
          <span style={{ color: "#9f8e78", fontSize: "13px", fontFamily: "monospace" }}>
            {formatDuration(duration)}
          </span>
        </div>
      </div>

      {/* 2-column grid below */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        {/* Left column — EVENT_REGISTRY_ALPHA */}
        <div>
          <div
            style={{
              fontSize: "11px",
              color: "#fd8b00",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              marginBottom: "8px",
              borderBottom: "1px solid #524533",
              paddingBottom: "4px",
            }}
          >
            EVENT_REGISTRY_ALPHA
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
            {disruption_events.map((ev, i) => {
              const isCurrent = Math.abs(ev.time_sec - currentTime) < 3;
              return (
                <button
                  key={i}
                  onClick={() => seekTo(ev.time_sec)}
                  className="w-full text-left cursor-pointer"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "5px 8px",
                    fontSize: "12px",
                    fontFamily: "monospace",
                    border: "none",
                    color: isCurrent ? "#60a5fa" : "#e2e2e2",
                    background: isCurrent ? "rgba(30,58,138,0.3)" : "#0e0e0e",
                    borderLeft: isCurrent ? "2px solid #60a5fa" : "2px solid transparent",
                  }}
                >
                  <span
                    style={{
                      color:
                        ev.status === "GOOD"
                          ? "#4ade80"
                          : ev.status === "SLOW"
                          ? "#ffb4ab"
                          : "#fd8b00",
                      fontSize: "11px",
                    }}
                  >
                    {ev.status === "GOOD" ? "\u2605" : ev.status === "SLOW" ? "\u2717" : "\u26A0"}
                  </span>
                  <span style={{ color: "#9f8e78", flexShrink: 0 }}>{ev.timestamp}</span>
                  <span style={{ color: isCurrent ? "#60a5fa" : "#FFB000", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {ev.label}
                  </span>
                  {ev.composure !== null && (
                    <span style={{ marginLeft: "auto", color: scoreColor(ev.composure), fontSize: "11px", flexShrink: 0 }}>
                      {ev.composure}%
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Right column — COACHING_DIAGNOSTICS */}
        <div>
          <div
            style={{
              fontSize: "11px",
              color: "#fd8b00",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              marginBottom: "8px",
              borderBottom: "1px solid #524533",
              paddingBottom: "4px",
            }}
          >
            COACHING_DIAGNOSTICS
          </div>

          <div
            style={{
              background: "#0e0e0e",
              borderTop: "1px solid #000",
              borderBottom: "1px solid #524533",
              padding: "12px",
            }}
          >
            {[
              { label: "RATING", value: rating, suffix: "/100" },
              { label: "CONFIDENCE", value: confidence, suffix: "%" },
              { label: "STRESS_LVL", value: stressLvl, suffix: "%" },
              { label: "EYE_CONTACT", value: eyeContact, suffix: "%" },
            ].map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between"
                style={{
                  height: "32px",
                  borderBottom: "1px solid #1b1b1b",
                  fontSize: "12px",
                  fontFamily: "monospace",
                }}
              >
                <span style={{ color: "#9f8e78", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  {item.label}
                </span>
                <span
                  className="font-bold"
                  style={{
                    color: scoreColor(item.label === "STRESS_LVL" ? 100 - item.value : item.value),
                    fontSize: "14px",
                  }}
                >
                  {item.value}{item.suffix}
                </span>
              </div>
            ))}
          </div>

          {/* Current coaching feedback */}
          {coaching_feedback.length > 0 && (
            <div style={{ marginTop: "12px" }}>
              <div style={{ fontSize: "10px", color: "#524533", textTransform: "uppercase", fontWeight: 700, marginBottom: "6px" }}>
                LATEST_COACHING_FEED
              </div>
              {coaching_feedback
                .filter((cb) => Math.abs(cb.time_sec - currentTime) < 15)
                .slice(0, 2)
                .map((cb, i) => (
                  <div
                    key={i}
                    style={{
                      borderLeft: "2px solid #FFB000",
                      paddingLeft: "8px",
                      marginBottom: "6px",
                    }}
                  >
                    <div style={{ color: "#fd8b00", fontSize: "11px", fontWeight: 700, fontFamily: "monospace" }}>
                      {cb.timestamp} {cb.label}
                    </div>
                    <div style={{ color: "#9f8e78", fontSize: "12px", lineHeight: "1.5" }}>
                      {cb.feedback.slice(0, 150)}
                      {cb.feedback.length > 150 ? "..." : ""}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
