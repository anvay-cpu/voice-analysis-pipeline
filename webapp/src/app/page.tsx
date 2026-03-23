"use client";

import { useEffect, useState } from "react";
import BloombergChrome from "@/components/BloombergChrome";
import { SAMPLE_REPORT } from "@/lib/sample-data";
import { getSessions, getReport } from "@/lib/api";
import type { SpeechReport } from "@/lib/types";

export default function Home() {
  const [report, setReport] = useState<SpeechReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadLatest() {
      try {
        const sessions = await getSessions();
        const complete = sessions.find((s) => s.status === "COMPLETE");
        if (complete) {
          const r = await getReport(complete.id);
          setReport(r);
        }
      } catch {
        // API not available — fall through to null
      } finally {
        setLoading(false);
      }
    }
    loadLatest();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: "#000" }}>
        <div style={{ color: "#FFB000", fontSize: "13px" }}>INITIALIZING_SYSTEM...</div>
      </div>
    );
  }

  return <BloombergChrome report={report} />;
}
