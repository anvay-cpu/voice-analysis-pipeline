"use client";

import { useEffect, useState, use } from "react";
import BloombergChrome from "@/components/BloombergChrome";
import { getReport } from "@/lib/api";
import type { SpeechReport } from "@/lib/types";

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [report, setReport] = useState<SpeechReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getReport(id)
      .then(setReport)
      .catch((err) => {
        // Fallback to Next.js API route
        fetch(`/api/speeches/${id}/report`)
          .then((res) => {
            if (!res.ok) throw new Error(`Report '${id}' not found`);
            return res.json();
          })
          .then(setReport)
          .catch((e) => setError(e.message));
      });
  }, [id]);

  if (error) {
    return (
      <div
        className="flex items-center justify-center h-screen"
        style={{ background: "#000" }}
      >
        <div className="text-center">
          <div className="text-xl font-bold mb-2" style={{ color: "#ef4444" }}>
            ERROR
          </div>
          <div className="text-sm" style={{ color: "#9f8e78" }}>
            {error}
          </div>
          <a
            href="/"
            className="block mt-4 text-sm underline"
            style={{ color: "#FFB000" }}
          >
            Go to dashboard
          </a>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div
        className="flex items-center justify-center h-screen"
        style={{ background: "#000" }}
      >
        <div className="text-sm" style={{ color: "#FFB000" }}>
          LOADING REPORT...
        </div>
      </div>
    );
  }

  return <BloombergChrome report={report} />;
}
