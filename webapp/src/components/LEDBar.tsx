"use client";

export default function LEDBar({
  value,
  max = 100,
  segments = 10,
  height = 12,
}: {
  value: number;
  max?: number;
  segments?: number;
  height?: number;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const filled = Math.round((pct / 100) * segments);

  return (
    <div className="flex" style={{ gap: "2px" }}>
      {Array.from({ length: segments }, (_, i) => (
        <div
          key={i}
          style={{
            flex: 1,
            height: `${height}px`,
            background: i < filled
              ? "linear-gradient(to bottom, #ffba43, #ffb000)"
              : "#353535",
            opacity: i < filled ? 1 : 0.2,
          }}
        />
      ))}
    </div>
  );
}
