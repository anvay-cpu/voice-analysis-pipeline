"""Phase 5.3 — Chart Generation.

Generates 5 chart types for the coaching report:
1. Radar chart (6-dimension scores)
2. Timeline heatmap (multi-channel quality over time)
3. Emotion arc (triple-line valence chart)
4. Regime flow (horizontal segment bar)
5. Filler word map (scatter plot)
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Color palette
BRAND_COLOR = "#4A90D9"
BRAND_LIGHT = "#A8C8F0"
COLORS_GOOD = "#2ECC71"
COLORS_WARN = "#F39C12"
COLORS_BAD = "#E74C3C"

REGIME_COLORS = {
    "Introduction": "#3498DB",
    "Main Argument": "#2ECC71",
    "Anecdote": "#9B59B6",
    "Data Presentation": "#F39C12",
    "Call to Action": "#E74C3C",
    "Conclusion": "#1ABC9C",
    "Q&A": "#34495E",
    "Unknown": "#BDC3C7",
}


class ChartGenerator:
    """Generate all charts for the coaching report."""

    def __init__(self, output_dir: str = "report_assets", dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def generate_all(
        self,
        scores: dict,
        timeline: list[dict],
        fusion_output: dict,
    ) -> dict:
        """Generate all charts and return paths.

        Args:
            scores: 6-dimension scores dict.
            timeline: Aligned 1-second timeline.
            fusion_output: Full fusion output.

        Returns:
            Dict mapping chart name to file path.
        """
        paths = {}
        paths["radar"] = self.radar_chart(scores)
        paths["heatmap"] = self.timeline_heatmap(timeline)
        paths["emotion_arc"] = self.emotion_arc(timeline, fusion_output)
        paths["regime_flow"] = self.regime_flow(fusion_output)
        paths["filler_map"] = self.filler_map(timeline)

        logger.info("Generated %d charts in %s", len(paths), self.output_dir)
        return paths

    # ── 1. Radar Chart ─────────────────────────────────────────────

    def radar_chart(self, scores: dict) -> str:
        """Generate hexagonal radar chart of 6 dimension scores."""
        dims = {k: v for k, v in scores.items() if k != "overall"}
        if not dims:
            dims = {"vocal_clarity": 50, "body_language": 50, "content_structure": 50,
                    "audience_engagement": 50, "emotional_expressiveness": 50,
                    "regime_adaptability": 50}

        labels = [k.replace("_", " ").title() for k in dims.keys()]
        values = list(dims.values())

        # Close the polygon
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color=BRAND_COLOR, alpha=0.25)
        ax.plot(angles, values, color=BRAND_COLOR, linewidth=2)

        # Score labels at each vertex
        for angle, value, label in zip(angles[:-1], values[:-1], labels):
            ax.text(angle, value + 5, f"{value:.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=BRAND_COLOR)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="gray")
        ax.set_title(f"Overall: {scores.get('overall', 0):.0f}/100",
                     fontsize=14, fontweight="bold", pad=20)

        path = os.path.join(self.output_dir, "radar_chart.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ── 2. Timeline Heatmap ────────────────────────────────────────

    def timeline_heatmap(self, timeline: list[dict]) -> str:
        """Generate multi-channel quality heatmap over time."""
        n = len(timeline)
        if n == 0:
            return self._empty_chart("timeline_heatmap.png", "No timeline data")

        channels = {
            "Speaking Rate": [],
            "Pitch Variety": [],
            "Eye Contact": [],
            "Fillers": [],
            "Posture": [],
            "Coherence": [],
        }

        for entry in timeline:
            voice = entry.get("voice") or {}
            body = entry.get("body") or {}

            # Rate: green if 3.5-5.0, warn if outside, red if extreme
            rate = voice.get("syllable_rate", 4.0)
            channels["Speaking Rate"].append(
                1.0 if 3.5 <= rate <= 5.0 else 0.5 if 2.5 <= rate <= 6.0 else 0.0
            )

            # Pitch variety
            cv = voice.get("pitch_cv", 0.2)
            channels["Pitch Variety"].append(
                1.0 if 0.15 <= cv <= 0.30 else 0.5 if 0.10 <= cv <= 0.40 else 0.0
            )

            # Eye contact
            gaze = body.get("gaze_engagement", 0.5)
            channels["Eye Contact"].append(
                1.0 if gaze >= 0.7 else 0.5 if gaze >= 0.4 else 0.0
            )

            # Fillers (inverse: green = no fillers)
            fillers = voice.get("filler_count", 0)
            channels["Fillers"].append(
                1.0 if fillers == 0 else 0.5 if fillers == 1 else 0.0
            )

            # Posture
            posture = body.get("posture_score", 50.0)
            channels["Posture"].append(
                1.0 if posture >= 70 else 0.5 if posture >= 50 else 0.0
            )

            # Coherence (placeholder — needs full coherence timeline)
            channels["Coherence"].append(0.7)

        # Build heatmap matrix
        channel_names = list(channels.keys())
        data = np.array([channels[c] for c in channel_names])

        fig, ax = plt.subplots(figsize=(12, 3))

        # Custom colormap: red → amber → green
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "quality", [COLORS_BAD, COLORS_WARN, COLORS_GOOD]
        )

        ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

        # Time labels
        tick_interval = max(1, n // 10)
        xticks = list(range(0, n, tick_interval))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t // 60}:{t % 60:02d}" for t in xticks], fontsize=8)
        ax.set_xlabel("Time", fontsize=10)

        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names, fontsize=9)
        ax.set_title("Speech Quality Timeline", fontsize=12, fontweight="bold")

        path = os.path.join(self.output_dir, "timeline_heatmap.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ── 3. Emotion Arc ─────────────────────────────────────────────

    def emotion_arc(self, timeline: list[dict], fusion_output: dict) -> str:
        """Generate triple-line emotion valence chart."""
        from src.fusion.emotion_coherence import VOICE_VALENCE, FACE_VALENCE, SENTIMENT_VALENCE

        if not timeline:
            return self._empty_chart("emotion_arc.png", "No timeline data")

        voice_vals = []
        face_vals = []
        content_vals = []

        for entry in timeline:
            voice = entry.get("voice") or {}
            body = entry.get("body") or {}
            content = entry.get("content") or {}

            voice_vals.append(VOICE_VALENCE.get(voice.get("emotion_label", "Neutral"), 0.0))
            face_vals.append(FACE_VALENCE.get(body.get("facial_emotion", "Neutral"), 0.0))
            content_vals.append(SENTIMENT_VALENCE.get(content.get("sentiment_polarity", "Neutral"), 0.0))

        # Smooth with rolling average
        window = min(10, len(timeline) // 3 + 1)
        voice_smooth = self._rolling_mean(voice_vals, window)
        face_smooth = self._rolling_mean(face_vals, window)
        content_smooth = self._rolling_mean(content_vals, window)

        x = list(range(len(timeline)))
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(x, voice_smooth, label="Voice Emotion", color="#E74C3C", linewidth=1.5)
        ax.plot(x, face_smooth, label="Facial Emotion", color="#3498DB", linewidth=1.5)
        ax.plot(x, content_smooth, label="Content Sentiment", color="#2ECC71", linewidth=1.5)

        # Regime boundaries as vertical lines
        boundaries = fusion_output.get("regime_boundaries", [])
        for b in boundaries:
            ax.axvline(x=b["time_sec"], color="gray", linestyle="--", alpha=0.5)
            ax.text(b["time_sec"], 1.1, b["to_regime"], fontsize=7,
                    rotation=45, ha="left", va="bottom", color="gray")

        ax.set_xlim(0, len(timeline) - 1)
        ax.set_ylim(-1.2, 1.4)
        ax.set_ylabel("Valence", fontsize=10)
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_title("Emotion Arc", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.axhline(y=0, color="gray", linewidth=0.5)

        path = os.path.join(self.output_dir, "emotion_arc.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ── 4. Regime Flow ─────────────────────────────────────────────

    def regime_flow(self, fusion_output: dict) -> str:
        """Generate horizontal bar showing speech structure."""
        timeline = fusion_output.get("timeline", [])
        if not timeline:
            return self._empty_chart("regime_flow.png", "No timeline data")

        # Extract regime segments
        segments = []
        current_regime = None
        seg_start = 0

        for entry in timeline:
            content = entry.get("content") or {}
            regime = content.get("regime_type", "Unknown")
            if regime != current_regime:
                if current_regime is not None:
                    segments.append((seg_start, entry["time_sec"], current_regime))
                current_regime = regime
                seg_start = entry["time_sec"]
        if current_regime is not None:
            segments.append((seg_start, len(timeline), current_regime))

        fig, ax = plt.subplots(figsize=(10, 2))

        for start, end, regime in segments:
            color = REGIME_COLORS.get(regime, "#BDC3C7")
            ax.barh(0, end - start, left=start, height=0.6, color=color,
                    edgecolor="white", linewidth=1)
            # Label in center of segment
            mid = (start + end) / 2
            if end - start > 3:  # Only label if wide enough
                ax.text(mid, 0, regime, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")

        # Transition scores
        transitions = fusion_output.get("transitions", {}).get("transitions", [])
        for t in transitions:
            score = t["score"]
            color = COLORS_GOOD if score >= 0.7 else COLORS_WARN if score >= 0.4 else COLORS_BAD
            ax.plot(t["time_sec"], 0.5, "v", color=color, markersize=8)
            ax.text(t["time_sec"], 0.65, f"{score:.0%}", ha="center",
                    fontsize=7, color=color)

        ax.set_xlim(0, len(timeline))
        ax.set_ylim(-0.5, 1.0)
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_title("Speech Structure", fontsize=12, fontweight="bold")
        ax.set_yticks([])

        path = os.path.join(self.output_dir, "regime_flow.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ── 5. Filler Word Map ─────────────────────────────────────────

    def filler_map(self, timeline: list[dict]) -> str:
        """Generate scatter plot of filler word locations."""
        if not timeline:
            return self._empty_chart("filler_map.png", "No timeline data")

        times = []
        sizes = []

        for entry in timeline:
            voice = entry.get("voice") or {}
            count = voice.get("filler_count", 0)
            if count > 0:
                times.append(entry["time_sec"])
                sizes.append(count * 40)  # Scale dot size

        fig, ax = plt.subplots(figsize=(10, 2.5))

        if times:
            ax.scatter(times, [1] * len(times), s=sizes, c=COLORS_BAD,
                       alpha=0.6, edgecolors="white")
            ax.set_ylabel("Fillers", fontsize=10)
        else:
            ax.text(len(timeline) / 2, 1, "No filler words detected",
                    ha="center", va="center", fontsize=11, color=COLORS_GOOD)

        ax.set_xlim(0, max(len(timeline), 1))
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_title("Filler Word Locations", fontsize=12, fontweight="bold")
        ax.set_yticks([])

        path = os.path.join(self.output_dir, "filler_map.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ── Helpers ────────────────────────────────────────────────────

    def _empty_chart(self, filename: str, message: str) -> str:
        """Generate a placeholder chart with a message."""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, message, ha="center", va="center",
                fontsize=12, color="gray", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    @staticmethod
    def _rolling_mean(values: list[float], window: int) -> list[float]:
        """Compute rolling mean with edge handling."""
        if not values or window <= 1:
            return values
        result = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            result.append(sum(values[start:end]) / (end - start))
        return result
