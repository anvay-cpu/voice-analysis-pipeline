PHASE 5.3 — CHART GENERATION
===============================

AGENT ROLE: Data Visualization Specialist
DEPENDS ON: Phase 5.1 (scores), Phase 4.5 (fusion timeline)
DELIVERS TO: Phase 5.4 (report builder)
RUNS ON: Local M1
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/report/chart_generator.py that creates the visual elements
for the coaching report using matplotlib.

CHARTS TO GENERATE:

1. RADAR CHART — 6 dimension scores
   Hexagonal radar with scores 0-100 on each axis.
   Fill area in brand color with alpha=0.3.
   Show score values at each vertex.
   Save as: report_assets/radar_chart.png

2. TIMELINE HEATMAP — multi-channel quality over time
   Horizontal bar chart with rows for:
   - Speaking rate (green=optimal, amber=fast/slow, red=extreme)
   - Pitch variety (green=expressive, amber=moderate, red=monotone)
   - Eye contact (green=audience, amber=mixed, red=notes/floor)
   - Fillers (green=none, amber=occasional, red=burst)
   - Posture (green=good, amber=moderate, red=poor)
   - Emotion coherence (green=aligned, amber=mixed, red=mismatch)
   X-axis = time in minutes:seconds
   Save as: report_assets/timeline_heatmap.png

3. EMOTION ARC — triple-line chart over time
   Three lines: voice valence, face valence, content valence
   Smoothed with rolling average (window=10 seconds)
   Regime boundaries shown as vertical dashed lines
   Save as: report_assets/emotion_arc.png

4. REGIME FLOW — horizontal bar showing speech structure
   Each regime segment as a colored block
   Width proportional to duration
   Color by regime type
   Transition scores shown between blocks
   Save as: report_assets/regime_flow.png

5. FILLER WORD MAP — scatter plot of filler locations
   X = time, Y = filler type
   Size = severity (1 filler = small dot, burst = large dot)
   Save as: report_assets/filler_map.png

IMPLEMENTATION:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

class ChartGenerator:
    def __init__(self, output_dir="report_assets", dpi=150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        # Set consistent style
        plt.style.use('seaborn-v0_8-whitegrid')

    def generate_all(self, scores, timeline, fusion_output) -> dict:
        """Generate all charts and return paths."""
        paths = {}
        paths["radar"] = self.radar_chart(scores)
        paths["heatmap"] = self.timeline_heatmap(timeline)
        paths["emotion_arc"] = self.emotion_arc(timeline, fusion_output)
        paths["regime_flow"] = self.regime_flow(fusion_output)
        paths["filler_map"] = self.filler_map(timeline)
        return paths
```

INSTALL matplotlib if not present:
```bash
pip install matplotlib
```

NO USER INPUT REQUIRED. NO GPU NEEDED.

COMPLETION: ✓ when all 5 charts generate correctly as PNG files.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════