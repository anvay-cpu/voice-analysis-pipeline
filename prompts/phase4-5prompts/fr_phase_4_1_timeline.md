PHASE 4.1 — TIMELINE ALIGNMENT
=================================

AGENT ROLE: Temporal Systems Engineer
DEPENDS ON: All 3 modality pipelines (voice, body, content)
DELIVERS TO: Phases 4.2, 4.3, 4.4, 5.1
RUNS ON: Local M1 (pure computation, no GPU needed)
ESTIMATED TIME: 25 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/fusion/timeline_aligner.py that takes the three separate
JSON outputs from Modalities 1, 2, and 3, and aligns them on a
shared 1-second resolution timeline.

═══════════════════════════════════════════════════════════════
THE PROBLEM
═══════════════════════════════════════════════════════════════

Each modality outputs at a DIFFERENT temporal resolution:
- Voice (M1): 2-second windows, 1-second hop → data every 1s
- Body (M2): 6-second windows, 3-second hop → data every 3s
- Content (M3): per-regime segment (variable, 15-120 seconds each)

These must be merged onto a UNIFIED 1-second timeline so that
at any given second, you can ask: "What was the speaker's voice
doing? What was their body doing? What was the content about?"

═══════════════════════════════════════════════════════════════
TASK 1: Build TimelineAligner class
═══════════════════════════════════════════════════════════════

```python
class TimelineAligner:
    """Aligns all three modality outputs onto a 1-second grid.

    Input:
        voice_output: list of dicts, each covering a 2s window
        body_output: list of dicts, each covering a 6s window
        content_output: list of dicts, each covering a regime segment

    Output:
        timeline: list of dicts, one per second, containing:
        {
            "time_sec": 42,
            "voice": { speaking_rate, f0_cv, filler_words, emotion, ... },
            "body": { posture_score, gesture, gaze, facial_emotion, ... },
            "content": { regime_type, sentiment, tone, readability, ... },
        }
    """
```

ALIGNMENT STRATEGY:

For voice (1s hop, fine resolution):
  - Direct mapping: each 1-second tick maps to the voice window
    that contains it. Most seconds have exactly one window.

For body (3s hop, coarser resolution):
  - Each 1-second tick inherits values from the body window
    that covers it. If a tick falls in an overlap zone between
    two windows, take the weighted average based on proximity
    to each window center.

For content (variable segment length):
  - Each 1-second tick inherits the regime segment it falls within.
  - Content features (sentiment, tone, readability) are constant
    across the entire segment — they don't vary per-second.

═══════════════════════════════════════════════════════════════
TASK 2: Build interpolation for body data
═══════════════════════════════════════════════════════════════

Body data at 3-second resolution needs upsampling to 1-second:

For CONTINUOUS values (posture_score, gaze ratios, velocity):
  - Linear interpolation between window centers

For CATEGORICAL values (gesture class, hand state, movement type):
  - Hold the value constant within the window (no interpolation)

For DISTRIBUTIONS (gesture_distribution, gaze_distribution):
  - Linear interpolation of each probability

```python
def interpolate_body_to_1sec(body_windows, total_seconds):
    """Upsample body data from 3s hop to 1s resolution."""
    timeline = [None] * total_seconds

    for sec in range(total_seconds):
        # Find the body window(s) that cover this second
        covering = [w for w in body_windows
                    if w["timestamp_start"] <= sec < w["timestamp_end"]]

        if len(covering) == 0:
            # Gap — use nearest window
            nearest = min(body_windows,
                          key=lambda w: abs(w["timestamp_start"] - sec))
            timeline[sec] = nearest
        elif len(covering) == 1:
            timeline[sec] = covering[0]
        else:
            # Overlap — weighted average by proximity to center
            timeline[sec] = weighted_merge(covering, sec)

    return timeline
```

═══════════════════════════════════════════════════════════════
TASK 3: Build the merged timeline
═══════════════════════════════════════════════════════════════

```python
def align(self, voice_output, body_output, content_output,
          duration_sec) -> list[dict]:
    """Create 1-second aligned timeline from all 3 modalities."""

    voice_1s = self._map_voice_to_1sec(voice_output, duration_sec)
    body_1s = self._interpolate_body_to_1sec(body_output, duration_sec)
    content_1s = self._map_content_to_1sec(content_output, duration_sec)

    timeline = []
    for sec in range(int(duration_sec)):
        timeline.append({
            "time_sec": sec,
            "voice": voice_1s[sec] if sec < len(voice_1s) else None,
            "body": body_1s[sec] if sec < len(body_1s) else None,
            "content": content_1s[sec] if sec < len(content_1s) else None,
        })

    return timeline
```

═══════════════════════════════════════════════════════════════
TASK 4: Build regime boundary index
═══════════════════════════════════════════════════════════════

Create a helper that identifies the exact seconds where regime
transitions happen. These feed into Phase 4.2.

```python
def find_regime_boundaries(self, timeline) -> list[dict]:
    """Find seconds where the regime type changes."""
    boundaries = []
    for i in range(1, len(timeline)):
        prev_regime = timeline[i-1]["content"].get("regime_type")
        curr_regime = timeline[i]["content"].get("regime_type")
        if prev_regime != curr_regime:
            boundaries.append({
                "time_sec": i,
                "from_regime": prev_regime,
                "to_regime": curr_regime,
            })
    return boundaries
```

═══════════════════════════════════════════════════════════════
TASK 5: Tests
═══════════════════════════════════════════════════════════════

Write tests/fusion/test_timeline.py:
- Create mock voice/body/content outputs at different resolutions
- Verify alignment produces correct number of 1-second entries
- Verify body interpolation is smooth (no jumps at window boundaries)
- Verify regime boundaries are detected at correct timestamps
- Test edge case: missing data from one modality → None, no crash

NO USER INPUT REQUIRED. Pure computation.

COMPLETION: ✓ when timeline produces correct aligned output for mock data.
