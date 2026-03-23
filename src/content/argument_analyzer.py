"""
Argument Structure Analysis — Phase 8 of Modality 3.

Uses Claude CLI (Max subscription) for deep rhetorical analysis.
No API key needed — routes through claude -p pipe mode.
Falls back gracefully if CLI is unavailable.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are an expert rhetoric and public speaking analyst.
Analyze the following speech segment and provide a structured assessment.

SPEECH CONTEXT:
- This is segment {segment_num} of {total_segments} in the speech.
- Regime type: {regime_type}
- Previous segment summary: {prev_summary}

SEGMENT TEXT:
{segment_text}

Respond in this EXACT JSON format (no markdown, no backticks):
{{"claims": ["list of central assertions made"], "evidence": ["list of supporting evidence provided"], "evidence_quality": "Strong/Moderate/Weak/None", "warrants": ["implicit or explicit logical connections"], "transitions_from_previous": "Smooth/Abrupt/Signposted/None", "transition_phrases_used": ["list of transition phrases"], "logical_fallacies": ["list of detected fallacies, or empty"], "rhetorical_devices": {{"ethos": "credibility appeals", "pathos": "emotional appeals", "logos": "logical appeals"}}, "effectiveness_score": 75, "coaching_suggestion": "one specific actionable suggestion"}}"""


class ArgumentAnalyzer:
    """Analyzes argument structure using Claude CLI.

    For each regime segment, extracts claims, evidence, warrants,
    transitions, fallacies, and rhetorical devices.
    """

    def __init__(self, use_llm: bool = True, model: str = "claude-sonnet-4-20250514"):
        self.use_llm = use_llm
        self.model = model

    def analyze_segment(
        self,
        segment_text: str,
        segment_num: int = 1,
        total_segments: int = 1,
        regime_type: str = "Main Argument",
        prev_summary: str = "",
    ) -> dict:
        """Analyze argument structure of a single segment."""
        if not self.use_llm:
            return self._unavailable_fallback()

        try:
            return self._analyze_with_llm(
                segment_text, segment_num, total_segments, regime_type, prev_summary
            )
        except Exception as e:
            logger.warning(f"Argument analysis failed: {e}")
            return self._unavailable_fallback()

    def analyze_full_speech(self, segments: list, regime_types: list) -> list:
        """Analyze all segments with rate limiting.

        Args:
            segments: List of segment dicts with 'text' key.
            regime_types: List of regime type strings, one per segment.

        Returns:
            List of analysis result dicts.
        """
        results = []
        prev_summary = "Beginning of speech"

        for i, (seg, regime) in enumerate(zip(segments, regime_types)):
            text = seg["text"] if isinstance(seg, dict) else seg

            result = self.analyze_segment(
                segment_text=text,
                segment_num=i + 1,
                total_segments=len(segments),
                regime_type=regime,
                prev_summary=prev_summary,
            )

            result["start_sec"] = seg.get("start_sec", 0.0) if isinstance(seg, dict) else 0.0
            result["end_sec"] = seg.get("end_sec", 0.0) if isinstance(seg, dict) else 0.0
            results.append(result)

            # Build running context
            claims = result.get("claims", [])
            prev_summary = f"Segment {i+1} ({regime}): " + (
                claims[0] if claims else "No clear claim"
            )

            # Rate limit between API calls
            if i < len(segments) - 1:
                time.sleep(1.0)

        return results

    def _analyze_with_llm(
        self, segment_text, segment_num, total_segments, regime_type, prev_summary
    ) -> dict:
        """Use Claude CLI for argument analysis."""
        from src.utils.claude_api_wrapper import call_claude

        # Truncate very long segments
        words = segment_text.split()
        if len(words) > 300:
            segment_text = " ".join(words[:300]) + "..."

        prompt = ANALYSIS_PROMPT.format(
            segment_num=segment_num,
            total_segments=total_segments,
            regime_type=regime_type,
            prev_summary=prev_summary,
            segment_text=segment_text,
        )

        response = call_claude(prompt, model=self.model, timeout=60)

        # Parse JSON response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse argument analysis JSON, attempting repair")
            return self._repair_json(cleaned)

    def _repair_json(self, text: str) -> dict:
        """Attempt to extract structured data from malformed JSON."""
        result = self._unavailable_fallback()
        # Try to extract key fields with simple parsing
        for field in ["claims", "evidence", "evidence_quality", "coaching_suggestion"]:
            if f'"{field}"' in text:
                try:
                    start = text.index(f'"{field}"')
                    # Find the value after the colon
                    colon_pos = text.index(":", start)
                    # Simple extraction attempt
                    rest = text[colon_pos + 1:].strip()
                    if rest.startswith('"'):
                        end = rest.index('"', 1)
                        result[field] = rest[1:end]
                except (ValueError, IndexError):
                    pass
        return result

    def _unavailable_fallback(self) -> dict:
        """Return placeholder when analysis is not available."""
        return {
            "claims": [],
            "evidence": [],
            "evidence_quality": "Not analyzed",
            "warrants": [],
            "transitions_from_previous": "Not analyzed",
            "transition_phrases_used": [],
            "logical_fallacies": [],
            "rhetorical_devices": {
                "ethos": "Not analyzed",
                "pathos": "Not analyzed",
                "logos": "Not analyzed",
            },
            "effectiveness_score": None,
            "coaching_suggestion": "Enable Claude CLI for argument analysis.",
        }
