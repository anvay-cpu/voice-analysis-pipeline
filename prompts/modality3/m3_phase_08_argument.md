PHASE 8 — ARGUMENT STRUCTURE ANALYSIS (CLAUDE API)
=====================================================

AGENT ROLE: Rhetoric Analyst
DEPENDS ON: Phase 2 (transcript), Phase 5 (regime segments)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 20 minutes (agent code), 0-5 minutes (user — API key)

OBJECTIVE:
Build src/content/argument_analyzer.py that uses the Claude API to
analyze the logical structure, evidence quality, and rhetorical
effectiveness of each speech segment.

═══════════════════════════════════════════════════════════════
WHY CLAUDE API INSTEAD OF A LOCAL MODEL
═══════════════════════════════════════════════════════════════

Argument analysis requires deep semantic understanding:
- Identifying implicit claims vs explicit claims
- Evaluating whether evidence supports a claim
- Detecting logical fallacies (straw man, false dichotomy, etc.)
- Understanding rhetorical strategies (ethos, pathos, logos)

This is a task where even large fine-tuned models struggle.
Claude Sonnet handles it well with zero-shot prompting.
Cost: ~$0.01-0.02 per 10-minute speech. Negligible.

If no API key is available, this phase is SKIPPED gracefully.
The pipeline still works — it just won't have argument analysis.

═══════════════════════════════════════════════════════════════
TASK 1: Build ArgumentAnalyzer class
═══════════════════════════════════════════════════════════════

```python
class ArgumentAnalyzer:
    """Analyzes argument structure using Claude API.

    For each regime segment, extracts:
    - Claims: Central assertions
    - Evidence: Data, examples, anecdotes supporting claims
    - Warrants: Logical connections between evidence and claims
    - Transitions: How the speaker moves between ideas
    - Fallacies: Detected logical errors
    - Rhetorical devices: ethos/pathos/logos usage
    """

    PROMPT_TEMPLATE = """You are an expert rhetoric and public speaking analyst.
Analyze the following speech segment and provide a structured assessment.

SPEECH CONTEXT:
- This is segment {segment_num} of {total_segments} in the speech.
- Regime type: {regime_type}
- Previous segment summary: {prev_summary}

SEGMENT TEXT:
{segment_text}

Respond in this EXACT JSON format (no markdown, no backticks):
{{
    "claims": ["list of central assertions made"],
    "evidence": ["list of supporting evidence provided"],
    "evidence_quality": "Strong/Moderate/Weak/None",
    "warrants": ["implicit or explicit logical connections"],
    "transitions_from_previous": "Smooth/Abrupt/Signposted/None",
    "transition_phrases_used": ["list of transition phrases"],
    "logical_fallacies": ["list of detected fallacies, or empty"],
    "rhetorical_devices": {{
        "ethos": "description of credibility appeals",
        "pathos": "description of emotional appeals",
        "logos": "description of logical appeals"
    }},
    "effectiveness_score": 75,
    "coaching_suggestion": "one specific actionable suggestion"
}}"""

    def __init__(self, api_key=None, model="claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.available = self.api_key is not None

    def analyze_segment(self, segment_text, segment_num, total_segments,
                        regime_type, prev_summary="") -> dict:
        if not self.available:
            return self._unavailable_fallback()

        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)

        prompt = self.PROMPT_TEMPLATE.format(
            segment_num=segment_num,
            total_segments=total_segments,
            regime_type=regime_type,
            prev_summary=prev_summary,
            segment_text=segment_text
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response
        try:
            result = json.loads(response.content[0].text)
            return result
        except json.JSONDecodeError:
            return self._parse_fallback(response.content[0].text)

    def _unavailable_fallback(self):
        """Return placeholder when API key is not available."""
        return {
            "claims": [],
            "evidence": [],
            "evidence_quality": "Not analyzed (no API key)",
            "effectiveness_score": None,
            "coaching_suggestion": "Enable Claude API for argument analysis."
        }
```

═══════════════════════════════════════════════════════════════
TASK 2: Batch analysis with rate limiting
═══════════════════════════════════════════════════════════════

```python
def analyze_full_speech(self, segments, regime_types):
    """Analyze all segments with rate limiting.

    For a 10-minute speech with ~8-12 segments,
    this makes 8-12 API calls costing ~$0.01-0.02 total.
    """
    results = []
    prev_summary = ""

    for i, (segment, regime) in enumerate(zip(segments, regime_types)):
        # Rate limit: 1 call per second (conservative)
        time.sleep(1.0)

        result = self.analyze_segment(
            segment_text=segment["text"],
            segment_num=i + 1,
            total_segments=len(segments),
            regime_type=regime,
            prev_summary=prev_summary
        )

        results.append(result)

        # Build running summary for context
        claims = result.get("claims", [])
        prev_summary = f"Segment {i+1} ({regime}): " + \
                       (claims[0] if claims else "No clear claim")

    return results
```

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 8: Argument Analysis

WHAT IS HAPPENING:
The Argument Analyzer uses the Claude API for deep rhetorical analysis.

WHY YOUR INPUT IS NEEDED:
An Anthropic API key is required. If you already set it in Phase 5,
this step is automatic. If not:

WHAT YOU NEED TO DO:
Step 1: Go to https://console.anthropic.com/settings/keys
Step 2: Create an API key (free tier includes $5 credit)
Step 3: In terminal:
  export ANTHROPIC_API_KEY="sk-ant-your-key-here"
  echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.zshrc

IF YOU DON'T WANT TO USE THE API:
That's fine. The pipeline works without argument analysis.
Set in config: argument.engine: "none"
Everything else still functions.

COMPLETION: ✓ when analyze_segment() returns structured JSON for test input
            OR when gracefully skipped with fallback message.