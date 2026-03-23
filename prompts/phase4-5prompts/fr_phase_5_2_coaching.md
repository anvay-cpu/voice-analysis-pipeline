PHASE 5.2 — COACHING WRITER (CLAUDE API)
==========================================

AGENT ROLE: Coaching Language Specialist
DEPENDS ON: Phase 5.1 (scores), Phase 4.5 (fusion output)
DELIVERS TO: Phase 5.4 (report builder)
RUNS ON: Local M1 (API calls via proxy at localhost:8080)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/report/coaching_writer.py that uses Claude API to
transform raw data into warm, specific, actionable coaching feedback.

ALL API CALLS MUST USE:
  base_url = "http://localhost:8080"
  api_key = "not-needed"
  model = "claude-sonnet-4-20250514"
This routes through the user's Max subscription. $0 cost.

TASKS:

1. Build CoachingWriter class with:
   - transform_segment(segment_data) → coaching text for one segment
   - generate_executive_summary(all_scores, all_segments) → summary
   - generate_practice_plan(scores, weaknesses) → 3 action items
   - Each method makes 1 Claude API call

2. System prompt for the coaching persona:
   "You are an expert public speaking coach with 20 years of experience.
    You give feedback that is WARM, SPECIFIC (with timestamps),
    ACTIONABLE (concrete tips), and INSIGHTFUL (connecting observations
    across voice, body, and content)."

3. Per-segment prompt template:
   Pass the raw data (voice metrics, body metrics, content metrics,
   fusion observations) and ask Claude to write 3-5 sentences of
   coaching feedback + 1-2 sentences of actionable advice.

4. Total API calls per 10-min speech:
   - 8-12 segment feedbacks × 1 call each = 8-12 calls
   - 1 executive summary = 1 call
   - 1 practice plan = 1 call
   - Total: 10-14 calls, ~$0.00 (through proxy)

5. Fallback: if API is unavailable, use template-based feedback:
   "Your speaking rate was {rate} syl/sec ({assessment}).
    Your posture score was {score}/100. Try: {generic_tip}."

NOTE: The coaching_writer.py code was already provided in a previous
conversation. Use that implementation with ONE change:
  Replace base_url="http://localhost:8317" with base_url="http://localhost:8080"

NO USER INPUT REQUIRED.

COMPLETION: ✓ when coaching text is generated for all segments.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════