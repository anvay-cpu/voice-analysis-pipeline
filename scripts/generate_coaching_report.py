#!/usr/bin/env python3
"""
Standalone Coaching Report Generator (LaTeX PDF)

Loads pre-computed voice + body outputs from Colab, re-runs content analysis
locally with Claude CLI (LLM-powered), then generates a professional LaTeX
PDF coaching report.

Usage:
    conda activate voice-pipeline
    python scripts/generate_coaching_report.py

Prerequisites:
    - Voice + body output JSONs in data/outputs/ (downloaded from Colab)
    - Claude Code CLI installed and authenticated (Max subscription)
    - LaTeX installed: brew install --cask mactex-no-gui
      OR: brew install basictex && sudo tlmgr install collection-fontsrecommended
"""

import json
import logging
import os
import subprocess
import sys
import time
import base64
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# HARDCODED INPUTS — Update these paths after downloading from Colab
# ═══════════════════════════════════════════════════════════════════

VIDEOS = {
    "test_good": {
        "label": "Good Speaker (TED Talk)",
        "voice_json": "data/outputs/test_good_speaker_5min.json",
        "body_json": "data/outputs/test_good_speaker_5min_body.json",
        "video_path": "data/raw/test_good_speaker_5min.mp4",
    },
    "test_nervous": {
        "label": "Nervous Speaker (Student Presentation)",
        "voice_json": "data/outputs/test_nervous_speaker_5min.json",
        "body_json": "data/outputs/test_nervous_speaker_5min_body.json",
        "video_path": "data/raw/test_nervous_speaker_5min.mp4",
    },
    "test_monotone": {
        "label": "Monotone Speaker (Dry Lecture)",
        "voice_json": "data/outputs/test_monotone_speaker_5min.json",
        "body_json": "data/outputs/test_monotone_speaker_5min_body.json",
        "video_path": "data/raw/test_monotone_speaker_5min.mp4",
    },
}

OUTPUT_DIR = "reports/latex"


def load_json(path: str) -> dict:
    """Load a JSON file, with clear error if missing."""
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found!")
        print(f"  → Download it from Colab: /content/Claude-assistant/{path}")
        return {}
    with open(path) as f:
        return json.load(f)


def run_content_pipeline(voice_output: dict) -> dict:
    """Run content analysis with LLM features enabled."""
    from src.content.pipeline import ContentAnalysisPipeline

    # Reconstruct transcript from windows if no top-level "transcript" key
    if "transcript" in voice_output:
        transcript = voice_output["transcript"]
    else:
        # Build word-level transcript from windows
        words = []
        for w in voice_output.get("windows", []):
            text = w.get("transcript", "").strip()
            if not text:
                continue
            start = w.get("start_sec", 0)
            end = w.get("end_sec", 0)
            window_words = text.split()
            n = len(window_words)
            if n == 0:
                continue
            dur = end - start
            for i, word in enumerate(window_words):
                words.append({
                    "word": word,
                    "start": start + (i / n) * dur,
                    "end": start + ((i + 1) / n) * dur,
                })
        full_text = " ".join(wd["word"] for wd in words)
        transcript = {"text": full_text, "words": words}
        print("  Reconstructed transcript: %d words from %d windows" % (
            len(words), len(voice_output.get("windows", []))))

    pipeline = ContentAnalysisPipeline()
    result = pipeline.process(transcript)
    pipeline.close()
    return result


def run_fusion(voice_output: dict, body_output: dict, content_output: dict, duration: float) -> dict:
    """Run multimodal fusion."""
    from src.fusion.fusion_engine import FusionEngine

    config_path = "configs/fusion_config.yaml"
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}

    engine = FusionEngine(config)
    voice_windows = voice_output.get("windows", [])
    return engine.fuse(voice_windows, body_output, content_output, duration)


def run_scoring(fusion_output: dict) -> dict:
    """Compute 6-dimension scores."""
    from src.scoring.dimension_scorer import DimensionScorer
    return DimensionScorer().score_all(fusion_output)


def run_coaching(scores: dict, fusion_output: dict, duration: float, content_summary: dict) -> dict:
    """Generate LLM-powered coaching feedback by sending all data to Claude."""
    from src.utils.claude_api_wrapper import call_claude

    # Gather key moments from fusion
    recovery = fusion_output.get("recovery", {})
    coherence = fusion_output.get("emotion_coherence", {})
    boundaries = fusion_output.get("regime_boundaries", [])

    moments = []
    # Disruptions
    for r in recovery.get("recoveries", []):
        d = r["disruption"]
        t = d["time_sec"]
        moments.append({
            "time": "%d:%02d" % (t // 60, t % 60),
            "type": "disruption",
            "label": d["type"],
            "composure": "%.0f%%" % (r["composure_score"] * 100),
            "recovery_sec": r.get("vocal_recovery_sec", 0),
        })
    # Emotion mismatches
    for m in coherence.get("mismatch_events", []):
        t = m["time_sec"]
        moments.append({
            "time": "%d:%02d" % (t // 60, t % 60),
            "type": "emotion_mismatch",
            "description": m.get("description", ""),
            "coherence": "%.0f%%" % (m.get("mean_coherence", 0) * 100),
        })
    # Regime transitions
    for b in boundaries:
        t = b["time_sec"]
        moments.append({
            "time": "%d:%02d" % (t // 60, t % 60),
            "type": "transition",
            "from": b["from_regime"],
            "to": b["to_regime"],
        })

    moments.sort(key=lambda m: m["time"])
    moments = moments[:12]

    # Build the prompt
    scores_text = "\n".join("  %s: %d/100" % (k.replace("_", " ").title(), v)
                            for k, v in scores.items())
    moments_text = json.dumps(moments, indent=2, default=str)
    content_text = json.dumps({
        "grammar_score": content_summary.get("grammar_score_overall", "N/A"),
        "readability_fkgl": content_summary.get("readability_level", "N/A"),
        "vocabulary_diversity": content_summary.get("vocabulary_diversity", "N/A"),
        "dominant_tone": content_summary.get("dominant_tone", "N/A"),
        "dominant_sentiment": content_summary.get("dominant_sentiment", "N/A"),
        "argument_effectiveness": content_summary.get("argument_effectiveness_avg", "N/A"),
        "regime_sequence": content_summary.get("regime_sequence", []),
    }, indent=2)

    system_prompt = """You are an expert public speaking coach with 20 years of experience coaching
TEDx speakers, executives, and students. You give feedback that is:
- WARM: encouraging and supportive, never harsh or clinical
- SPECIFIC: reference exact timestamps and what happened there
- ACTIONABLE: provide concrete exercises and techniques to practice
- INSIGHTFUL: connect patterns across voice, body, and content
- HUMAN: write like you're sitting across from the speaker over coffee

Write in flowing paragraphs, not bullet points. Use "you" throughout.
Do NOT list raw scores numerically — weave insights into natural language."""

    # 1. Executive Summary
    print("    Generating executive summary...")
    exec_prompt = """Write a 6-8 sentence executive summary for a speech coaching session.

Scores:
%s

Speech duration: %.0f seconds (%.1f minutes)
Disruptions detected: %d
Emotion coherence: %s/100

Content analysis:
%s

Write as if you're opening a coaching session. Acknowledge what they did well first,
then gently introduce areas for growth. Make it feel personal and encouraging.
Do NOT use bullet points. Do NOT list scores as numbers — describe them naturally.""" % (
        scores_text, duration, duration / 60,
        recovery.get("total_disruptions", 0),
        coherence.get("coherence_score_0_100", 0),
        content_text,
    )

    try:
        exec_summary = call_claude(exec_prompt, system=system_prompt, max_tokens=800, timeout=30)
    except Exception as e:
        logger.warning("LLM exec summary failed: %s", e)
        exec_summary = "Analysis complete. See detailed feedback below."

    # 2. Per-moment feedback
    print("    Generating per-moment coaching feedback (%d moments)..." % len(moments))
    feedback_list = []
    for moment in moments:
        t = moment["time"]
        mtype = moment.get("type", "")
        label = moment.get("label", moment.get("description", moment.get("to", "")))

        try:
            prompt = """Write 4-6 sentences of personalized coaching feedback for this moment in a speech.

Moment at %s:
%s

Overall speaker scores:
%s

Explain what happened at this moment in human terms (not technical jargon).
Then give ONE specific, concrete technique they can practice to improve.
Write warmly and encouragingly — like a mentor, not a critic.
Address the speaker as "you".""" % (t, json.dumps(moment, default=str), scores_text)

            feedback_text = call_claude(prompt, system=system_prompt, max_tokens=400, timeout=45)
        except Exception as e:
            logger.warning("LLM moment feedback failed at %s: %s", t, e)
            feedback_text = "At %s, there was a %s moment. Focus on maintaining your rhythm through these transitions." % (t, mtype)

        feedback_list.append({
            "timestamp": t,
            "type": mtype,
            "label": label,
            "feedback": feedback_text,
        })

    # 3. Practice plan
    print("    Generating practice plan...")
    weakest = sorted(
        [(k, v) for k, v in scores.items() if k != "overall"],
        key=lambda x: x[1]
    )[:3]

    practice_prompt = """Create 3 specific, creative practice exercises for a speaker to improve.

Their 3 weakest dimensions:
%s

All scores:
%s

For each exercise provide:
- A catchy title (5-8 words)
- A detailed description (3-4 sentences) explaining exactly how to do the exercise
- How often to practice (e.g., "3x per week, 10 minutes")

Make the exercises fun and engaging, not boring drills. Reference specific issues
from their scores. Write as a coach who genuinely wants them to succeed.

Return as JSON array: [{"title": "...", "description": "...", "frequency": "..."}]""" % (
        json.dumps([{"dimension": d.replace("_", " ").title(), "score": s} for d, s in weakest], indent=2),
        scores_text,
    )

    try:
        practice_response = call_claude(practice_prompt, system=system_prompt, max_tokens=800, timeout=30)
        # Parse JSON from response
        start_idx = practice_response.index("[")
        end_idx = practice_response.rindex("]") + 1
        practice_plan = json.loads(practice_response[start_idx:end_idx])
    except Exception as e:
        logger.warning("LLM practice plan failed: %s", e)
        practice_plan = [
            {"title": "Mirror Practice for Awareness", "description": "Practice your speech in front of a mirror. Focus on one dimension per session.", "frequency": "3x per week, 15 minutes"},
            {"title": "Record and Review", "description": "Record yourself giving a 2-minute speech. Watch it back and note one strength and one area to improve.", "frequency": "2x per week, 10 minutes"},
            {"title": "Pause Power Drill", "description": "Read a passage aloud. At every period, pause for 3 full seconds. This builds comfort with silence.", "frequency": "Daily, 5 minutes"},
        ]

    return {
        "executive_summary": exec_summary,
        "segment_feedback": feedback_list,
        "practice_plan": practice_plan,
    }


def run_charts(scores: dict, fusion_output: dict, chart_dir: str) -> dict:
    """Generate all charts."""
    from src.report.chart_generator import ChartGenerator

    os.makedirs(chart_dir, exist_ok=True)
    gen = ChartGenerator(output_dir=chart_dir)
    return gen.generate_all(scores, fusion_output["timeline"], fusion_output)


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters and convert markdown to LaTeX."""
    import re as _re

    # Convert markdown bold **text** → \textbf{text} BEFORE escaping
    text = _re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)

    # Convert em-dash
    text = text.replace("—", "---")
    text = text.replace("–", "--")

    # Single-pass replacement to avoid double-escaping
    special = {
        "\\": "\\textbackslash ",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde ",
        "^": "\\textasciicircum ",
        "<": "\\textless ",
        ">": "\\textgreater ",
    }

    result = []
    for ch in text:
        if ch in special:
            result.append(special[ch])
        else:
            result.append(ch)
    return "".join(result)


def score_color(score: float) -> str:
    """Return LaTeX color name based on score."""
    if score >= 75:
        return "ScoreGreen"
    elif score >= 55:
        return "ScoreAmber"
    else:
        return "ScoreRed"


def build_latex_report(
    label: str,
    scores: dict,
    coaching: dict,
    charts: dict,
    content_summary: dict,
    fusion_output: dict,
    duration: float,
    processing_time: float,
) -> str:
    """Build the complete LaTeX document using template substitution."""

    dims = {k: v for k, v in scores.items() if k != "overall"}
    overall = scores.get("overall", 0)
    duration_min = duration / 60

    # Executive summary
    exec_summary = escape_latex(coaching.get("executive_summary", "Analysis complete."))

    # Score cards
    score_rows = []
    for dim, val in dims.items():
        dim_label = dim.replace("_", " ").title()
        color = score_color(val)
        score_rows.append(
            "    %s & \\textcolor{%s}{\\textbf{%.0f/100}} \\\\" % (escape_latex(dim_label), color, val)
        )
    score_table = "\n".join(score_rows)

    # Overall score row
    overall_color = score_color(overall)
    overall_row = "\\textbf{Overall} & \\textcolor{%s}{\\textbf{%.0f/100}} \\\\" % (overall_color, overall)

    # Segment feedback
    feedback_sections = []
    for item in coaching.get("segment_feedback", []):
        timestamp = item.get("timestamp", "")
        ftype = item.get("type", "").replace("_", " ").title()
        flabel = escape_latex(item.get("label", ""))
        ftext = escape_latex(item.get("feedback", ""))
        feedback_sections.append(
            "\\noindent\\colorbox{FeedbackBg}{\\parbox{\\dimexpr\\textwidth-2\\fboxsep\\relax}{%%\n"
            "\\textcolor{BrandBlue}{\\textbf{%s}} \\hfill \\textit{%s}\\\\[4pt]\n"
            "\\textbf{%s}\\\\[4pt]\n"
            "%s\n"
            "}}\n"
            "\\vspace{8pt}" % (timestamp, ftype, flabel, ftext)
        )
    feedback_tex = "\n".join(feedback_sections) if feedback_sections else "\\textit{No specific coaching moments identified.}"

    # Practice plan
    practice_items = []
    for i, ex in enumerate(coaching.get("practice_plan", []), 1):
        title = escape_latex(ex.get("title", "Exercise %d" % i))
        desc = escape_latex(ex.get("description", ""))
        freq = escape_latex(ex.get("frequency", ""))
        practice_items.append(
            "\\noindent\\colorbox{PracticeBg}{\\parbox{\\dimexpr\\textwidth-2\\fboxsep\\relax}{%%\n"
            "\\textbf{%d. %s}\\\\[4pt]\n"
            "%s\\\\[4pt]\n"
            "\\textit{%s}\n"
            "}}\n"
            "\\vspace{8pt}" % (i, title, desc, freq)
        )
    practice_tex = "\n".join(practice_items) if practice_items else "\\textit{Keep practicing regularly!}"

    # Content analysis summary
    content_stats = ""
    if content_summary:
        grammar = content_summary.get("grammar_score_overall", "N/A")
        readability = content_summary.get("readability_level", "N/A")
        vocab = content_summary.get("vocabulary_diversity", "N/A")
        tone = escape_latex(str(content_summary.get("dominant_tone", "N/A")))
        sentiment = escape_latex(str(content_summary.get("dominant_sentiment", "N/A")))
        arg_score = content_summary.get("argument_effectiveness_avg", "N/A")

        content_stats = (
            "\\subsection*{Content Analysis Details}\n"
            "\\begin{tabular}{ll}\n"
            "Grammar Score & %s/100 \\\\\n"
            "Readability (FKGL) & %s \\\\\n"
            "Vocabulary Diversity (TTR) & %s \\\\\n"
            "Dominant Tone & %s \\\\\n"
            "Dominant Sentiment & %s \\\\\n"
            "Argument Effectiveness & %s/100 \\\\\n"
            "\\end{tabular}\n"
            "\\vspace{12pt}" % (grammar, readability, vocab, tone, sentiment, arg_score)
        )

    # Fusion stats
    recovery = fusion_output.get("recovery", {})
    coherence = fusion_output.get("emotion_coherence", {})
    disruptions = recovery.get("total_disruptions", 0)
    composure = recovery.get("mean_composure", 0)
    coherence_score = coherence.get("coherence_score_0_100", 0)
    boundaries = len(fusion_output.get("regime_boundaries", []))

    # Chart includes
    def chart_include(name, width="0.95"):
        path = charts.get(name, "")
        if path and os.path.exists(path):
            abs_path = os.path.abspath(path)
            return "\\includegraphics[width=%s\\textwidth]{%s}" % (width, abs_path)
        return "\\textit{Chart not available.}"

    label_escaped = escape_latex(label)
    composure_pct = "%.0f\\%%" % (composure * 100)

    # Build template (plain string, no f-string)
    tex = r"""\documentclass[11pt, a4paper]{article}

%% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{hyperref}
\usepackage{float}

%% Colors
\definecolor{BrandBlue}{HTML}{4A90D9}
\definecolor{BrandDark}{HTML}{2C3E50}
\definecolor{BrandLight}{HTML}{A8C8F0}
\definecolor{ScoreGreen}{HTML}{2ECC71}
\definecolor{ScoreAmber}{HTML}{F39C12}
\definecolor{ScoreRed}{HTML}{E74C3C}
\definecolor{FeedbackBg}{HTML}{F8F9FA}
\definecolor{PracticeBg}{HTML}{EAF4FD}

%% Page Style
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{BrandBlue}{AI Public Speaking Assistant}}
\fancyhead[R]{\small\textcolor{gray}{%%LABEL%%}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

%% Section Formatting
\titleformat{\section}{\Large\bfseries\color{BrandDark}}{}{0em}{}[\vspace{-4pt}\color{BrandLight}\rule{\textwidth}{1pt}]
\titleformat{\subsection}{\large\bfseries\color{BrandBlue}}{}{0em}{}

\hypersetup{
    colorlinks=true,
    linkcolor=BrandBlue,
    urlcolor=BrandBlue,
}

\begin{document}

\begin{titlepage}
\centering
\vspace*{3cm}

{\Huge\bfseries\textcolor{BrandDark}{Speech Coaching Report}}

\vspace{1cm}

{\Large\textcolor{gray}{%%LABEL%%}}

\vspace{1.5cm}

\begin{tikzpicture}
    \draw[BrandBlue, line width=4pt] (0,0) circle (2cm);
    \node[font=\fontsize{40}{48}\selectfont\bfseries, text=BrandDark] at (0, 0.2) {%%OVERALL%%};
    \node[font=\large, text=gray] at (0, -0.6) {/100};
\end{tikzpicture}

\vspace{0.5cm}

{\large Overall Score}

\vspace{2cm}

\begin{tabular}{cc}
\textbf{Duration} & %%DURATION_MIN%% minutes \\
\textbf{Disruptions} & %%DISRUPTIONS%% detected \\
\textbf{Emotion Coherence} & %%COHERENCE%%/100 \\
\textbf{Processing Time} & %%PROCTIME%% seconds \\
\end{tabular}

\vfill

{\small\textcolor{gray}{Generated by AI Public Speaking Assistant}}\\
{\small\textcolor{gray}{Powered by Claude API (Max Subscription)}}

\end{titlepage}

\section{Executive Summary}

%%EXEC_SUMMARY%%

\section{Dimension Scores}

\begin{table}[H]
\centering
\renewcommand{\arraystretch}{1.4}
\begin{tabularx}{0.7\textwidth}{Xr}
\toprule
\textbf{Dimension} & \textbf{Score} \\
\midrule
%%SCORE_TABLE%%
\midrule
%%OVERALL_ROW%%
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[H]
\centering
%%CHART_RADAR%%
\caption{Score breakdown across 6 coaching dimensions.}
\end{figure}

%%CONTENT_STATS%%

\section{Multimodal Analysis}

\subsection*{Fusion Statistics}
\begin{tabular}{ll}
Timeline Duration & %%DURATION_SEC%% seconds (%%DURATION_MIN%% min) \\
Regime Boundaries & %%BOUNDARIES%% transitions \\
Disruptions Detected & %%DISRUPTIONS%% \\
Mean Composure & %%COMPOSURE%% \\
Emotion Coherence & %%COHERENCE%%/100 \\
\end{tabular}

\section{Speech Quality Timeline}

\begin{figure}[H]
\centering
%%CHART_HEATMAP%%
\caption{Multi-channel quality heatmap. Green = good, amber = acceptable, red = needs work.}
\end{figure}

\section{Emotion Arc}

\begin{figure}[H]
\centering
%%CHART_EMOTION%%
\caption{Emotional valence over time: voice (red), facial (blue), content sentiment (green).}
\end{figure}

\section{Speech Structure}

\begin{figure}[H]
\centering
%%CHART_REGIME%%
\caption{Speech regime flow showing structural transitions.}
\end{figure}

\section{Key Moments \& Coaching Feedback}

%%FEEDBACK%%

\section{Filler Words}

\begin{figure}[H]
\centering
%%CHART_FILLER%%
\caption{Location and frequency of filler words throughout the speech.}
\end{figure}

\section{Practice Plan}

Based on your scores, here are 3 targeted exercises:

%%PRACTICE%%

\vfill
\begin{center}
\small\textcolor{gray}{AI Public Speaking Assistant $\cdot$ Processing time: %%PROCTIME%%s $\cdot$ Claude API (Max Subscription)}
\end{center}

\end{document}
"""

    # Substitute all placeholders
    tex = tex.replace("%%LABEL%%", label_escaped)
    tex = tex.replace("%%OVERALL%%", "%.0f" % overall)
    tex = tex.replace("%%DURATION_MIN%%", "%.1f" % duration_min)
    tex = tex.replace("%%DURATION_SEC%%", "%.0f" % duration)
    tex = tex.replace("%%DISRUPTIONS%%", str(disruptions))
    tex = tex.replace("%%COHERENCE%%", str(coherence_score))
    tex = tex.replace("%%PROCTIME%%", "%.0f" % processing_time)
    tex = tex.replace("%%COMPOSURE%%", composure_pct)
    tex = tex.replace("%%BOUNDARIES%%", str(boundaries))
    tex = tex.replace("%%EXEC_SUMMARY%%", exec_summary)
    tex = tex.replace("%%SCORE_TABLE%%", score_table)
    tex = tex.replace("%%OVERALL_ROW%%", overall_row)
    tex = tex.replace("%%CONTENT_STATS%%", content_stats)
    tex = tex.replace("%%FEEDBACK%%", feedback_tex)
    tex = tex.replace("%%PRACTICE%%", practice_tex)
    tex = tex.replace("%%CHART_RADAR%%", chart_include("radar", "0.55"))
    tex = tex.replace("%%CHART_HEATMAP%%", chart_include("heatmap"))
    tex = tex.replace("%%CHART_EMOTION%%", chart_include("emotion_arc"))
    tex = tex.replace("%%CHART_REGIME%%", chart_include("regime_flow"))
    tex = tex.replace("%%CHART_FILLER%%", chart_include("filler_map"))

    return tex


def compile_latex(tex_path: str, output_dir: str) -> str | None:
    """Compile LaTeX to PDF using pdflatex."""
    try:
        for i in range(2):  # Run twice for references
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", output_dir, tex_path],
                capture_output=True, text=True, timeout=60,
            )
        pdf_path = tex_path.replace(".tex", ".pdf")
        if os.path.exists(pdf_path):
            return pdf_path
        else:
            logger.error("pdflatex failed:\n%s", result.stdout[-500:])
            return None
    except FileNotFoundError:
        logger.error("pdflatex not found. Install with: brew install --cask mactex-no-gui")
        return None
    except subprocess.TimeoutExpired:
        logger.error("pdflatex timed out")
        return None


def process_one_video(key: str, config: dict) -> dict | None:
    """Process a single video: content analysis → fusion → scoring → coaching → report."""
    label = config["label"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Load pre-computed outputs
    print("\n[1/7] Loading Colab outputs...")
    voice_output = load_json(config["voice_json"])
    body_output = load_json(config["body_json"])

    if not voice_output or not body_output:
        print("  SKIPPED — missing input files")
        return None

    duration = voice_output.get("metadata", {}).get("duration_sec", 0)
    if not duration:
        duration = voice_output.get("summary", {}).get("duration_sec", 0)
    if not duration:
        duration = body_output.get("duration_sec", 300.0)

    print(f"  Voice: {len(voice_output.get('windows', []))} windows")
    print(f"  Body: {len(body_output.get('segments', []))} segments")
    print(f"  Duration: {duration:.0f}s")

    start = time.time()

    # Re-run content analysis with LLM
    print("\n[2/7] Running content analysis (with Claude LLM)...")
    content_output = run_content_pipeline(voice_output)
    content_segments = content_output.get("segments", [])
    content_summary = content_output.get("summary", {})
    print(f"  Content segments: {len(content_segments)}")
    print(f"  Grammar: {content_summary.get('grammar_score_overall', 'N/A')}")
    print(f"  Tone: {content_summary.get('dominant_tone', 'N/A')}")

    # Fusion
    print("\n[3/7] Multimodal fusion...")
    fusion_output = run_fusion(voice_output, body_output, content_output, duration)
    print(f"  Timeline: {len(fusion_output.get('timeline', []))} seconds")
    print(f"  Boundaries: {len(fusion_output.get('regime_boundaries', []))}")

    # Scoring
    print("\n[4/7] Computing scores...")
    scores = run_scoring(fusion_output)
    for dim, val in scores.items():
        print(f"  {dim}: {val:.0f}/100")

    # Coaching feedback (LLM-powered)
    print("\n[5/7] Generating coaching feedback (Claude LLM)...")
    coaching = run_coaching(scores, fusion_output, duration, content_summary)
    n_feedback = len(coaching.get("segment_feedback", []))
    print(f"  Executive summary: {len(coaching.get('executive_summary', ''))} chars")
    print(f"  Feedback moments: {n_feedback}")
    print(f"  Practice exercises: {len(coaching.get('practice_plan', []))}")

    # Charts
    report_dir = os.path.join(OUTPUT_DIR, key)
    chart_dir = os.path.join(report_dir, "charts")
    print("\n[6/7] Generating charts...")
    charts = run_charts(scores, fusion_output, chart_dir)
    print(f"  Generated {len(charts)} charts")

    # LaTeX report
    elapsed = time.time() - start
    print("\n[7/7] Building LaTeX report...")
    tex_content = build_latex_report(
        label=label,
        scores=scores,
        coaching=coaching,
        charts=charts,
        content_summary=content_summary,
        fusion_output=fusion_output,
        duration=duration,
        processing_time=elapsed,
    )

    os.makedirs(report_dir, exist_ok=True)
    tex_path = os.path.join(report_dir, "speech_report.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"  LaTeX source: {tex_path}")

    pdf_path = compile_latex(tex_path, report_dir)
    if pdf_path:
        print(f"  PDF report: {pdf_path}")
    else:
        print("  PDF compilation failed — .tex file saved for manual compilation")

    total_time = time.time() - start
    print(f"\n  COMPLETE in {total_time:.0f}s")

    return {
        "label": label,
        "scores": scores,
        "tex_path": tex_path,
        "pdf_path": pdf_path,
        "processing_time": total_time,
    }


def main():
    print("=" * 60)
    print("  AI Public Speaking Assistant")
    print("  Coaching Report Generator (LaTeX PDF)")
    print("  LLM-Powered via Claude Max Subscription")
    print("=" * 60)

    # Check prerequisites
    print("\nChecking prerequisites...")

    # Claude CLI
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "claude-sonnet-4-20250514"],
            input="Reply with OK",
            capture_output=True, text=True, timeout=15,
        )
        if "OK" in result.stdout:
            print("  Claude CLI: OK")
        else:
            print("  Claude CLI: WARNING — may not be working")
    except Exception as e:
        print(f"  Claude CLI: NOT AVAILABLE ({e})")
        print("  → LLM features will fall back to templates")

    # pdflatex
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, timeout=5)
        print("  pdflatex: OK")
    except FileNotFoundError:
        print("  pdflatex: NOT FOUND")
        print("  → Install: brew install --cask mactex-no-gui")
        print("  → .tex files will be saved for manual compilation")

    # Check input files
    print("\nChecking input files...")
    missing = []
    for key, config in VIDEOS.items():
        voice_ok = os.path.exists(config["voice_json"])
        body_ok = os.path.exists(config["body_json"])
        status_v = "OK" if voice_ok else "MISSING"
        status_b = "OK" if body_ok else "MISSING"
        print(f"  {key}: voice={status_v}, body={status_b}")
        if not voice_ok:
            missing.append(config["voice_json"])
        if not body_ok:
            missing.append(config["body_json"])

    if missing:
        print(f"\n  Missing {len(missing)} file(s). Download from Colab:")
        for f in missing:
            print(f"    /content/Claude-assistant/{f}")
        print("\n  Copy to Drive → download, or run in Colab:")
        print("    !cp /content/Claude-assistant/data/outputs/*.json \\")
        print("       /content/drive/MyDrive/Claude-assistant/data/outputs/")

    # Process available videos
    results = []
    for key, config in VIDEOS.items():
        if os.path.exists(config["voice_json"]) and os.path.exists(config["body_json"]):
            result = process_one_video(key, config)
            if result:
                results.append(result)

    if not results:
        print("\nNo videos processed. Download the output JSONs from Colab first.")
        return

    # Summary
    print(f"\n{'='*60}")
    print("  GENERATION COMPLETE")
    print(f"{'='*60}")
    for r in results:
        pdf_status = r["pdf_path"] if r["pdf_path"] else "LaTeX only"
        print(f"  {r['label']}: {r['scores']['overall']:.0f}/100 → {pdf_status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
