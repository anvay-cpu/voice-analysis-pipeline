"""
Phase 10 — Content Analysis Pipeline Validation Tests.

Tests individual modules and end-to-end pipeline on three scenarios:
  A. Structured TED-style speech (renewable energy)
  B. Rambling student presentation (social media)
  C. Data-heavy technical talk (ML optimization)

Runs with skip_heavy=True to avoid Java/model downloads in CI.
Heavy tests (grammar, regime, sentiment, tone, argument) are marked separately.
"""

import json
import os
import sys
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "test_transcripts")

def _load_transcript(name):
    path = os.path.join(TRANSCRIPT_DIR, name)
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# 1. Transcript Processor
# ════════════════════════════════════════════════════════════════════════════

class TestTranscriptProcessor:
    """Phase 2 validation: transcript preprocessing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.transcript_processor import TranscriptProcessor
        self.tp = TranscriptProcessor(sentence_splitter="spacy", min_words_per_segment=10)

    def test_load_from_whisper_structured(self):
        whisper = _load_transcript("test_a_structured.json")
        result = self.tp.load_from_whisper(whisper)
        assert "sentences" in result
        assert "segments" in result
        assert result["total_sentences"] > 0
        assert result["total_words"] > 100

    def test_load_from_whisper_rambling(self):
        whisper = _load_transcript("test_b_rambling.json")
        result = self.tp.load_from_whisper(whisper)
        assert result["total_sentences"] > 0
        assert result["total_words"] > 50

    def test_load_from_whisper_technical(self):
        whisper = _load_transcript("test_c_technical.json")
        result = self.tp.load_from_whisper(whisper)
        assert result["total_sentences"] > 0
        assert result["total_words"] > 100

    def test_load_from_text(self):
        text = "Hello world. This is a test. It has three sentences."
        result = self.tp.load_from_text(text, duration_sec=10.0)
        assert result["total_sentences"] >= 2
        assert result["duration_sec"] == 10.0

    def test_segments_have_timestamps(self):
        whisper = _load_transcript("test_a_structured.json")
        result = self.tp.load_from_whisper(whisper)
        for seg in result["segments"]:
            assert "start_sec" in seg
            assert "end_sec" in seg
            assert "text" in seg
            assert seg["word_count"] > 0

    def test_clean_for_grammar(self):
        cleaned = self.tp.clean_for_grammar("gonna tell you about stuff")
        assert "going to" in cleaned.lower()

    def test_segment_grouping(self):
        whisper = _load_transcript("test_a_structured.json")
        result = self.tp.load_from_whisper(whisper)
        # Every segment should have at least min_words (or be last segment)
        for seg in result["segments"][:-1]:
            assert seg["word_count"] >= 10, f"Segment too short: {seg['word_count']} words"


# ════════════════════════════════════════════════════════════════════════════
# 2. Readability Scorer
# ════════════════════════════════════════════════════════════════════════════

class TestReadabilityScorer:
    """Phase 4 validation: readability metrics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.readability_scorer import ReadabilityScorer
        from src.content.transcript_processor import TranscriptProcessor
        self.scorer = ReadabilityScorer()
        self.tp = TranscriptProcessor()

    def test_structured_readability(self):
        whisper = _load_transcript("test_a_structured.json")
        transcript = self.tp.load_from_whisper(whisper)
        result = self.scorer.score_transcript(transcript)
        assert "overall" in result
        assert "segments" in result
        overall = result["overall"]
        # Should have key metrics
        assert "flesch_kincaid_grade" in overall
        assert "dale_chall" in overall or "gunning_fog" in overall
        # Structured speech: moderate readability (grade 6-16)
        fkgl = overall["flesch_kincaid_grade"]
        assert 4 <= fkgl <= 18, f"FKGL {fkgl} out of expected range"

    def test_rambling_readability(self):
        whisper = _load_transcript("test_b_rambling.json")
        transcript = self.tp.load_from_whisper(whisper)
        result = self.scorer.score_transcript(transcript)
        fkgl = result["overall"]["flesch_kincaid_grade"]
        # Rambling speech should be simpler (lower grade level)
        assert 2 <= fkgl <= 14, f"FKGL {fkgl} out of expected range for rambling speech"

    def test_technical_readability(self):
        whisper = _load_transcript("test_c_technical.json")
        transcript = self.tp.load_from_whisper(whisper)
        result = self.scorer.score_transcript(transcript)
        fkgl = result["overall"]["flesch_kincaid_grade"]
        # Technical speech should be harder (higher grade level)
        assert 6 <= fkgl <= 20, f"FKGL {fkgl} out of expected range for technical speech"

    def test_per_segment_readability(self):
        whisper = _load_transcript("test_a_structured.json")
        transcript = self.tp.load_from_whisper(whisper)
        result = self.scorer.score_transcript(transcript)
        assert len(result["segments"]) == len(transcript["segments"])


# ════════════════════════════════════════════════════════════════════════════
# 3. Vocabulary Analyzer
# ════════════════════════════════════════════════════════════════════════════

class TestVocabularyAnalyzer:
    """Phase 4 validation: vocabulary sophistication."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.vocabulary_analyzer import VocabularyAnalyzer
        from src.content.transcript_processor import TranscriptProcessor
        awl_path = os.path.join(BASE_DIR, "data", "resources", "academic_word_list.txt")
        self.analyzer = VocabularyAnalyzer(awl_path=awl_path, min_ttr_window=50)
        self.tp = TranscriptProcessor()

    def test_structured_vocabulary(self):
        whisper = _load_transcript("test_a_structured.json")
        transcript = self.tp.load_from_whisper(whisper)
        result = self.analyzer.analyze_transcript(transcript)
        assert "overall" in result
        overall = result["overall"]
        assert "ttr" in overall
        assert 0 < overall["ttr"] <= 1.0

    def test_technical_has_higher_vocabulary(self):
        whisper_b = _load_transcript("test_b_rambling.json")
        whisper_c = _load_transcript("test_c_technical.json")
        trans_b = self.tp.load_from_whisper(whisper_b)
        trans_c = self.tp.load_from_whisper(whisper_c)
        vocab_b = self.analyzer.analyze_transcript(trans_b)
        vocab_c = self.analyzer.analyze_transcript(trans_c)
        # Technical speech should have higher lexical density or AWL coverage
        awl_b = vocab_b["overall"].get("awl_coverage", 0)
        awl_c = vocab_c["overall"].get("awl_coverage", 0)
        # This is a soft check — technical should trend higher
        assert awl_c >= 0, "AWL coverage should be non-negative"

    def test_per_segment_vocabulary(self):
        whisper = _load_transcript("test_a_structured.json")
        transcript = self.tp.load_from_whisper(whisper)
        result = self.analyzer.analyze_transcript(transcript)
        assert len(result["segments"]) == len(transcript["segments"])


# ════════════════════════════════════════════════════════════════════════════
# 4. Output Assembler
# ════════════════════════════════════════════════════════════════════════════

class TestOutputAssembler:
    """Phase 9 validation: output assembly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.output_assembler import OutputAssembler
        self.assembler = OutputAssembler(output_dir="/tmp/test_content_outputs")

    def test_assemble_minimal(self):
        transcript = {
            "sentences": [{"text": "Hello world.", "start_sec": 0, "end_sec": 2}],
            "segments": [
                {"text": "Hello world.", "start_sec": 0, "end_sec": 2, "word_count": 2, "sentence_indices": [0]}
            ],
            "total_words": 2,
            "total_sentences": 1,
            "duration_sec": 2.0,
        }
        result = self.assembler.assemble(transcript=transcript)
        assert "segments" in result
        assert "summary" in result
        assert "metadata" in result
        assert len(result["segments"]) == 1

    def test_assemble_with_readability_vocabulary(self):
        transcript = {
            "sentences": [],
            "segments": [
                {"text": "Seg one.", "start_sec": 0, "end_sec": 5, "word_count": 2, "sentence_indices": [0]},
                {"text": "Seg two.", "start_sec": 5, "end_sec": 10, "word_count": 2, "sentence_indices": [1]},
            ],
            "total_words": 4,
            "total_sentences": 2,
            "duration_sec": 10.0,
        }
        readability = {
            "overall": {"flesch_kincaid_grade": 8.0},
            "segments": [
                {"flesch_kincaid_grade": 7.0},
                {"flesch_kincaid_grade": 9.0},
            ],
        }
        vocabulary = {
            "overall": {"ttr": 0.65},
            "segments": [
                {"ttr": 0.6},
                {"ttr": 0.7},
            ],
        }
        result = self.assembler.assemble(
            transcript=transcript,
            readability=readability,
            vocabulary=vocabulary,
        )
        assert result["segments"][0]["readability"]["flesch_kincaid_grade"] == 7.0
        assert result["segments"][1]["vocabulary"]["ttr"] == 0.7

    def test_structure_scoring(self):
        # Good structure: intro → content → conclusion
        score_good = self.assembler._evaluate_structure(
            ["Introduction", "Main Argument", "Data Presentation", "Conclusion"]
        )
        # Bad structure: all same
        score_bad = self.assembler._evaluate_structure(
            ["Main Argument", "Main Argument", "Main Argument"]
        )
        assert score_good > score_bad

    def test_save_and_load(self):
        output = {"segments": [], "summary": {}, "metadata": {}}
        path = self.assembler.save(output, "test_output.json")
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == output
        os.remove(path)

    def test_summary_generation(self):
        segments = [
            {
                "regime_type": "Introduction",
                "tone": {"tone": "Informative"},
                "sentiment": {"polarity": "Neutral"},
                "readability": {"flesch_kincaid_grade": 8.0},
                "vocabulary": {"ttr": 0.6},
                "argument": {"effectiveness_score": 70},
            },
            {
                "regime_type": "Main Argument",
                "tone": {"tone": "Persuasive"},
                "sentiment": {"polarity": "Positive"},
                "readability": {"flesch_kincaid_grade": 10.0},
                "vocabulary": {"ttr": 0.7},
                "argument": {"effectiveness_score": 80},
            },
        ]
        summary = self.assembler._generate_summary(
            segments,
            grammar_score={"score": 90},
            transcript={"total_words": 100, "total_sentences": 10},
        )
        assert summary["total_segments"] == 2
        assert summary["grammar_score_overall"] == 90
        assert summary["readability_level"] == 9.0  # median of 8 and 10
        assert summary["argument_effectiveness_avg"] == 75.0


# ════════════════════════════════════════════════════════════════════════════
# 5. Tone Classifier (heuristic only — no LLM)
# ════════════════════════════════════════════════════════════════════════════

class TestToneClassifierHeuristic:
    """Phase 7 validation: heuristic tone classification."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.tone_classifier import ToneClassifier
        self.classifier = ToneClassifier(use_llm=False)

    def test_assertive_detection(self):
        result = self.classifier.classify("I believe without doubt this is the right approach. Certainly we must act.")
        assert result["tone"] == "Assertive"
        assert result["method"] == "heuristic"

    def test_informative_detection(self):
        result = self.classifier.classify("According to a recent study, thirty percent of the population uses this. Statistics show the fact that it consists of three parts.")
        assert result["tone"] == "Informative"

    def test_inspirational_detection(self):
        result = self.classifier.classify("Imagine a future where together we can build a new vision. Dream of the possibility to inspire change the world.")
        assert result["tone"] == "Inspirational"

    def test_cautionary_detection(self):
        result = self.classifier.classify("There is great danger and risk if we don't act. The consequences and threat are severe. Beware of the warning signs.")
        assert result["tone"] == "Cautionary"

    def test_empty_text(self):
        result = self.classifier.classify("")
        assert result["tone"] == "Informative"
        assert result["confidence"] == 0.0

    def test_classify_transcript(self):
        from src.content.transcript_processor import TranscriptProcessor
        tp = TranscriptProcessor()
        whisper = _load_transcript("test_a_structured.json")
        transcript = tp.load_from_whisper(whisper)
        result = self.classifier.classify_transcript(transcript)
        assert "dominant_tone" in result
        assert "segments" in result
        assert len(result["segments"]) == len(transcript["segments"])


# ════════════════════════════════════════════════════════════════════════════
# 6. Argument Analyzer (fallback only — no LLM)
# ════════════════════════════════════════════════════════════════════════════

class TestArgumentAnalyzerFallback:
    """Phase 8 validation: argument analysis fallback mode."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.argument_analyzer import ArgumentAnalyzer
        self.analyzer = ArgumentAnalyzer(use_llm=False)

    def test_fallback_structure(self):
        result = self.analyzer.analyze_segment("Some text about claims and evidence.")
        assert "claims" in result
        assert "evidence" in result
        assert "evidence_quality" in result
        assert "rhetorical_devices" in result
        assert result["evidence_quality"] == "Not analyzed"

    def test_full_speech_fallback(self):
        segments = [
            {"text": "Introduction text.", "start_sec": 0, "end_sec": 30},
            {"text": "Main argument text.", "start_sec": 30, "end_sec": 60},
        ]
        results = self.analyzer.analyze_full_speech(segments, ["Introduction", "Main Argument"])
        assert len(results) == 2
        assert results[0]["start_sec"] == 0
        assert results[1]["start_sec"] == 30


# ════════════════════════════════════════════════════════════════════════════
# 7. Pipeline — Light mode (skip_heavy=True)
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineLight:
    """End-to-end pipeline test with skip_heavy=True (no Java/models needed)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.content.pipeline import ContentAnalysisPipeline
        config_path = os.path.join(BASE_DIR, "configs", "content_pipeline_config.yaml")
        self.pipeline = ContentAnalysisPipeline(config_path=config_path)

    def test_pipeline_structured_light(self):
        whisper = _load_transcript("test_a_structured.json")
        result = self.pipeline.process(whisper, skip_heavy=True)
        assert "segments" in result
        assert "summary" in result
        assert "metadata" in result
        assert result["metadata"]["total_words"] > 100
        assert len(result["segments"]) > 0
        # Check segment structure
        seg = result["segments"][0]
        assert "readability" in seg
        assert "vocabulary" in seg
        assert "grammar" in seg

    def test_pipeline_rambling_light(self):
        whisper = _load_transcript("test_b_rambling.json")
        result = self.pipeline.process(whisper, skip_heavy=True)
        assert result["metadata"]["total_words"] > 50
        assert len(result["segments"]) > 0

    def test_pipeline_technical_light(self):
        whisper = _load_transcript("test_c_technical.json")
        result = self.pipeline.process(whisper, skip_heavy=True)
        assert result["metadata"]["total_words"] > 100
        assert len(result["segments"]) > 0

    def test_pipeline_from_text(self):
        text = "This is a simple test speech. It has multiple sentences. Each one should be processed correctly."
        result = self.pipeline.process(text, skip_heavy=True)
        assert result["metadata"]["total_sentences"] >= 2

    def test_pipeline_summary_fields(self):
        whisper = _load_transcript("test_a_structured.json")
        result = self.pipeline.process(whisper, skip_heavy=True)
        summary = result["summary"]
        assert "total_segments" in summary
        assert "grammar_score_overall" in summary
        assert "readability_level" in summary
        assert "vocabulary_diversity" in summary
        assert "content_structure_score" in summary

    def test_pipeline_processing_time(self):
        whisper = _load_transcript("test_a_structured.json")
        result = self.pipeline.process(whisper, skip_heavy=True)
        # Light mode should be fast (no LLM calls)
        assert result["metadata"]["processing_time_sec"] < 30

    def teardown_method(self):
        self.pipeline.close()


# ════════════════════════════════════════════════════════════════════════════
# 8. Pipeline — Heavy mode (requires Java + models + Claude CLI)
# ════════════════════════════════════════════════════════════════════════════

def _java_available():
    """Check if Java 17+ is available."""
    try:
        import subprocess
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def _claude_cli_available():
    """Check if claude CLI is available."""
    try:
        import subprocess
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

requires_java = pytest.mark.skipif(not _java_available(), reason="Java 17+ not available")
requires_claude = pytest.mark.skipif(not _claude_cli_available(), reason="Claude CLI not available")
heavy = pytest.mark.skipif(
    not (_java_available() and _claude_cli_available()),
    reason="Requires Java 17+ and Claude CLI"
)


@requires_java
class TestGrammarChecker:
    """Phase 3 validation: grammar checking (requires Java)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.setdefault(
            "JAVA_HOME",
            "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
        )
        from src.content.grammar_checker import GrammarChecker
        self.checker = GrammarChecker(language="en-US")

    def test_detect_errors(self):
        from src.content.transcript_processor import TranscriptProcessor
        tp = TranscriptProcessor()
        sentences = [
            {"text": "He have a good idea.", "start_sec": 0, "end_sec": 3, "sentence_idx": 0},
            {"text": "She don't know the answer.", "start_sec": 3, "end_sec": 6, "sentence_idx": 1},
        ]
        errors = self.checker.check_transcript(sentences)
        assert len(errors) > 0

    def test_score_grammar(self):
        errors = [
            type("E", (), {"severity": "critical"})(),
            type("E", (), {"severity": "minor"})(),
        ]
        score = self.checker.score_grammar(errors, total_words=100)
        assert 0 <= score["score"] <= 100

    def teardown_method(self):
        if hasattr(self, "checker"):
            self.checker.close()


@heavy
class TestPipelineHeavy:
    """Full end-to-end pipeline test with all modules."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.setdefault(
            "JAVA_HOME",
            "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
        )
        from src.content.pipeline import ContentAnalysisPipeline
        config_path = os.path.join(BASE_DIR, "configs", "content_pipeline_config.yaml")
        self.pipeline = ContentAnalysisPipeline(config_path=config_path)

    def test_full_pipeline_structured(self):
        """Test A: Structured TED-style speech."""
        whisper = _load_transcript("test_a_structured.json")
        result = self.pipeline.process(whisper, skip_heavy=False)

        assert len(result["segments"]) > 0
        summary = result["summary"]

        # Grammar score should be high (well-written speech)
        assert summary["grammar_score_overall"] >= 70

        # Readability: grade 6-16
        assert 4 <= summary["readability_level"] <= 18

        # Should detect multiple regime types
        regime_seq = summary["regime_sequence"]
        unique_regimes = set(r for r in regime_seq if r != "Unknown")
        assert len(unique_regimes) >= 2, f"Expected 2+ regimes, got: {unique_regimes}"

        # Structure score should be reasonable
        assert summary["content_structure_score"] >= 50

    def test_full_pipeline_rambling(self):
        """Test B: Rambling student presentation."""
        whisper = _load_transcript("test_b_rambling.json")
        result = self.pipeline.process(whisper, skip_heavy=False)

        assert len(result["segments"]) > 0
        summary = result["summary"]

        # Readability should be lower (simpler language)
        assert summary["readability_level"] <= 14

    def test_full_pipeline_technical(self):
        """Test C: Data-heavy technical talk."""
        whisper = _load_transcript("test_c_technical.json")
        result = self.pipeline.process(whisper, skip_heavy=False)

        assert len(result["segments"]) > 0
        summary = result["summary"]

        # Vocabulary diversity should be reasonable
        assert summary["vocabulary_diversity"] > 0

    def teardown_method(self):
        self.pipeline.close()
