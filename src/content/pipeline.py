"""
Content Analysis Pipeline — Phase 9 of Modality 3.

Master orchestrator that chains all content analysis steps.
"""

import argparse
import json
import logging
import os
import time

import yaml

logger = logging.getLogger(__name__)


class ContentAnalysisPipeline:
    """Chains all content analysis modules into a single pipeline.

    Steps: preprocess → grammar → readability → vocabulary →
           regime detection → sentiment → tone → argument → assemble
    """

    def __init__(self, config_path: str = "configs/content_pipeline_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._modules = {}
        self._load_modules()

    def _load_modules(self):
        """Lazy-load all analysis modules."""
        logger.info("Loading content analysis modules...")

        from src.content.transcript_processor import TranscriptProcessor
        self._modules["transcript"] = TranscriptProcessor(
            sentence_splitter=self.config.get("transcript", {}).get("sentence_splitter", "spacy"),
            min_words_per_segment=self.config.get("transcript", {}).get("min_words_per_segment", 10),
        )

        from src.content.readability_scorer import ReadabilityScorer
        self._modules["readability"] = ReadabilityScorer()

        from src.content.vocabulary_analyzer import VocabularyAnalyzer
        self._modules["vocabulary"] = VocabularyAnalyzer(
            awl_path=self.config.get("vocabulary", {}).get("awl_path", "data/resources/academic_word_list.txt"),
            min_ttr_window=self.config.get("vocabulary", {}).get("min_ttr_window", 100),
        )

        from src.content.output_assembler import OutputAssembler
        self._modules["assembler"] = OutputAssembler(
            output_dir=self.config.get("output", {}).get("output_dir", "data/outputs"),
        )

        logger.info("Content analysis modules loaded (lazy modules deferred)")

    def _load_heavy_modules(self):
        """Load modules that require large model downloads or Java."""
        if "grammar" not in self._modules:
            try:
                from src.content.grammar_checker import GrammarChecker
                lang = self.config.get("grammar", {}).get("language", "en-US")
                self._modules["grammar"] = GrammarChecker(language=lang)
            except Exception as e:
                logger.warning(f"Grammar checker unavailable: {e}")
                self._modules["grammar"] = None

        if "regime" not in self._modules:
            try:
                from src.content.regime_detector import RegimeDetector
                regime_cfg = self.config.get("regime", {})
                self._modules["regime"] = RegimeDetector(
                    model_name=regime_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    half_window=regime_cfg.get("half_window", 5),
                    min_prominence=regime_cfg.get("min_prominence", 0.15),
                    min_segment_sentences=regime_cfg.get("min_segment_sentences", 3),
                )
            except Exception as e:
                logger.warning(f"Regime detector unavailable: {e}")
                self._modules["regime"] = None

        if "sentiment" not in self._modules:
            try:
                from src.content.sentiment_analyzer import SentimentAnalyzer
                model_name = self.config.get("sentiment", {}).get(
                    "model_name", "cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                self._modules["sentiment"] = SentimentAnalyzer(model_name=model_name)
            except Exception as e:
                logger.warning(f"Sentiment analyzer unavailable: {e}")
                self._modules["sentiment"] = None

        if "tone" not in self._modules:
            from src.content.tone_classifier import ToneClassifier
            self._modules["tone"] = ToneClassifier(use_llm=True)

        if "argument" not in self._modules:
            arg_engine = self.config.get("argument", {}).get("engine", "claude_cli")
            if arg_engine == "none":
                self._modules["argument"] = None
            else:
                from src.content.argument_analyzer import ArgumentAnalyzer
                self._modules["argument"] = ArgumentAnalyzer(use_llm=True)

    def process(self, transcript_input, skip_heavy: bool = False) -> dict:
        """Run the full content analysis pipeline.

        Args:
            transcript_input: Either a dict (Whisper output), a file path (JSON),
                              or a pre-processed transcript dict.
            skip_heavy: If True, skip grammar/regime/sentiment (for quick testing).

        Returns:
            Complete content analysis dict.
        """
        start_time = time.time()
        tp = self._modules["transcript"]

        # Step 1: Preprocess transcript
        logger.info("Step 1: Preprocessing transcript...")
        if isinstance(transcript_input, str):
            if os.path.isfile(transcript_input):
                transcript = tp.load_from_json(transcript_input)
            else:
                transcript = tp.load_from_text(transcript_input, duration_sec=0.0)
        elif isinstance(transcript_input, dict):
            if "sentences" in transcript_input:
                transcript = transcript_input  # Already processed
            else:
                transcript = tp.load_from_whisper(transcript_input)
        else:
            raise ValueError(f"Unsupported transcript input type: {type(transcript_input)}")

        logger.info(f"  {transcript['total_sentences']} sentences, {transcript['total_words']} words")

        # Load heavy modules
        if not skip_heavy:
            self._load_heavy_modules()

        # Step 2: Grammar
        grammar_errors = []
        grammar_score = {"score": 100, "total_errors": 0}
        grammar_mod = self._modules.get("grammar")
        if grammar_mod and not skip_heavy:
            logger.info("Step 2: Checking grammar...")
            # Clean text for grammar checking
            cleaned_sentences = []
            for s in transcript["sentences"]:
                cleaned = tp.clean_for_grammar(s["text"])
                cleaned_sentences.append({**s, "text": cleaned})
            grammar_errors = grammar_mod.check_transcript(cleaned_sentences)
            grammar_score = grammar_mod.score_grammar(grammar_errors, transcript["total_words"])
            logger.info(f"  Grammar score: {grammar_score['score']}")
        else:
            logger.info("Step 2: Grammar check skipped")

        # Step 3: Readability
        logger.info("Step 3: Computing readability...")
        readability = self._modules["readability"].score_transcript(transcript)
        logger.info(f"  FKGL: {readability['overall'].get('flesch_kincaid_grade', 'N/A')}")

        # Step 4: Vocabulary
        logger.info("Step 4: Analyzing vocabulary...")
        vocabulary = self._modules["vocabulary"].analyze_transcript(transcript)
        logger.info(f"  TTR: {vocabulary['overall'].get('ttr', 'N/A')}")

        # Step 5: Regime detection — produces the PRIMARY segmentation
        regimes = []
        regime_mod = self._modules.get("regime")
        if regime_mod and not skip_heavy:
            logger.info("Step 5: Detecting regimes...")
            regimes = regime_mod.detect_and_label(transcript)
            logger.info(f"  Found {len(regimes)} regimes")
        else:
            logger.info("Step 5: Regime detection skipped")

        # Build regime-based segments for downstream analysis
        # When regimes exist, use them as the primary unit (fewer, larger segments)
        if regimes:
            regime_segments = []
            for r in regimes:
                start_sent = r.start_sentence if hasattr(r, "start_sentence") else r.get("start_sentence", 0)
                end_sent = r.end_sentence if hasattr(r, "end_sentence") else r.get("end_sentence", 0)
                regime_segments.append({
                    "text": r.text if hasattr(r, "text") else r.get("text", ""),
                    "start_sec": r.start_sec if hasattr(r, "start_sec") else r.get("start_sec", 0),
                    "end_sec": r.end_sec if hasattr(r, "end_sec") else r.get("end_sec", 0),
                    "word_count": len((r.text if hasattr(r, "text") else r.get("text", "")).split()),
                    "sentence_indices": list(range(start_sent, end_sent)),
                })
            # Replace transcript segments with regime-based segments
            analysis_transcript = {**transcript, "segments": regime_segments}
            logger.info(f"  Using {len(regime_segments)} regime segments as primary unit")
        else:
            analysis_transcript = transcript

        # Step 6: Sentiment (on regime segments)
        sentiment = {"overall": {}, "segments": []}
        sentiment_mod = self._modules.get("sentiment")
        if sentiment_mod and not skip_heavy:
            logger.info("Step 6: Analyzing sentiment...")
            sentiment = sentiment_mod.analyze_transcript(analysis_transcript)
            logger.info(f"  Overall: {sentiment['overall'].get('polarity', 'N/A')}")
        else:
            logger.info("Step 6: Sentiment analysis skipped")

        # Step 7: Tone classification (on regime segments — few LLM calls)
        tone = {"dominant_tone": "Unknown", "segments": []}
        tone_mod = self._modules.get("tone")
        if tone_mod and not skip_heavy:
            logger.info("Step 7: Classifying tone...")
            tone = tone_mod.classify_transcript(analysis_transcript)
            logger.info(f"  Dominant tone: {tone.get('dominant_tone', 'N/A')}")
        else:
            logger.info("Step 7: Tone classification skipped")

        # Step 8: Argument analysis (on regime segments)
        argument_results = []
        arg_mod = self._modules.get("argument")
        if arg_mod and not skip_heavy and regimes:
            logger.info("Step 8: Analyzing arguments...")
            regime_types = [r.regime_type if hasattr(r, "regime_type") else r.get("regime_type", "Unknown")
                           for r in regimes]
            argument_results = arg_mod.analyze_full_speech(regime_segments, regime_types)
            logger.info(f"  Analyzed {len(argument_results)} segments")
        else:
            logger.info("Step 8: Argument analysis skipped")

        # Recompute readability/vocabulary on regime segments if available
        if regimes:
            logger.info("Step 3b: Recomputing readability on regime segments...")
            readability = self._modules["readability"].score_transcript(analysis_transcript)
            logger.info("Step 4b: Recomputing vocabulary on regime segments...")
            vocabulary = self._modules["vocabulary"].analyze_transcript(analysis_transcript)

        # Step 9: Assemble (using regime-aligned segments)
        logger.info("Step 9: Assembling output...")
        assembler = self._modules["assembler"]
        output = assembler.assemble(
            transcript=analysis_transcript,
            grammar_errors=grammar_errors,
            grammar_score=grammar_score,
            readability=readability,
            vocabulary=vocabulary,
            regimes=regimes,
            sentiment=sentiment,
            tone=tone,
            argument_results=argument_results,
        )

        elapsed = time.time() - start_time
        output["metadata"]["processing_time_sec"] = round(elapsed, 2)
        logger.info(f"Content analysis complete in {elapsed:.1f}s")

        return output

    def close(self):
        """Clean up resources."""
        grammar = self._modules.get("grammar")
        if grammar and hasattr(grammar, "close"):
            grammar.close()


def main():
    parser = argparse.ArgumentParser(description="Content Analysis Pipeline (Modality 3)")
    parser.add_argument("--transcript", required=True, help="Path to transcript JSON")
    parser.add_argument("--config", default="configs/content_pipeline_config.yaml")
    parser.add_argument("--output", default=None, help="Output filename")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip grammar/regime/sentiment")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = ContentAnalysisPipeline(config_path=args.config)
    result = pipeline.process(args.transcript, skip_heavy=args.skip_heavy)

    output_name = args.output or "content_analysis.json"
    path = pipeline._modules["assembler"].save(result, output_name)
    print(f"Output saved to: {path}")

    # Print summary
    summary = result.get("summary", {})
    print(f"\n--- Content Analysis Summary ---")
    print(f"Segments: {summary.get('total_segments', 0)}")
    print(f"Grammar score: {summary.get('grammar_score_overall', 'N/A')}")
    print(f"Readability (FKGL): {summary.get('readability_level', 'N/A')}")
    print(f"Vocabulary diversity: {summary.get('vocabulary_diversity', 'N/A')}")
    print(f"Dominant sentiment: {summary.get('dominant_sentiment', 'N/A')}")
    print(f"Dominant tone: {summary.get('dominant_tone', 'N/A')}")
    print(f"Structure score: {summary.get('content_structure_score', 'N/A')}")

    pipeline.close()


if __name__ == "__main__":
    main()
