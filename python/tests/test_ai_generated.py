"""Tests for tobira.adversarial.ai_generated — AI-generated text detection."""

from __future__ import annotations

from tobira.adversarial.ai_generated import AIGeneratedDetector, AIGeneratedResult


class TestAIGeneratedResult:
    def test_default_indicators_empty(self) -> None:
        result = AIGeneratedResult(detected=False, confidence=0.0)
        assert result.indicators == []

    def test_frozen(self) -> None:
        import pytest

        result = AIGeneratedResult(detected=False, confidence=0.0)
        with pytest.raises(AttributeError):
            result.detected = True  # type: ignore[misc]

    def test_with_indicators(self) -> None:
        result = AIGeneratedResult(
            detected=True,
            confidence=0.85,
            indicators=["low_entropy", "uniform_vocabulary"],
        )
        assert result.detected is True
        assert result.confidence == 0.85
        assert len(result.indicators) == 2


class TestAIGeneratedDetectorShortText:
    def test_empty_string(self) -> None:
        detector = AIGeneratedDetector()
        result = detector.detect("")
        assert result.detected is False
        assert result.confidence == 0.0

    def test_short_text(self) -> None:
        detector = AIGeneratedDetector()
        result = detector.detect("Hello world")
        assert result.detected is False
        assert result.confidence == 0.0

    def test_below_minimum_length(self) -> None:
        detector = AIGeneratedDetector()
        result = detector.detect("a" * 49)
        assert result.detected is False


class TestAIGeneratedDetectorEntropy:
    def test_character_entropy_moderate(self) -> None:
        score = AIGeneratedDetector._character_entropy(
            "This is a sample text with moderate entropy for testing purposes."
        )
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_character_entropy_very_low(self) -> None:
        # Highly repetitive text should have very low entropy
        score = AIGeneratedDetector._character_entropy("aaaa" * 100)
        assert score is not None
        assert score == 0.0  # Below 2.5 threshold

    def test_character_entropy_empty(self) -> None:
        score = AIGeneratedDetector._character_entropy("")
        assert score is None


class TestAIGeneratedDetectorVocabulary:
    def test_vocabulary_uniformity_short(self) -> None:
        # Too few tokens → None
        score = AIGeneratedDetector._vocabulary_uniformity("short text")
        assert score is None

    def test_vocabulary_uniformity_sufficient(self) -> None:
        text = " ".join(
            [
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "the",
                "lazy",
                "dog",
                "and",
                "the",
                "cat",
                "sat",
                "on",
                "the",
                "mat",
                "while",
                "birds",
                "flew",
                "across",
                "the",
                "sky",
            ]
        )
        score = AIGeneratedDetector._vocabulary_uniformity(text)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_vocabulary_uniformity_cjk(self) -> None:
        # CJK text uses character-level analysis
        text = (
            "今日は天気が良いので散歩に出かけました。"
            "公園では桜の花が美しく咲いていて、"
            "多くの人々が花見を楽しんでいました。"
        )
        score = AIGeneratedDetector._vocabulary_uniformity(text)
        # May be None if < 20 characters after filtering
        if score is not None:
            assert 0.0 <= score <= 1.0


class TestAIGeneratedDetectorSentenceLength:
    def test_too_few_sentences(self) -> None:
        score = AIGeneratedDetector._sentence_length_uniformity("One sentence only.")
        assert score is None

    def test_uniform_sentences(self) -> None:
        text = (
            "This sentence is exactly twenty characters long. "
            "This sentence is exactly twenty characters long. "
            "This sentence is exactly twenty characters long. "
            "This sentence is exactly twenty characters long."
        )
        score = AIGeneratedDetector._sentence_length_uniformity(text)
        assert score is not None
        assert score >= 0.6  # Uniform = high score

    def test_varied_sentences(self) -> None:
        text = (
            "Short. "
            "This is a medium length sentence with some words. "
            "And this is a very long sentence that goes on "
            "and on with many words and clauses. "
            "Tiny."
        )
        score = AIGeneratedDetector._sentence_length_uniformity(text)
        assert score is not None
        assert score <= 0.3  # Varied = low score


class TestAIGeneratedDetectorRepetition:
    def test_too_few_tokens(self) -> None:
        score = AIGeneratedDetector._repetition_patterns("too few")
        assert score is None

    def test_repetitive_text(self) -> None:
        text = " ".join(["buy now today"] * 10)
        score = AIGeneratedDetector._repetition_patterns(text)
        assert score is not None
        assert score >= 0.5

    def test_diverse_text(self) -> None:
        words = [
            "alpha", "bravo", "charlie", "delta", "echo",
            "foxtrot", "golf", "hotel", "india", "juliet",
            "kilo", "lima", "mike", "november", "oscar",
            "papa", "quebec", "romeo", "sierra", "tango",
            "uniform", "victor", "whiskey", "xray", "yankee",
        ]
        text = " ".join(words)
        score = AIGeneratedDetector._repetition_patterns(text)
        assert score is not None
        assert score <= 0.3


class TestAIGeneratedDetectorPunctuation:
    def test_too_few_sentences(self) -> None:
        score = AIGeneratedDetector._punctuation_regularity("No sentences here")
        assert score is None

    def test_regular_punctuation(self) -> None:
        text = (
            "First part, second part. "
            "Another part, more text. "
            "Third part, final bit."
        )
        score = AIGeneratedDetector._punctuation_regularity(text)
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestAIGeneratedDetectorIntegration:
    def test_detect_returns_result(self) -> None:
        detector = AIGeneratedDetector()
        text = (
            "Dear valued customer, we are writing to inform you about "
            "an exciting opportunity. Our company offers premium services "
            "that will enhance your experience. Please take a moment to "
            "review our latest offerings. We believe you will find them "
            "quite beneficial for your needs. Thank you for your continued "
            "support and loyalty. We look forward to serving you better."
        )
        result = detector.detect(text)
        assert isinstance(result, AIGeneratedResult)
        assert isinstance(result.detected, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.indicators, list)

    def test_custom_threshold(self) -> None:
        # Very high threshold should make detection harder
        detector = AIGeneratedDetector(threshold=0.99)
        text = (
            "This is a sample text that is long enough to be analysed "
            "by the detector module for statistical features. It needs "
            "to contain enough content for meaningful analysis. The text "
            "should have multiple sentences of varying structure."
        )
        result = detector.detect(text)
        assert result.detected is False

    def test_very_low_threshold(self) -> None:
        # Very low threshold — any text with scores should be detected
        detector = AIGeneratedDetector(threshold=0.01)
        text = (
            "This is a test. This is a test. This is a test. "
            "This is a test. This is a test. This is a test. "
            "This is a test. This is a test. This is a test."
        )
        result = detector.detect(text)
        assert result.detected is True

    def test_japanese_text(self) -> None:
        detector = AIGeneratedDetector()
        text = (
            "本日は晴天なり。明日も晴れるでしょう。週末には雨が降るかもしれません。"
            "来週の天気予報はまだ不確実です。気温は徐々に上昇する見込みです。"
            "外出の際は日焼け止めをお忘れなく。"
        )
        result = detector.detect(text)
        assert isinstance(result, AIGeneratedResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_indicators_only_include_high_scores(self) -> None:
        detector = AIGeneratedDetector()
        text = (
            "Dear valued customer, we are writing to inform you about "
            "an exciting new opportunity that awaits you. Our premium "
            "services are designed to enhance your daily experience. "
            "Please review our latest catalog of offerings today."
        )
        result = detector.detect(text)
        # Indicators should only contain names with scores >= 0.5
        for indicator in result.indicators:
            assert indicator in {
                "low_entropy",
                "uniform_vocabulary",
                "uniform_sentence_length",
                "repetitive_patterns",
                "regular_punctuation",
            }


class TestAIGeneratedDetectorModuleExports:
    def test_importable_from_package(self) -> None:
        from tobira.adversarial import AIGeneratedDetector, AIGeneratedResult

        assert AIGeneratedDetector is not None
        assert AIGeneratedResult is not None
