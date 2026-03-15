"""AI-generated text detection using statistical heuristics.

Detects characteristics of AI-generated text (e.g. SpamGPT) by analyzing
statistical features such as character entropy, vocabulary uniformity,
sentence length variance, and repetition patterns. No external ML model
is required — all analysis is purely statistical.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AIGeneratedResult:
    """Result of AI-generated text analysis.

    Attributes:
        detected: Whether AI-generated text was detected.
        confidence: Overall confidence score (0.0 to 1.0).
        indicators: Names of triggered indicators.
    """

    detected: bool
    confidence: float
    indicators: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentence splitting: handles both Western (. ! ?) and Japanese (。！？) punctuation
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")

# Word splitting for Latin scripts (simple whitespace + punctuation boundary)
_WORD_SPLIT = re.compile(r"[^\s\w]|\s+")

# CJK Unicode ranges for character-level segmentation detection
_CJK_PATTERN = re.compile(
    r"[\u3040-\u309F"  # Hiragana
    r"\u30A0-\u30FF"  # Katakana
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs
    r"\uF900-\uFAFF"  # CJK Compatibility Ideographs
    r"\u3400-\u4DBF"  # CJK Extension A
    r"]"
)

# Minimum text length for reliable analysis
_MIN_TEXT_LENGTH = 50

# Minimum number of sentences for sentence-level analysis
_MIN_SENTENCES = 3


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class AIGeneratedDetector:
    """Detect AI-generated text using statistical heuristics.

    Combines multiple lightweight indicators to estimate whether a given text
    was produced by a language model. Designed to run without external
    dependencies or GPU, making it suitable as a pre-processing step.

    Args:
        threshold: Confidence threshold above which text is flagged as
            AI-generated. Defaults to ``0.65``.
    """

    def __init__(self, *, threshold: float = 0.65) -> None:
        self._threshold = threshold

    def detect(self, text: str) -> AIGeneratedResult:
        """Analyse *text* for AI-generated characteristics.

        Args:
            text: Input text to analyse.

        Returns:
            An :class:`AIGeneratedResult` with detection outcome, confidence
            score, and the names of triggered indicators.
        """
        if len(text) < _MIN_TEXT_LENGTH:
            return AIGeneratedResult(detected=False, confidence=0.0)

        scores: list[tuple[str, float]] = []

        entropy_score = self._character_entropy(text)
        if entropy_score is not None:
            scores.append(("low_entropy", entropy_score))

        vocab_score = self._vocabulary_uniformity(text)
        if vocab_score is not None:
            scores.append(("uniform_vocabulary", vocab_score))

        sentence_score = self._sentence_length_uniformity(text)
        if sentence_score is not None:
            scores.append(("uniform_sentence_length", sentence_score))

        repetition_score = self._repetition_patterns(text)
        if repetition_score is not None:
            scores.append(("repetitive_patterns", repetition_score))

        punct_score = self._punctuation_regularity(text)
        if punct_score is not None:
            scores.append(("regular_punctuation", punct_score))

        if not scores:
            return AIGeneratedResult(detected=False, confidence=0.0)

        # Weighted average with equal weights
        confidence = sum(s for _, s in scores) / len(scores)
        confidence = max(0.0, min(1.0, confidence))

        indicators = [name for name, s in scores if s >= 0.5]

        return AIGeneratedResult(
            detected=confidence >= self._threshold,
            confidence=round(confidence, 4),
            indicators=indicators,
        )

    # -- Indicator helpers ---------------------------------------------------

    @staticmethod
    def _character_entropy(text: str) -> float | None:
        """Measure character-level Shannon entropy.

        AI-generated text tends to have *moderate* entropy — more uniform than
        human text which has burstier character distributions. Returns a score
        in [0, 1] where higher means more likely AI-generated.
        """
        if not text:
            return None

        freq = Counter(text)
        total = len(text)
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in freq.values()
        )

        # English text typically has entropy ~4.0-4.5 bits/char.
        # AI text tends toward 3.5-4.2 (slightly lower / more predictable).
        # We flag text with entropy in the "too uniform" band.
        # Very low entropy (< 2.5) is likely short/repetitive human text.
        if entropy < 2.5:
            return 0.0
        if entropy > 5.0:
            return 0.0
        # Peak suspicion around 3.5-4.0
        if 3.2 <= entropy <= 4.2:
            return 0.6 + 0.4 * (1.0 - abs(entropy - 3.7) / 0.5)
        return 0.3

    @staticmethod
    def _vocabulary_uniformity(text: str) -> float | None:
        """Measure vocabulary diversity via type-token ratio (TTR).

        AI-generated text often has a more uniform TTR compared to human
        text which tends to reuse certain words more heavily or use more
        varied vocabulary depending on context.
        """
        is_cjk = bool(_CJK_PATTERN.search(text))
        if is_cjk:
            # For CJK, use character-level analysis
            tokens = [ch for ch in text if not ch.isspace()]
        else:
            tokens = [t.lower() for t in _WORD_SPLIT.split(text) if t.strip()]

        if len(tokens) < 20:
            return None

        unique = len(set(tokens))
        ttr = unique / len(tokens)

        # AI-generated text typically has TTR in 0.4-0.7 range
        # Human text is more variable (very low for repetitive, very high
        # for creative writing)
        if 0.45 <= ttr <= 0.65:
            return 0.6
        if 0.35 <= ttr <= 0.75:
            return 0.3
        return 0.1

    @staticmethod
    def _sentence_length_uniformity(text: str) -> float | None:
        """Measure variation in sentence lengths.

        AI-generated text tends to produce sentences of similar length,
        while human text has more variable sentence lengths.
        """
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) < _MIN_SENTENCES:
            return None

        lengths = [len(s) for s in sentences]
        mean = sum(lengths) / len(lengths)
        if mean == 0:
            return None

        variance = sum((v - mean) ** 2 for v in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean  # Coefficient of variation

        # Low CV = uniform sentence lengths = more likely AI-generated
        # Human text typically has CV > 0.5
        if cv < 0.25:
            return 0.8
        if cv < 0.40:
            return 0.6
        if cv < 0.55:
            return 0.3
        return 0.1

    @staticmethod
    def _repetition_patterns(text: str) -> float | None:
        """Detect repetitive n-gram patterns.

        AI-generated text sometimes repeats phrases or structural patterns
        more frequently than human text.
        """
        is_cjk = bool(_CJK_PATTERN.search(text))
        if is_cjk:
            tokens = [ch for ch in text if not ch.isspace()]
        else:
            tokens = [t.lower() for t in _WORD_SPLIT.split(text) if t.strip()]

        if len(tokens) < 20:
            return None

        # Analyse trigram repetitions
        ngram_size = 3
        ngrams: list[str] = []
        for i in range(len(tokens) - ngram_size + 1):
            ngrams.append(" ".join(tokens[i : i + ngram_size]))

        if not ngrams:
            return None

        freq = Counter(ngrams)
        repeated = sum(1 for count in freq.values() if count > 1)
        ratio = repeated / len(freq) if freq else 0.0

        # Higher repetition ratio = more likely AI-generated
        if ratio > 0.3:
            return 0.7
        if ratio > 0.15:
            return 0.5
        if ratio > 0.05:
            return 0.3
        return 0.1

    @staticmethod
    def _punctuation_regularity(text: str) -> float | None:
        """Detect overly regular punctuation usage.

        AI-generated text often uses punctuation in predictable patterns
        (e.g. consistent comma placement, regular period spacing).
        """
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) < _MIN_SENTENCES:
            return None

        # Analyse comma density per sentence
        densities = []
        for sent in sentences:
            comma_count = sent.count(",") + sent.count("、")
            word_count = max(len(sent.split()), 1)
            densities.append(comma_count / word_count)

        if not densities:
            return None

        mean_density = sum(densities) / len(densities)
        if mean_density == 0:
            # No commas at all — not informative
            return 0.2

        variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
        cv = math.sqrt(variance) / mean_density if mean_density > 0 else 0.0

        # Low CV in comma usage = overly regular = more likely AI-generated
        if cv < 0.3:
            return 0.6
        if cv < 0.5:
            return 0.4
        return 0.1
