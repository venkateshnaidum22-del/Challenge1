"""
emotion_detector.py
Detects emotion from input text using a Hugging Face transformer model.
Falls back to VADER/TextBlob-style rule-based scoring if the model is unavailable.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

# Granular emotion categories (bonus: beyond simple pos/neg/neutral)
EmotionLabel = Literal[
    "happy", "excited", "sad", "angry", "frustrated",
    "fearful", "surprised", "inquisitive", "neutral"
]

EMOTION_KEYWORDS: dict[EmotionLabel, list[str]] = {
    "excited": [
        "amazing", "incredible", "fantastic", "best news", "wonderful",
        "thrilled", "can't wait", "awesome", "love this", "outstanding",
        "spectacular", "brilliant", "great", "excellent", "yes!"
    ],
    "happy": [
        "happy", "glad", "pleased", "delighted", "good news", "thankful",
        "grateful", "appreciate", "nice", "enjoy", "joyful", "smile",
    ],
    "frustrated": [
        "frustrated", "annoying", "irritating", "ridiculous", "unacceptable",
        "terrible", "waste", "broken", "useless", "nonsense", "again",
    ],
    "angry": [
        "angry", "furious", "outrage", "demand", "unbelievable", "pathetic",
        "disgusting", "hate", "worst", "incompetent", "never", "lawsuit",
    ],
    "sad": [
        "sad", "disappointed", "unfortunate", "sorry to hear", "regret",
        "heartbroken", "miss", "lost", "grief", "tragic", "unfortunately",
    ],
    "fearful": [
        "worried", "scared", "afraid", "concerned", "anxious", "nervous",
        "panic", "dread", "fear", "uncertain", "unsure", "risk",
    ],
    "surprised": [
        "wow", "really?", "seriously?", "no way", "unexpected", "surprised",
        "shocked", "unbelievable", "whoa", "wait—", "didn't expect",
    ],
    "inquisitive": [
        "how", "why", "what", "when", "where", "could you", "can you",
        "would you", "is it", "are you", "do you", "does it", "explain",
        "tell me", "clarify", "wondering",
    ],
    "neutral": [],
}

# Intensity signals — multiplier for modulation strength
INTENSITY_BOOSTERS = [
    "very", "extremely", "incredibly", "absolutely", "totally",
    "completely", "utterly", "so", "really", "truly", "deeply",
    "!!", "!!!", "CAPS"
]


@dataclass
class EmotionResult:
    label: EmotionLabel
    confidence: float          # 0.0 – 1.0
    intensity: float           # 0.0 – 1.0  (used for bonus intensity scaling)
    raw_scores: dict[str, float]


def _keyword_score(text: str) -> dict[str, float]:
    """Simple keyword frequency scoring."""
    lower = text.lower()
    scores: dict[str, float] = {}
    for label, keywords in EMOTION_KEYWORDS.items():
        if not keywords:
            scores[label] = 0.1  # baseline for neutral
            continue
        hits = sum(1 for kw in keywords if kw in lower)
        scores[label] = hits
    return scores


def _intensity_score(text: str) -> float:
    """Returns 0.0–1.0 representing detected intensity."""
    lower = text.lower()
    score = 0.0
    for booster in INTENSITY_BOOSTERS:
        if booster in lower:
            score += 0.15
    # ALL CAPS words are a strong intensity signal
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    score += caps_words * 0.2
    # Exclamation marks
    score += text.count("!") * 0.1
    return min(score, 1.0)


def detect_emotion_hf(text: str) -> EmotionResult | None:
    """
    Try to use a Hugging Face emotion classification pipeline.
    Returns None if the model cannot be loaded (e.g., offline environment).
    Maps HF labels to our internal EmotionLabel set.
    """
    try:
        from transformers import pipeline  # type: ignore
        clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        results = clf(text[:512])[0]  # truncate for speed
        # HF model outputs: joy, sadness, anger, fear, surprise, disgust, neutral
        label_map = {
            "joy": "happy",
            "sadness": "sad",
            "anger": "angry",
            "fear": "fearful",
            "surprise": "surprised",
            "disgust": "frustrated",
            "neutral": "neutral",
        }
        raw_scores = {label_map.get(r["label"].lower(), "neutral"): r["score"] for r in results}
        best = max(raw_scores, key=lambda k: raw_scores[k])
        confidence = raw_scores[best]

        # Refine "happy" to "excited" if intensity is high
        intensity = _intensity_score(text)
        if best == "happy" and intensity > 0.4:
            best = "excited"

        return EmotionResult(
            label=best,
            confidence=confidence,
            intensity=intensity,
            raw_scores=raw_scores,
        )
    except Exception:
        return None


def detect_emotion_fallback(text: str) -> EmotionResult:
    """
    Rule-based fallback using keyword scoring.
    Also handles inquisitive detection via question marks and question words.
    """
    scores = _keyword_score(text)
    intensity = _intensity_score(text)

    # Inquisitive boost: ends with '?' or has multiple question words
    if text.strip().endswith("?"):
        scores["inquisitive"] = scores.get("inquisitive", 0) + 2

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values()) or 1
    confidence = scores[best] / total

    # Tie or very low scores → neutral
    if confidence < 0.25:
        best = "neutral"
        confidence = 0.5

    # Upgrade happy → excited if intensity is high
    if best == "happy" and intensity > 0.5:
        best = "excited"

    return EmotionResult(
        label=best,
        confidence=round(confidence, 3),
        intensity=round(intensity, 3),
        raw_scores={k: round(v / total, 3) for k, v in scores.items()},
    )


def detect_emotion(text: str) -> EmotionResult:
    """
    Primary entry point. Tries the HuggingFace model first; falls back to
    the keyword-based detector if the model is unavailable.
    """
    result = detect_emotion_hf(text)
    if result is not None:
        return result
    return detect_emotion_fallback(text)
