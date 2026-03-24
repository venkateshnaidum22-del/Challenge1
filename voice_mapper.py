"""
voice_mapper.py
Maps detected emotion + intensity to vocal parameters for TTS modulation.
Covers gTTS (rate via audio manipulation) and pyttsx3 (direct parameter control).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VoiceParams:
    """
    rate     : words-per-minute (pyttsx3) or speed multiplier (audio post-processing)
    pitch    : semitone shift relative to baseline (−12 to +12). Applied via audio post-process.
    volume   : amplitude multiplier (0.0 = silent, 1.0 = normal, 2.0 = double)
    emphasis : pause_factor — extra pauses between phrases (multiplier on silence length)
    label    : human-readable label for logging/display
    """
    rate: float         # words per minute for pyttsx3; multiplier for audio speed
    pitch: float        # semitone offset
    volume: float       # amplitude multiplier
    emphasis: float     # pause multiplier
    label: str


# Baseline (neutral) parameters
BASELINE = VoiceParams(rate=175, pitch=0.0, volume=1.0, emphasis=1.0, label="Neutral")

# Per-emotion base configuration
EMOTION_BASE: dict[str, VoiceParams] = {
    "happy": VoiceParams(
        rate=190, pitch=2.0, volume=1.1, emphasis=0.9, label="Happy"
    ),
    "excited": VoiceParams(
        rate=215, pitch=4.5, volume=1.3, emphasis=0.7, label="Excited"
    ),
    "sad": VoiceParams(
        rate=140, pitch=-3.0, volume=0.8, emphasis=1.5, label="Sad"
    ),
    "angry": VoiceParams(
        rate=200, pitch=-1.5, volume=1.5, emphasis=0.6, label="Angry"
    ),
    "frustrated": VoiceParams(
        rate=185, pitch=-1.0, volume=1.2, emphasis=0.8, label="Frustrated"
    ),
    "fearful": VoiceParams(
        rate=160, pitch=1.0, volume=0.9, emphasis=1.4, label="Fearful"
    ),
    "surprised": VoiceParams(
        rate=210, pitch=5.0, volume=1.2, emphasis=0.7, label="Surprised"
    ),
    "inquisitive": VoiceParams(
        rate=170, pitch=1.5, volume=1.0, emphasis=1.1, label="Inquisitive"
    ),
    "neutral": BASELINE,
}


def get_voice_params(emotion_label: str, intensity: float) -> VoiceParams:
    """
    Returns final VoiceParams for the given emotion, scaled by intensity.

    Intensity scaling (bonus objective):
      - intensity = 0.0  → parameters are softened 50% toward baseline
      - intensity = 0.5  → use the base emotion config as-is
      - intensity = 1.0  → parameters are boosted 50% beyond base
    """
    base = EMOTION_BASE.get(emotion_label, BASELINE)

    # Normalise intensity into a [-0.5, +0.5] scaling delta around the base
    scale = (intensity - 0.5)  # range: -0.5 to +0.5

    def blend(base_val: float, neutral_val: float) -> float:
        """Linearly interpolate between neutral and an amplified base value."""
        delta = base_val - neutral_val
        scaled_delta = delta * (1 + scale)
        return neutral_val + scaled_delta

    return VoiceParams(
        rate=round(blend(base.rate, BASELINE.rate), 1),
        pitch=round(blend(base.pitch, BASELINE.pitch), 2),
        volume=round(blend(base.volume, BASELINE.volume), 3),
        emphasis=round(blend(base.emphasis, BASELINE.emphasis), 3),
        label=base.label,
    )
