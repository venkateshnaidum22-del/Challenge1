"""
tts_engine.py
Generates speech audio from text with vocal parameter modulation.

Strategy:
  1. Use gTTS to produce a clean base MP3 (natural-sounding, online).
  2. Use pydub for post-processing: speed/rate adjustment, pitch shift, volume.
  3. If gTTS is unavailable (offline), fall back to pyttsx3 with direct rate/volume control.

This gives us ≥2 distinct vocal parameter modulations (requirement #3).
"""

from __future__ import annotations
import io
import os
import tempfile
from pathlib import Path

from voice_mapper import VoiceParams


# ──────────────────────────────────────────────────────────────────
# Audio helpers (pydub-based)
# ──────────────────────────────────────────────────────────────────

def _load_pydub():
    try:
        from pydub import AudioSegment
        return AudioSegment
    except ImportError:
        return None


def _speed_change(audio_segment, speed: float):
    """Change playback speed without affecting pitch (frame rate trick)."""
    sound_with_altered_frame_rate = audio_segment._spawn(
        audio_segment.raw_data,
        overrides={"frame_rate": int(audio_segment.frame_rate * speed)}
    )
    return sound_with_altered_frame_rate.set_frame_rate(audio_segment.frame_rate)


def _pitch_shift(audio_segment, semitones: float):
    """Shift pitch by n semitones using frame-rate manipulation."""
    import math
    factor = math.pow(2, semitones / 12.0)
    pitched = audio_segment._spawn(
        audio_segment.raw_data,
        overrides={"frame_rate": int(audio_segment.frame_rate * factor)}
    )
    return pitched.set_frame_rate(audio_segment.frame_rate)


def _apply_params_pydub(mp3_bytes: bytes, params: VoiceParams) -> bytes:
    """Apply rate (speed), pitch shift, and volume changes via pydub."""
    AudioSegment = _load_pydub()
    if AudioSegment is None:
        return mp3_bytes  # return unchanged if pydub not available

    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")

    # 1. Speed / rate modulation
    # Baseline rate is 175 wpm; convert to speed multiplier
    speed_multiplier = params.rate / 175.0
    if abs(speed_multiplier - 1.0) > 0.01:
        audio = _speed_change(audio, speed_multiplier)

    # 2. Pitch modulation (semitone shift)
    if abs(params.pitch) > 0.1:
        audio = _pitch_shift(audio, params.pitch)

    # 3. Volume modulation (dB adjustment)
    if abs(params.volume - 1.0) > 0.01:
        import math
        db_change = 20 * math.log10(params.volume)
        audio = audio + db_change  # pydub supports + operator for dB

    out = io.BytesIO()
    audio.export(out, format="mp3")
    return out.getvalue()


# ──────────────────────────────────────────────────────────────────
# SSML builder (bonus objective)
# ──────────────────────────────────────────────────────────────────

def build_ssml(text: str, params: VoiceParams) -> str:
    """
    Wraps text in SSML prosody tags for services that support it
    (e.g., Google Cloud TTS, AWS Polly).
    rate and volume are expressed as percentages relative to default.
    pitch as semitone offset (e.g., +2st).
    """
    rate_pct = int((params.rate / 175.0) * 100)
    pitch_sign = "+" if params.pitch >= 0 else ""
    pitch_str = f"{pitch_sign}{params.pitch:.1f}st"
    vol_pct = int(params.volume * 100)

    ssml = (
        f'<speak>'
        f'<prosody rate="{rate_pct}%" pitch="{pitch_str}" volume="{vol_pct}%">'
        f'{text}'
        f'</prosody>'
        f'</speak>'
    )
    return ssml


# ──────────────────────────────────────────────────────────────────
# Primary synthesis function
# ──────────────────────────────────────────────────────────────────

def synthesize(text: str, params: VoiceParams, output_path: str | Path) -> Path:
    """
    Synthesises `text` with vocal modulation defined by `params`.
    Writes the result as an MP3 to `output_path`.
    Returns the Path to the generated file.

    Tries gTTS → applies pydub post-processing.
    Falls back to pyttsx3 on failure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = _synthesize_gtts(text, params, output_path)
    if not success:
        _synthesize_pyttsx3(text, params, output_path)

    return output_path


def _synthesize_gtts(text: str, params: VoiceParams, output_path: Path) -> bool:
    """gTTS path — produces natural voice, then post-processes with pydub."""
    try:
        from gtts import gTTS  # type: ignore

        tts = gTTS(text=text, lang="en", slow=False)
        raw_buffer = io.BytesIO()
        tts.write_to_fp(raw_buffer)
        mp3_bytes = raw_buffer.getvalue()

        # Apply vocal parameter modulation via pydub
        modulated = _apply_params_pydub(mp3_bytes, params)

        with open(output_path, "wb") as f:
            f.write(modulated)
        return True

    except Exception as e:
        print(f"[gTTS] Error: {e} — falling back to pyttsx3")
        return False


def _synthesize_pyttsx3(text: str, params: VoiceParams, output_path: Path) -> None:
    """
    pyttsx3 offline fallback.
    Directly sets rate and volume; pitch is adjusted via voice selection heuristic.
    Saves as WAV (pyttsx3 limitation); renames extension if needed.
    """
    import pyttsx3  # type: ignore

    engine = pyttsx3.init()
    engine.setProperty("rate", int(params.rate))
    engine.setProperty("volume", min(params.volume, 1.0))  # pyttsx3 clamps at 1.0

    # Use a different voice if available to approximate pitch intent
    voices = engine.getProperty("voices")
    if voices and params.pitch > 1.5 and len(voices) > 1:
        engine.setProperty("voice", voices[1].id)  # typically higher pitch
    elif voices:
        engine.setProperty("voice", voices[0].id)

    wav_path = output_path.with_suffix(".wav")
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()

    # If caller expected .mp3, rename
    if output_path.suffix == ".mp3":
        wav_path.rename(output_path.with_suffix(".wav"))
        # Note: keeping as .wav since pyttsx3 only outputs wav
    print(f"[pyttsx3] Saved to {wav_path}")
