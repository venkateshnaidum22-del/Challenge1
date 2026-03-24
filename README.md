# Challenge1
# 🎙 Empathy Engine — Emotionally Expressive Text-to-Speech

> *Challenge 1 Submission — AI-Driven Sales & Customer Interaction Assessment*

---

## Overview

The Empathy Engine is a service that detects the emotional tone of any input text and synthesises speech with vocal characteristics that match that emotion. Rather than delivering a flat, robotic voice, the engine dynamically modulates **rate**, **pitch**, and **volume** based on detected emotion and its intensity, producing human-like, resonant audio.

---

## Architecture

```
Input Text
    │
    ▼
[ Emotion Detector ]          ← HuggingFace distilroberta (online) / keyword fallback
    │
    ├── emotion label (e.g. "excited")
    └── intensity score (0.0 – 1.0)
    │
    ▼
[ Voice Parameter Mapper ]    ← Maps emotion + intensity → {rate, pitch, volume}
    │
    ▼
[ TTS Engine ]
    ├── gTTS → raw MP3 → pydub post-processing (speed/pitch/volume)
    └── pyttsx3 fallback (offline)
    │
    ▼
Audio Output (.mp3)
```

---

## Emotion Categories (Granular — Bonus Objective ✅)

| Emotion | Rate (WPM) | Pitch (st) | Volume |
|---|---|---|---|
| Excited | 215 | +4.5 | 130% |
| Happy | 190 | +2.0 | 110% |
| Neutral | 175 | 0 | 100% |
| Inquisitive | 170 | +1.5 | 100% |
| Fearful | 160 | +1.0 | 90% |
| Sad | 140 | −3.0 | 80% |
| Frustrated | 185 | −1.0 | 120% |
| Angry | 200 | −1.5 | 150% |
| Surprised | 210 | +5.0 | 120% |

> Values shown are at baseline intensity (0.5). Intensity scaling interpolates between baseline and amplified extremes.

---

## Intensity Scaling (Bonus Objective ✅)

Every emotion has its parameters scaled by the detected intensity (0.0–1.0):

- **"This is good."** → intensity ≈ 0.1 → mild pitch/rate increase
- **"This is THE BEST NEWS EVER!!!"** → intensity ≈ 0.9 → maximum pitch/rate boost

Signals used to compute intensity: exclamation marks, ALL-CAPS words, booster adverbs ("extremely", "totally", etc.)

---

## SSML Support (Bonus Objective ✅)

Every request also generates an SSML string compatible with Google Cloud TTS and AWS Polly:

```xml
<speak>
  <prosody rate="123%" pitch="+4.5st" volume="130%">
    This is absolutely incredible!
  </prosody>
</speak>
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd empathy-engine
pip install -r requirements.txt
```

> **Note:** The HuggingFace emotion model (`j-hartmann/emotion-english-distilroberta-base`) will be downloaded automatically on first run (~250 MB). An internet connection is required for this and for gTTS. The service works fully offline using pyttsx3 + keyword-based detection if needed.

### 2. (Optional) Environment variables

```bash
# .env file — only needed for Google Cloud TTS or ElevenLabs integration
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
ELEVENLABS_API_KEY=your_key_here
```

---

## Running the Service

### Web Interface (recommended)

```bash
python app.py
# Visit http://localhost:5000
```

Type any text and hear it spoken with emotional expression. The UI shows:
- Detected emotion + confidence
- Intensity score (progress bar)
- Voice parameters used
- SSML markup
- Embedded audio player

### CLI

```bash
python cli.py "This is absolutely incredible news!"
python cli.py "I can't believe this happened again." --output frustrated.mp3
python cli.py "How does this feature actually work?" --ssml
```

---

## Design Choices

### Emotion Detection
The service tries a HuggingFace transformer model first (`j-hartmann/emotion-english-distilroberta-base`), which is a DistilRoBERTa model fine-tuned on multiple emotion datasets (GoEmotions, ISEAR, etc.). This gives 7 base emotions with per-class confidence scores.

A keyword-based fallback was implemented for two reasons:
1. Robustness in offline or rate-limited environments.
2. The keyword model adds the "inquisitive" category not present in the HF model.

### Vocal Modulation
Two layers of modulation are applied:
1. **Direct TTS parameter control** via pyttsx3 (rate, volume, voice selection for pitch approximation)
2. **Post-processing via pydub** on gTTS output: frame-rate manipulation for pitch shifting (mathematically precise semitone shifts); speed change without pitch drift; dB-level volume adjustment.

This satisfies requirement #3 (≥2 distinct vocal parameters) and goes significantly beyond it.

### Intensity Scaling
Rather than binary emotion categories, the emotion label is treated as a *direction* and intensity as a *magnitude*. The final parameters are interpolated linearly between the neutral baseline and the amplified emotion extreme. This means the same emotion label produces subtly different audio for mild vs. strong expressions.

---

## File Structure

```
empathy-engine/
├── app.py               # Flask web application
├── cli.py               # Command-line interface
├── emotion_detector.py  # Emotion classification (HF + fallback)
├── voice_mapper.py      # Emotion → VoiceParams mapping
├── tts_engine.py        # Speech synthesis + audio post-processing
├── requirements.txt
└── outputs/
    └── audio/           # Generated audio files
```
