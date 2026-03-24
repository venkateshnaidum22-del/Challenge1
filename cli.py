"""
cli.py — Command-line interface for the Empathy Engine.

Usage:
    python cli.py "Your text here"
    python cli.py "Your text here" --output my_audio.mp3
"""

import argparse
import sys
from pathlib import Path

from emotion_detector import detect_emotion
from voice_mapper import get_voice_params
from tts_engine import synthesize, build_ssml


def main():
    parser = argparse.ArgumentParser(
        description="🎙 Empathy Engine — Emotionally expressive text-to-speech"
    )
    parser.add_argument("text", nargs="?", help="Text to synthesise")
    parser.add_argument("--output", "-o", default="output.mp3", help="Output audio file path")
    parser.add_argument("--ssml", action="store_true", help="Print SSML markup")
    args = parser.parse_args()

    text = args.text
    if not text:
        text = input("Enter text: ").strip()
    if not text:
        print("No text provided. Exiting.")
        sys.exit(1)

    print("\n🔍 Detecting emotion…")
    result = detect_emotion(text)
    print(f"   Emotion    : {result.label.upper()}  (confidence: {result.confidence:.0%})")
    print(f"   Intensity  : {result.intensity:.0%}")

    params = get_voice_params(result.label, result.intensity)
    print(f"\n🎚  Voice parameters ({params.label}):")
    print(f"   Rate       : {params.rate} wpm")
    pitch_sign = "+" if params.pitch >= 0 else ""
    print(f"   Pitch      : {pitch_sign}{params.pitch} semitones")
    print(f"   Volume     : {params.volume:.0%}")
    print(f"   Emphasis   : {params.emphasis:.2f}x")

    if args.ssml:
        ssml = build_ssml(text, params)
        print(f"\n📄 SSML:\n{ssml}")

    print(f"\n🔊 Synthesising audio…")
    output_path = synthesize(text, params, args.output)
    print(f"✅ Audio saved to: {output_path}\n")


if __name__ == "__main__":
    main()
