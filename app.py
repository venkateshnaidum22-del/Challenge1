"""
app.py — Empathy Engine Web Interface
Flask application with:
  - POST /api/synthesize  → JSON response with emotion details + audio URL
  - GET  /audio/<filename> → stream generated audio
  - GET  /               → serve the single-page web UI
"""

from __future__ import annotations
import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string

from emotion_detector import detect_emotion
from voice_mapper import get_voice_params
from tts_engine import synthesize, build_ssml

app = Flask(__name__)

AUDIO_DIR = Path("outputs/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ── HTML Template ──────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Empathy Engine 🎙</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f0f2f5;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }
  .card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    max-width: 700px;
    width: 100%;
    padding: 2.5rem;
  }
  h1 { font-size: 1.8rem; color: #1a1a2e; margin-bottom: 0.25rem; }
  .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 2rem; }
  textarea {
    width: 100%;
    min-height: 120px;
    border: 1.5px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
    transition: border-color 0.2s;
    outline: none;
  }
  textarea:focus { border-color: #4f6ef7; }
  button {
    margin-top: 1rem;
    width: 100%;
    padding: 0.85rem;
    background: #4f6ef7;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }
  button:hover { background: #3a57e8; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .result {
    margin-top: 1.5rem;
    padding: 1.25rem;
    background: #f8f9ff;
    border-radius: 10px;
    display: none;
  }
  .emotion-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
  }
  .params-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem;
    margin: 0.75rem 0;
  }
  .param-item {
    background: white;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    border: 1px solid #e8eaf6;
  }
  .param-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
  .param-value { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; }
  audio { width: 100%; margin-top: 1rem; border-radius: 8px; }
  .ssml-block {
    margin-top: 0.75rem;
    background: #1e1e2e;
    color: #a8dadc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .error { color: #e74c3c; margin-top: 0.75rem; font-size: 0.9rem; }
  .spinner { display: none; text-align: center; margin-top: 1rem; color: #666; }
  .intensity-bar { height: 6px; background: #e0e0e0; border-radius: 3px; margin-top: 0.5rem; }
  .intensity-fill { height: 100%; border-radius: 3px; background: #4f6ef7; transition: width 0.4s; }

  /* Emotion colour palette */
  .emo-happy     { background: #fff3cd; color: #856404; }
  .emo-excited   { background: #d1ecf1; color: #0c5460; }
  .emo-sad       { background: #d6d8db; color: #383d41; }
  .emo-angry     { background: #f8d7da; color: #721c24; }
  .emo-frustrated{ background: #ffeeba; color: #664d03; }
  .emo-fearful   { background: #e2e3e5; color: #41464b; }
  .emo-surprised { background: #cce5ff; color: #004085; }
  .emo-inquisitive{background: #d4edda; color: #155724; }
  .emo-neutral   { background: #f8f9fa; color: #6c757d; }
</style>
</head>
<body>
<div class="card">
  <h1>🎙 Empathy Engine</h1>
  <p class="subtitle">Type any text and hear it spoken with matching emotional expression.</p>

  <textarea id="inputText" placeholder="Try: 'This is absolutely incredible news!' or 'I can't believe you did this again…'"></textarea>
  <button id="btn" onclick="synthesize()">Generate Emotional Speech</button>
  <div class="spinner" id="spinner">⏳ Analysing & synthesising…</div>
  <div id="error" class="error"></div>

  <div class="result" id="result">
    <span class="emotion-badge" id="emotionBadge">—</span>
    <div class="params-grid">
      <div class="param-item"><div class="param-label">Rate (WPM)</div><div class="param-value" id="pRate">—</div></div>
      <div class="param-item"><div class="param-label">Pitch Shift</div><div class="param-value" id="pPitch">—</div></div>
      <div class="param-item"><div class="param-label">Volume</div><div class="param-value" id="pVol">—</div></div>
      <div class="param-item"><div class="param-label">Confidence</div><div class="param-value" id="pConf">—</div></div>
    </div>
    <div>Intensity <div class="intensity-bar"><div class="intensity-fill" id="intensityBar" style="width:0%"></div></div></div>
    <audio id="player" controls></audio>
    <div class="ssml-block" id="ssmlBlock"></div>
  </div>
</div>

<script>
async function synthesize() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) return;
  document.getElementById('btn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('result').style.display = 'none';
  document.getElementById('error').textContent = '';

  try {
    const res = await fetch('/api/synthesize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');

    const badge = document.getElementById('emotionBadge');
    badge.textContent = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
    badge.className = 'emotion-badge emo-' + data.emotion;

    document.getElementById('pRate').textContent = data.params.rate + ' wpm';
    const pitchSign = data.params.pitch >= 0 ? '+' : '';
    document.getElementById('pPitch').textContent = pitchSign + data.params.pitch + ' st';
    document.getElementById('pVol').textContent = (data.params.volume * 100).toFixed(0) + '%';
    document.getElementById('pConf').textContent = (data.confidence * 100).toFixed(0) + '%';
    document.getElementById('intensityBar').style.width = (data.intensity * 100) + '%';
    document.getElementById('ssmlBlock').textContent = data.ssml;

    const player = document.getElementById('player');
    player.src = data.audio_url + '?t=' + Date.now();
    player.load();

    document.getElementById('result').style.display = 'block';
  } catch(e) {
    document.getElementById('error').textContent = '❌ ' + e.message;
  } finally {
    document.getElementById('btn').disabled = false;
    document.getElementById('spinner').style.display = 'none';
  }
}

document.getElementById('inputText').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) synthesize();
});
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/synthesize", methods=["POST"])
def api_synthesize():
    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 chars)"}), 400

    # 1. Emotion detection
    emotion_result = detect_emotion(text)

    # 2. Voice parameter mapping (with intensity scaling)
    params = get_voice_params(emotion_result.label, emotion_result.intensity)

    # 3. SSML generation
    ssml = build_ssml(text, params)

    # 4. Synthesise audio
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = AUDIO_DIR / filename
    synthesize(text, params, output_path)

    return jsonify({
        "emotion": emotion_result.label,
        "confidence": emotion_result.confidence,
        "intensity": emotion_result.intensity,
        "params": {
            "rate": params.rate,
            "pitch": params.pitch,
            "volume": params.volume,
            "emphasis": params.emphasis,
        },
        "ssml": ssml,
        "audio_url": f"/audio/{filename}",
    })


@app.route("/audio/<filename>")
def serve_audio(filename: str):
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "Audio not found"}), 404
    ext = audio_path.suffix.lstrip(".")
    mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
    return send_file(audio_path, mimetype=mime)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
