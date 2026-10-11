from flask import Flask, request, jsonify, send_file, render_template
from vosk import Model, KaldiRecognizer
from pathlib import Path
import subprocess
import uuid
import json
import wave
import re

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
AUDIO.mkdir(exist_ok=True)

# ============================================================
# VOSK - OFFLINE SPEECH RECOGNITION
# ============================================================

VOSK_MODEL = ROOT / "models" / "stt" / "vosk-model-small-en-us-0.15"

print("Loading speech recognition model...")
vosk_model = Model(str(VOSK_MODEL))
print("Speech recognition ready!")


# ============================================================
# PIPER - OFFLINE TEXT TO SPEECH
# ============================================================

PIPER_MODEL = next(
    (p for p in (ROOT / "models" / "tts").glob("*.onnx")),
    None
)

if PIPER_MODEL is None:
    raise FileNotFoundError("Piper voice model not found.")

PIPER_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


# ============================================================
# FFMPEG
# ============================================================

FFMPEG_EXE = Path(
    r"C:\Users\LATHIKA.K\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__, template_folder="web")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/audio/<filename>")
def audio(filename):
    return send_file(
        AUDIO / filename,
        mimetype="audio/wav"
    )


# ============================================================
# PROCESS VOICE INPUT
# ============================================================

@app.route("/process", methods=["POST"])
def process():

    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    job = str(uuid.uuid4())

    webm = AUDIO / f"{job}.webm"
    wav = AUDIO / f"{job}.wav"
    response_wav = AUDIO / f"{job}_response.wav"

    request.files["audio"].save(webm)

    # --------------------------------------------------------
    # WEBM → WAV
    # --------------------------------------------------------

    subprocess.run(
        [
            str(FFMPEG_EXE),
            "-y",
            "-i",
            str(webm),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav)
        ],
        check=True
    )

    # --------------------------------------------------------
    # SPEECH → TEXT
    # --------------------------------------------------------

    wf = wave.open(str(wav), "rb")

    recognizer = KaldiRecognizer(
        vosk_model,
        wf.getframerate()
    )

    while True:

        data = wf.readframes(4000)

        if len(data) == 0:
            break

        recognizer.AcceptWaveform(data)

    result = json.loads(recognizer.FinalResult())

    text = result.get("text", "").strip()

    wf.close()

    if not text:
        return jsonify({
            "error": "I couldn't understand the audio."
        })

    print("YOU:", text)

    # --------------------------------------------------------
    # LOCAL MEDICAL AI
    # --------------------------------------------------------

    medical_prompt = f"""
You are an offline medical education assistant for a
BTech Health Science and Technology student.

Your specialization includes:
- Anatomy
- Physiology
- Pathology
- Medical diagnostics
- Biomedical instrumentation
- Medical imaging
- Healthcare technology
- Biomedical engineering
- Clinical terminology

Rules:
1. Answer the user's question directly.
2. Keep the answer concise.
3. Use correct medical terminology.
4. Do not give unrelated information.
5. Do not repeat the question.
6. For simple questions, answer in 2 to 4 sentences.
7. For technical questions, use short bullet points.
8. If the question is unrelated to medicine, healthcare,
   or biomedical technology, say:
   "I am specialized in medical and biomedical questions."
9. Never invent medical facts.
10. This is for educational purposes, not personal diagnosis.

User question:
{text}
"""

    result = subprocess.run(
        [
            "ollama",
            "run",
            "qwen2.5:0.5b",
            medical_prompt
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    answer = result.stdout.strip()

    # --------------------------------------------------------
    # CLEAN OLLAMA TERMINAL CONTROL CHARACTERS
    # --------------------------------------------------------

    answer = re.sub(
        r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])',
        '',
        answer
    )

    answer = answer.strip()

    if not answer:
        answer = "Sorry, I could not generate an answer."

    print("AI:", answer)

    # --------------------------------------------------------
    # TEXT → SPEECH
    # --------------------------------------------------------

    subprocess.run(
        [
            str(PIPER_PYTHON),
            "-m",
            "piper",
            "--model",
            str(PIPER_MODEL),
            "--output_file",
            str(response_wav)
        ],
        input=answer,
        text=True,
        encoding="utf-8",
        check=True
    )

    # --------------------------------------------------------
    # SEND RESULT TO BROWSER
    # --------------------------------------------------------

    return jsonify({
        "text": text,
        "answer": answer,
        "audio_url": f"/audio/{response_wav.name}"
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 55)
    print("       EDGE AI MEDICAL VOICE ASSISTANT")
    print("=" * 55)
    print("")
    print("Open Chrome and go to:")
    print("https://localhost:5000")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        ssl_context="adhoc",
        debug=False
    )