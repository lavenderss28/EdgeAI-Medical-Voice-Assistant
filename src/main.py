import os
import sys
import time
import wave
import subprocess
import tempfile
import threading
import queue
import re
import json
from pathlib import Path

import numpy as np
import sounddevice as sd
import psutil

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

WHISPER_EXE = ROOT / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe"
WHISPER_MODEL = ROOT / "models" / "whisper" / "ggml-base.en.bin"

LLM_MODEL = ROOT / "models" / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

PIPER_MODEL = ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"

PYTHON = sys.executable

AUDIO_DIR = ROOT / "audio"
LOG_DIR = ROOT / "logs"

AUDIO_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1

# CPU ONLY
CPU_THREADS = max(2, min(8, os.cpu_count() or 4))

# Silence detection
SILENCE_THRESHOLD = 0.012
SILENCE_TIME = 1.0
MAX_RECORDING_TIME = 12

# ============================================================
# UTILITY
# ============================================================

def print_header():
    print()
    print("=" * 60)
    print("        EDGE AI OFFLINE VOICE ASSISTANT")
    print("=" * 60)
    print("ASR  : Whisper.cpp")
    print("LLM  : Qwen 0.5B Q4_K_M")
    print("TTS  : Piper")
    print("MODE : CPU ONLY / OFFLINE")
    print("=" * 60)
    print()


def resource_snapshot():

    process = psutil.Process(os.getpid())

    return {
        "time": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "process_ram_mb": process.memory_info().rss / (1024 * 1024)
    }


# ============================================================
# AUDIO RECORDING
# ============================================================

def record_until_silence():

    print("🎤 Listening...")

    audio_chunks = []

    silence_start = None
    started = False

    start_time = time.perf_counter()

    def callback(indata, frames, time_info, status):

        if status:
            print(status)

        audio_chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=1024
    ):

        while True:

            time.sleep(0.05)

            if not audio_chunks:
                continue

            current = audio_chunks[-1]

            volume = float(np.sqrt(np.mean(current ** 2)))

            if volume > SILENCE_THRESHOLD:

                started = True
                silence_start = None

            elif started:

                if silence_start is None:
                    silence_start = time.perf_counter()

                elif time.perf_counter() - silence_start > SILENCE_TIME:
                    break

            if time.perf_counter() - start_time > MAX_RECORDING_TIME:
                break

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0)

    # Convert float32 -> int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    filename = AUDIO_DIR / f"input_{int(time.time())}.wav"

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    return filename


# ============================================================
# WHISPER ASR
# ============================================================

def transcribe(audio_file):

    print("📝 Transcribing...")

    start = time.perf_counter()

    cmd = [
        str(WHISPER_EXE),
        "-m", str(WHISPER_MODEL),
        "-f", str(audio_file),
        "-t", str(CPU_THREADS),
        "-nt",
        "-np",
        "-l", "en"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    elapsed = time.perf_counter() - start

    output = result.stdout.strip()

    # Remove common whisper logging
    lines = []

    for line in output.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("["):
            continue

        if line.startswith("whisper_"):
            continue

        lines.append(line)

    text = " ".join(lines).strip()

    print(f"ASR latency: {elapsed:.2f}s")
    print(f"USER: {text}")

    return text, elapsed


# ============================================================
# LOCAL LLM
# ============================================================

def generate_response(user_text):

    print("🧠 Thinking...")

    start = time.perf_counter()

    prompt = (
        "You are a helpful offline voice assistant. "
        "Answer clearly and briefly. "
        "Do not mention that you are an AI unless asked. "
        "Keep answers suitable for spoken conversation.\n\n"
        f"User: {user_text}\n"
        "Assistant:"
    )

    cmd = [
        "llama-cli",
        "-m", str(LLM_MODEL),

        # CPU ONLY
        "-ngl", "0",

        # CPU threads
        "-t", str(CPU_THREADS),

        # Context
        "-c", "2048",

        # Generation
        "-n", "160",

        # Temperature
        "--temp", "0.7",

        # Prompt
        "-p", prompt,

        # No interactive mode
        "--no-cnv"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    elapsed = time.perf_counter() - start

    response = result.stdout.strip()

    # Try to remove llama.cpp diagnostics
    lines = response.splitlines()

    cleaned = []

    for line in lines:

        if not line.strip():
            continue

        if line.startswith("llama_"):
            continue

        if "load_tensors" in line:
            continue

        if "system_info" in line:
            continue

        if "sampler" in line:
            continue

        cleaned.append(line)

    response = " ".join(cleaned).strip()

    # Remove accidental prompt echo
    if "Assistant:" in response:
        response = response.split("Assistant:", 1)[-1].strip()

    print(f"LLM latency: {elapsed:.2f}s")
    print(f"ASSISTANT: {response}")

    return response, elapsed


# ============================================================
# PIPER TTS
# ============================================================

def speak(text):

    print("🔊 Speaking...")

    start = time.perf_counter()

    output_file = AUDIO_DIR / f"response_{int(time.time())}.wav"

    # Piper CLI
    cmd = [
        PYTHON,
        "-m",
        "piper",
        "-m",
        str(PIPER_MODEL),
        "-f",
        str(output_file),
        "--",
        text
    ]

    subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    synthesis_time = time.perf_counter() - start

    # Windows audio playback through PowerShell
    ps = (
        "$player = New-Object System.Media.SoundPlayer;"
        f"$player.SoundLocation = '{output_file}';"
        "$player.PlaySync();"
    )

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True
    )

    print(f"TTS latency: {synthesis_time:.2f}s")

    return synthesis_time


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print_header()

    print("CPU threads:", CPU_THREADS)
    print("CPU:", psutil.cpu_count(logical=True))
    print("RAM:", round(psutil.virtual_memory().total / (1024**3), 2), "GB")

    print()
    print("Everything is local.")
    print("Press Ctrl+C to exit.")
    print()

    while True:

        try:

            before = resource_snapshot()

            # -----------------------------------------------
            # 1. RECORD
            # -----------------------------------------------

            audio = record_until_silence()

            if audio is None:
                continue

            # -----------------------------------------------
            # 2. ASR
            # -----------------------------------------------

            text, asr_time = transcribe(audio)

            if not text:
                print("❌ No speech detected.")
                continue

            # -----------------------------------------------
            # 3. LLM
            # -----------------------------------------------

            response, llm_time = generate_response(text)

            if not response:
                print("❌ LLM produced no response.")
                continue

            # -----------------------------------------------
            # 4. TTS
            # -----------------------------------------------

            tts_time = speak(response)

            # -----------------------------------------------
            # METRICS
            # -----------------------------------------------

            after = resource_snapshot()

            total_time = asr_time + llm_time + tts_time

            print()
            print("-" * 60)
            print("BENCHMARK")
            print("-" * 60)
            print(f"ASR time       : {asr_time:.2f} sec")
            print(f"LLM time       : {llm_time:.2f} sec")
            print(f"TTS time       : {tts_time:.2f} sec")
            print(f"Pipeline time  : {total_time:.2f} sec")
            print(f"CPU usage      : {after['cpu_percent']:.1f}%")
            print(f"RAM usage      : {after['ram_percent']:.1f}%")
            print(
                f"Process RAM    : "
                f"{after['process_ram_mb']:.1f} MB"
            )
            print("-" * 60)
            print()

            # Save benchmark
            log_file = LOG_DIR / "benchmark.jsonl"

            entry = {
                "timestamp": time.time(),
                "asr_seconds": asr_time,
                "llm_seconds": llm_time,
                "tts_seconds": tts_time,
                "total_seconds": total_time,
                "cpu_percent": after["cpu_percent"],
                "ram_percent": after["ram_percent"],
                "process_ram_mb": after["process_ram_mb"]
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        except KeyboardInterrupt:

            print()
            print("Stopping Edge AI assistant...")
            break

        except Exception as e:

            print()
            print("ERROR:", e)
            print()
            print("Continuing...")
            time.sleep(1)


if __name__ == "__main__":
    main()
