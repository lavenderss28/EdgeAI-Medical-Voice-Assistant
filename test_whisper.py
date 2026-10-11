from faster_whisper import WhisperModel

print("Loading Whisper model...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

print("Whisper model loaded successfully!")
