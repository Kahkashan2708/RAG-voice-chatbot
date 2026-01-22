from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(
    title="Hindi ASR Service",
    description="ASR using Faster-Whisper",
    version="1.0"
)

# Load model
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

@app.get("/")
def health():
    return {"status": "ASR service running"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        segments, info = model.transcribe(
            audio_path,
            language="hi",
            beam_size=5
        )

        text = " ".join(seg.text for seg in segments)

        return {
            "language": info.language,
            "confidence": info.language_probability,
            "transcription": text.strip()
        }

    finally:
        os.remove(audio_path)
