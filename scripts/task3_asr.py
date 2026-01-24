from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import torch
import nemo.collections.asr as nemo_asr


# FastAPI app

app = FastAPI(
    title="Hindi ASR Service",
    description="ASR using AI4Bharat IndicConformer (NeMo)",
    version="1.0"
)


# Load IndicConformer model
MODEL_PATH ="RAG-voice-chatbot/models/indicconformer_stt_hi_hybrid_rnnt_large.nemo"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    restore_path=MODEL_PATH,
    map_location=DEVICE
)
model.eval()


# Health check
@app.get("/")
def health():
    return {"status": "IndicConformer ASR service running"}


# Transcription endpoint
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        with torch.no_grad():
            transcription = model.transcribe(
                audio=[audio_path],
                language_id="hi"   # Hindi
            )

        return {
            "language": "hi",
            "transcription": transcription[0]
        }

    finally:
        os.remove(audio_path)
