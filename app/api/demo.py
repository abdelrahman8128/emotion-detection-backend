"""
Demo endpoint — accepts a raw audio chunk and returns an emotion prediction.
No auth required; for local testing only.
"""
import io
import numpy as np
import soundfile as sf
import librosa

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.prediction import predict_emotion

router = APIRouter(prefix="/demo", tags=["demo"])

SR = 22050


@router.post("/predict")
async def predict_chunk(audio: UploadFile = File(...)):
    """
    Accepts a WAV/WebM audio blob, resamples to 22050 Hz mono,
    and returns the predicted emotion + confidence.
    """
    try:
        data = await audio.read()
        buf = io.BytesIO(data)
        samples, orig_sr = sf.read(buf, always_2d=False)

        # Convert to mono float32
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)

        # Resample to model SR
        if orig_sr != SR:
            samples = librosa.resample(samples, orig_sr=orig_sr, target_sr=SR)

        if len(samples) == 0:
            raise HTTPException(status_code=400, detail="Empty audio")

        emotion, confidence = predict_emotion(samples, SR)
        return {"emotion": emotion, "confidence": confidence}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
