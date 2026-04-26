# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the server

```bash
MODEL_DIR=./models/results python3 -m uvicorn main:app --reload --port 8000
```

`MODEL_DIR` must point at the directory containing the TFLite model files and pickles. The server auto-creates all database tables and seeds a default admin on first start.

## Environment

`.env` must contain:
```
SUPA_CONNECTION_STRING=postgresql://...   # Supabase Postgres URL (standard postgres:// scheme)
SECRET_KEY=...                            # JWT signing key (defaults to dev-secret if missing)
ADMIN_USERNAME=...                        # Optional: override default admin credentials
ADMIN_PASSWORD=...
```

The database module rewrites the URL scheme to `postgresql+asyncpg://` automatically. SSL is always required (`connect_args={"ssl": "require"}`).

## Architecture

This is an **async FastAPI** backend for a call-centre emotion detection system. The main flow is:

1. An agent uploads a stereo WAV call recording (`POST /calls/upload`)
2. A **background task** splits the stereo file into left (agent) and right (customer) channels, segments each channel into 3-second chunks, runs ML inference on every chunk, saves `EmotionSegment` rows, then runs alert detection
3. Consumers poll `GET /calls/{id}/timeline` for the results

### ML inference pipeline

The model is a **3-model TFLite ensemble** (weights: MLP 10%, Dual 50%, Model3 40%):

| Model | File | Input |
|---|---|---|
| MLP | `tflite_mlp.tflite` | 624-dim feature vector (scaled by `scaler.pkl`) |
| Dual (CNN-MLP) | `emotion_detection_dual.tflite` | 624-dim features (`scaler_dual.pkl`) + (128,128,1) mel-spectrogram image |
| Model3 | `tflite_model3.tflite` | 624-dim feature vector (scaled by `scaler.pkl`) |

Feature extraction lives in `app/services/features.py` — `extract_features()` produces the 624-dim vector, `extract_melspec_image()` produces the CNN input. Both accept a raw `np.ndarray` audio chunk (not a file path). The feature composition (MFCCs, chroma, mel, tonnetz, F0, HPSS, ZCR, RMS, etc.) must stay in sync with the Kaggle training notebook at `models/results/.virtual_documents/__notebook_source__.ipynb`.

On Apple Silicon, `tflite-runtime` is unavailable — `tensorflow` is used as fallback via a try/except in `prediction.py`. The interpreters and scalers are loaded once at module import time (module-level globals), so the first request is slow but subsequent ones are fast.

### Auth

Single `/auth/login` endpoint checks admin table first, then agent table. Returns a JWT with `{"sub": "<uuid>", "role": "admin"|"agent"}`. Token lifetime is 24 hours. Two FastAPI dependency functions enforce access: `require_agent` and `require_admin` (in `app/core/deps.py`).

### Router layout

| Prefix | File | Auth |
|---|---|---|
| `/auth` | `app/api/auth.py` | none |
| `/calls` | `app/api/calls.py` | JWT (agent or admin) |
| `/me` | `app/api/agents.py` | agent only |
| `/admin` | `app/api/admin.py` | admin only |
| `/demo` | `app/api/demo.py` | none |

### Static pages

`static/` contains plain HTML+JS frontends served at `/static/*`:
- `demo.html` — live mic emotion demo (captures raw PCM via `ScriptProcessorNode`, encodes WAV in JS, posts to `/demo/predict`)
- `agent.html` — agent-facing UI
- `admin.html` — admin dashboard

### Alert detection

`app/services/alerts.py` runs after a call finishes. Two alert types:
- `anger_sustained` — customer angry with confidence > 0.5 for 40+ consecutive 3-second segments (≥ 2 minutes)
- `escalation` — pattern `neutral → sad → angry` in consecutive segments

### Database models

`Admin`, `Agent`, `Call`, `EmotionSegment`, `Alert` (all in `app/db/models.py`). Tables are created via SQLAlchemy `create_all` at startup — there are no Alembic migration files currently in use. Audio files are saved to `./uploads/` during processing and deleted afterwards.

## Key constraints

- Audio convention: **left channel = agent, right channel = customer**. Mono files duplicate both channels.
- All audio is resampled to **22050 Hz**, segmented into **3-second chunks**.
- The demo endpoint (`/demo/predict`) sends raw WAV — do not change it to WebM/Opus as `soundfile` cannot decode compressed formats without ffmpeg.
- `tflite-runtime` is commented out in `requirements.txt`; `tensorflow` is the active dependency.
