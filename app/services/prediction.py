"""
Emotion prediction service — 3-model TFLite ensemble.

Models (ensemble weights: mlp=0.1, dual=0.5, model3=0.4):
  - MLP        : tflite_mlp.tflite       (624-dim features)
  - Dual       : emotion_detection_dual.tflite  (624-dim features + 128×128 mel image)
  - Model3     : tflite_model3.tflite    (624-dim features)

Set MODEL_DIR env var to point at the folder containing these files.
Defaults to ./models/results (relative to the project root).
"""
import os
import numpy as np
import joblib

from app.services.features import extract_features, extract_melspec_image

MODEL_DIR = os.getenv("MODEL_DIR", "./models/results")

# ── Load scalers & label encoder ─────────────────────────────────────────────
_scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
_scaler_dual = joblib.load(os.path.join(MODEL_DIR, "scaler_dual.pkl"))
_label_enc   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ── Ensemble hyper-params (from inference_config.json) ────────────────────────
_W_MLP    = 0.1
_W_DUAL   = 0.5
_W_MODEL3 = 0.4

_T_MLP    = 0.7
_T_DUAL   = 0.7
_T_MODEL3 = 0.8

# ── Load TFLite interpreters ──────────────────────────────────────────────────
def _load_interpreter(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=path)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=path)
        except ImportError:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


_mlp_interp    = _load_interpreter("tflite_mlp.tflite")
_dual_interp   = _load_interpreter("emotion_detection_dual.tflite")
_model3_interp = _load_interpreter("tflite_model3.tflite")

_mlp_in    = _mlp_interp.get_input_details()[0]["index"]
_mlp_out   = _mlp_interp.get_output_details()[0]["index"]

_model3_in  = _model3_interp.get_input_details()[0]["index"]
_model3_out = _model3_interp.get_output_details()[0]["index"]

# Dual model has two inputs: find which is features (rank-2) vs image (rank-4)
_dual_inputs = _dual_interp.get_input_details()
_dual_feat_idx = next(d["index"] for d in _dual_inputs if len(d["shape"]) == 2)
_dual_img_idx  = next(d["index"] for d in _dual_inputs if len(d["shape"]) == 4)
_dual_out      = _dual_interp.get_output_details()[0]["index"]


# ── Softmax with temperature ──────────────────────────────────────────────────
def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = logits / temperature
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ── Public API ────────────────────────────────────────────────────────────────
def predict_emotion(audio_chunk: np.ndarray, sr: int = 22050) -> tuple[str, float]:
    """
    Run the 3-model ensemble on one 3-second audio chunk.
    Returns (emotion_label, confidence).
    """
    raw_feats = extract_features(audio_chunk, sr)

    feats_mlp    = _scaler.transform(raw_feats.reshape(1, -1)).astype(np.float32)
    feats_dual   = _scaler_dual.transform(raw_feats.reshape(1, -1)).astype(np.float32)
    mel_img      = extract_melspec_image(audio_chunk, sr)[np.newaxis]  # (1,128,128,1)

    # MLP
    _mlp_interp.set_tensor(_mlp_in, feats_mlp)
    _mlp_interp.invoke()
    p_mlp = _softmax(_mlp_interp.get_tensor(_mlp_out)[0], _T_MLP)

    # Dual (CNN-MLP)
    _dual_interp.set_tensor(_dual_feat_idx, feats_dual)
    _dual_interp.set_tensor(_dual_img_idx, mel_img)
    _dual_interp.invoke()
    p_dual = _softmax(_dual_interp.get_tensor(_dual_out)[0], _T_DUAL)

    # Model3
    _model3_interp.set_tensor(_model3_in, feats_mlp)
    _model3_interp.invoke()
    p_model3 = _softmax(_model3_interp.get_tensor(_model3_out)[0], _T_MODEL3)

    p_ensemble = _W_MLP * p_mlp + _W_DUAL * p_dual + _W_MODEL3 * p_model3

    class_idx  = int(np.argmax(p_ensemble))
    emotion    = _label_enc.classes_[class_idx]
    confidence = round(float(p_ensemble[class_idx]), 4)

    return emotion, confidence
