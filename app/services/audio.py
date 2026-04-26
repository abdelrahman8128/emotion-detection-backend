"""
Audio utilities: load a stereo WAV, split channels, resample, segment into
3-second chunks ready for emotion inference.
"""
import numpy as np
import soundfile as sf
import librosa

SR = 22050          # target sample rate for the ML models
CHUNK_DURATION = 3  # seconds per segment


def load_stereo(file_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load a stereo audio file and return (left_channel, right_channel, sample_rate).
    Convention: left = agent, right = customer.
    If mono is supplied both channels will be identical.
    """
    audio, original_sr = sf.read(file_path, always_2d=True)  # shape: (samples, channels)

    if audio.shape[1] >= 2:
        left = audio[:, 0].astype(np.float32)
        right = audio[:, 1].astype(np.float32)
    else:
        left = right = audio[:, 0].astype(np.float32)

    # Resample both channels to the model's expected SR
    if original_sr != SR:
        left = librosa.resample(left, orig_sr=original_sr, target_sr=SR)
        right = librosa.resample(right, orig_sr=original_sr, target_sr=SR)

    return left, right, SR


def segment_audio(channel: np.ndarray, sr: int = SR) -> list[np.ndarray]:
    """
    Split a 1-D audio array into fixed-length chunks of CHUNK_DURATION seconds.
    The last chunk is zero-padded if shorter than CHUNK_DURATION.
    Returns a list of numpy arrays, each of length (sr * CHUNK_DURATION).
    """
    chunk_len = sr * CHUNK_DURATION
    chunks = []

    for start in range(0, len(channel), chunk_len):
        chunk = channel[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)

    return chunks


def split_and_segment(file_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Full pipeline: load stereo file → split channels → segment each channel.
    Returns (agent_chunks, customer_chunks).
    """
    agent_channel, customer_channel, sr = load_stereo(file_path)
    agent_chunks = segment_audio(agent_channel, sr)
    customer_chunks = segment_audio(customer_channel, sr)
    return agent_chunks, customer_chunks
