import torch
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

import wave, numpy as np
import TTS.tts.models.xtts as _xtts
def _load_audio(audiopath, sampling_rate):
    with wave.open(audiopath, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != sampling_rate:
        # linear resample
        new_n = int(len(data) * sampling_rate / sr)
        data = np.interp(np.linspace(0, len(data)-1, new_n), np.arange(len(data)), data).astype(np.float32)
    audio = torch.from_numpy(data).unsqueeze(0)
    audio = audio.clamp(-1.0, 1.0)
    return audio
_xtts.load_audio = _load_audio

from TTS.api import TTS
import time
t0=time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print("model loaded in", round(time.time()-t0,1), "s")
t0=time.time()
tts.tts_to_file(
    text="Hello, this is a cloned voice test from Hermes.",
    speaker_wav="/home/kb/.hermes/cache/scratch/voicemail-6323.wav",
    language="en",
    file_path="/home/kb/.hermes/cache/scratch/cloned_test.wav",
)
print("synth done in", round(time.time()-t0,1), "s")
