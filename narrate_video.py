#!/usr/bin/env /home/kb/.python311/python/bin/python3.11
"""Narrate a video with a cloned voice.
Usage:
  narrate_video.py --video in.mp4 --text "narration" --ref ref.wav [--out name] [--lang en]
Outputs final MP4 with cloned-voice audio track, served on port 8799.
"""
import argparse, os, subprocess, sys, time

PY311 = "/home/kb/.python311/python/bin/python3.11"
FFMPEG = "/home/kb/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
SCRATCH = "/home/kb/.hermes/cache/scratch"
HOST = "192.168.0.77"
PORT = 8799

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="narrated")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--gain", type=float, default=2.5, help="audio volume multiplier (default 2.5)")
    args = ap.parse_args()

    # 1. synthesize cloned voice
    synth_script = f'''
import torch, wave, numpy as np, os, time
_orig = torch.load
def _pl(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)
torch.load = _pl
import TTS.tts.models.xtts as _xtts
def _load_audio(audiopath, sampling_rate):
    with wave.open(audiopath, "rb") as wf:
        sr=wf.getframerate(); n=wf.getnframes(); raw=wf.readframes(n)
        ch=wf.getnchannels(); sw=wf.getsampwidth()
    if sw==2: data=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    elif sw==4: data=np.frombuffer(raw,dtype=np.int32).astype(np.float32)/2147483648.0
    else: data=np.frombuffer(raw,dtype=np.uint8).astype(np.float32)/128.0-1.0
    if ch>1: data=data.reshape(-1,ch).mean(axis=1)
    if sr!=sampling_rate:
        new_n=int(len(data)*sampling_rate/sr)
        data=np.interp(np.linspace(0,len(data)-1,new_n),np.arange(len(data)),data).astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0).clamp(-1.0,1.0)
_xtts.load_audio=_load_audio
from TTS.api import TTS
os.environ["COQUI_TOS_AGREED"]="1"
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
wav="{SCRATCH}/{args.out}.wav"
tts.tts_to_file(text={args.text!r}, speaker_wav={args.ref!r}, language={args.lang!r}, file_path=wav)
print("WAV_OK", wav)
'''
    r = subprocess.run([PY311, "-c", synth_script], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("SYNTH FAILED:", r.stderr[-500:]); sys.exit(1)
    wav_path = os.path.join(SCRATCH, args.out + ".wav")

    # 2. mux audio into video
    out_mp4 = os.path.join(SCRATCH, args.out + ".mp4")
    cmd = [FFMPEG, "-y", "-i", args.video, "-i", wav_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-af", f"volume={args.gain}", out_mp4]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"FINAL: {out_mp4}")
    print(f"URL: http://{HOST}:{PORT}/{args.out}.mp4")
    return 0

if __name__ == "__main__":
    sys.exit(main())