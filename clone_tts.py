#!/usr/bin/env /home/kb/.python311/python/bin/python3.11
"""Clone TTS: synthesize speech in a cloned voice using XTTS v2.
Usage:
  clone_tts.py --text "Hello" --ref /path/to/ref.wav [--out name] [--lang en] [--cast]
Outputs WAV + MP3 to scratch dir and (with --cast) plays on Kitchen Speaker.
"""
import argparse, os, subprocess, sys, time

PY311 = "/home/kb/.python311/python/bin/python3.11"
FFMPEG = "/home/kb/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
SCRATCH = "/home/kb/.hermes/cache/scratch"
HOST = "192.168.0.77"
PORT = 8799

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="nest_clip")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--gain", type=float, default=2.5, help="audio volume multiplier (default 2.5)")
    ap.add_argument("--cast", action="store_true")
    args = ap.parse_args()

    import torch
    _orig = torch.load
    def _pl(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    torch.load = _pl

    import wave, numpy as np
    import TTS.tts.models.xtts as _xtts
    def _load_audio(audiopath, sampling_rate):
        with wave.open(audiopath, "rb") as wf:
            sr = wf.getframerate(); n = wf.getnframes()
            raw = wf.readframes(n); ch = wf.getnchannels(); sw = wf.getsampwidth()
        if sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        if sr != sampling_rate:
            new_n = int(len(data) * sampling_rate / sr)
            data = np.interp(np.linspace(0, len(data)-1, new_n), np.arange(len(data)), data).astype(np.float32)
        audio = torch.from_numpy(data).unsqueeze(0).clamp(-1.0, 1.0)
        return audio
    _xtts.load_audio = _load_audio

    from TTS.api import TTS
    os.environ["COQUI_TOS_AGREED"] = "1"
    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    wav_path = os.path.join(SCRATCH, args.out + ".wav")
    t0 = time.time()
    tts.tts_to_file(text=args.text, speaker_wav=args.ref, language=args.lang, file_path=wav_path)
    print(f"synth done in {time.time()-t0:.1f}s -> {wav_path}", flush=True)

    mp3_path = os.path.join(SCRATCH, args.out + ".mp3")
    subprocess.run([FFMPEG, "-y", "-i", wav_path, "-af", f"volume={args.gain}", "-codec:a", "libmp3lame", "-q:a", "2", mp3_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"mp3 -> {mp3_path}", flush=True)
    url = f"http://{HOST}:{PORT}/{args.out}.mp3"
    print(f"URL: {url}", flush=True)

    if args.cast:
        cast_script = f'''
import pychromecast, time
chromecasts, browser = pychromecast.get_chromecasts(timeout=10)
cast = next((c for c in chromecasts if c.name == "Kitchen Speaker"), None)
if not cast:
    print("Kitchen Speaker not found"); raise SystemExit(1)
cast.wait()
try:
    cast.quit_app(); time.sleep(2)
except Exception as e:
    print("quit err", e)
cast.set_volume(0.4)
mc = cast.media_controller
mc.play_media("{url}", "audio/mpeg", title="Cloned TTS")
for _ in range(8):
    time.sleep(1)
    print("state:", mc.status.player_state)
    if mc.status.player_state == "PLAYING":
        break
mc.play()
time.sleep(2)
print("final state:", mc.status.player_state)
pychromecast.discovery.stop_discovery(browser)
'''
        subprocess.run(["/usr/bin/python3", "-c", cast_script])
    return 0

if __name__ == "__main__":
    sys.exit(main())