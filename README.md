# Voice Clone Tools

Cloned-voice TTS (XTTS v2) and video narration scripts for Hermes Agent on a mini PC.

## Setup
- Python 3.11 via python-build-standalone at `~/.python311`
- `pip install TTS transformers==4.39.3 torchcodec`
- Set `COQUI_TOS_AGREED=1`

## Scripts
- `clone_tts.py` — synthesize speech in a cloned voice; outputs WAV+MP3; optional `--cast` to Nest Mini
- `narrate_video.py` — mux cloned-voice narration into an MP4
- `test_clone.py` — smoke test

## Usage
```bash
~/.python311/python/bin/python3.11 clone_tts.py --text "Hello" --ref ref.wav --out clip --cast
~/.python311/python/bin/python3.11 narrate_video.py --video in.mp4 --text "narration" --ref ref.wav --out out
```
