# Voice Storage Directory (`storage/voices`)

Place your reference audio files for voice cloning in this folder.

## Naming Format

For each voice, supply:
1. **Audio File (Required)**: e.g. `voice_fi.wav` (3–15 seconds of clear reference speech audio). Supported formats: `.wav`, `.flac`, `.mp3`, `.ogg`.
2. **Transcript File (Required)**: e.g. `voice_fi.txt` containing the exact spoken text of `voice_fi.wav`.
   - *Note*: Whisper ASR is disabled. The server reads transcriptions directly from `.txt` files for faster and deterministic voice cloning initialization.
3. **Voice Config File (Optional)**: e.g. `voice_fi.json` or `voice_fi.yaml` to specify default generation settings for this specific voice.

## Example File Layout
```
storage/voices/
├── voice_fi.wav      # Audio file
├── voice_fi.txt      # Text transcript of voice_fi.wav
├── voice_fi.json     # Optional custom settings for voice_fi
├── voice_en.wav      # Audio file
├── voice_en.txt      # Text transcript of voice_en.wav
└── voice_en.yaml     # Optional settings for voice_en
```

## Per-Voice Settings Format (`.json` or `.yaml`)

```json
{
  "language": "fi",
  "speed": 1.0,
  "num_step": 32,
  "guidance_scale": 2.0,
  "description": "Finnish reference voice"
}
```
