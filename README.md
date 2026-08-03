# OmniVoice TTS API Server

High-performance zero-shot voice cloning Text-to-Speech (TTS) API server powered by **OmniVoice** and **FastAPI**, with support for custom per-voice configurations and explicit reference transcriptions.

## Features

- **Zero-Shot Voice Cloning**: Clone any voice using a short reference audio file (3–15 seconds).
- **No Whisper Overhead**: Reads transcription text directly from accompanying `.txt` files (e.g. `voice_fi.txt`), ensuring fast, deterministic voice prompt loading without initializing Whisper ASR.
- **Easy Per-Voice Configuration**: Customize voice settings (`language`, `speed`, `num_step`, `guidance_scale`) via optional `.json` or `.yaml` files per voice.
- **CUDA Accelerated**: Built for PyTorch with CUDA support (`load_asr=False`, `device=cuda`).
- **OpenAI Compatible**: Supports standard `POST /v1/audio/speech` endpoint in addition to custom REST endpoints.
- **Prompt Pre-computation & Caching**: Caches voice clone embeddings in memory for low-latency synthesis.

---

## Directory Structure

```
omnivoiceTTSserver/
├── app/
│   ├── config.py           # Configuration loader (config.yaml & env vars)
│   ├── main.py             # FastAPI server & route handlers
│   ├── model_manager.py    # OmniVoice loader & inference engine
│   ├── schemas.py          # Pydantic request/response models
│   └── voice_manager.py    # Voice catalog, transcript matcher & settings manager
├── storage/
│   └── voices/             # Reference audio, transcript & config storage
│       ├── voice_fi.wav    # Audio reference
│       ├── voice_fi.txt    # Audio transcription (Whisper bypassed)
│       └── voice_fi.json   # Per-voice custom configuration settings
├── config.yaml             # Global server settings
├── Dockerfile              # Container definition for GPU deployment
├── requirements.txt        # Python dependencies
├── test_client.py          # API test client
└── README.md
```

---

## Voice Setup & Configuration

Voices are placed in `storage/voices/`. Each voice consists of:

1. **Audio File (`<voice_id>.wav`)**: Reference audio file (e.g., `voice_fi.wav`). Supported formats: `.wav`, `.flac`, `.mp3`, `.ogg`.
2. **Transcript File (`<voice_id>.txt`)**: Exact text matching the audio clip (e.g., `voice_fi.txt`).
3. **Optional Settings File (`<voice_id>.json` or `.yaml`)**: Custom settings per voice.

### Example Per-Voice Config (`storage/voices/voice_fi.json`)

```json
{
  "language": "fi",
  "speed": 1.0,
  "num_step": 32,
  "guidance_scale": 2.0,
  "description": "Finnish male speaker reference"
}
```

---

## Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Running the Server

```bash
python -m app.main
```

Or using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Testing in a Web Browser

Once the server is running, open your web browser and navigate to:

1. **OmniVoice Studio Web UI (`http://localhost:8000/`)**:
   - Built-in interactive Web Studio interface.
   - Select cloned voices from a dropdown, view reference transcripts and per-voice custom settings.
   - Type prompt text, adjust speed, diffusion steps (`num_step`), and guidance scale (`guidance_scale`).
   - Click **Generate Speech** to synthesize and listen to the audio directly in the built-in HTML5 player or download the file.

2. **Swagger Interactive API Documentation (`http://localhost:8000/docs`)**:
   - Interactive OpenAPI documentation interface.
   - Click on any endpoint (e.g. `POST /api/v1/tts` or `POST /v1/audio/speech`), click **Try it out**, fill in parameters, and execute requests directly in your browser.

### 1. Synthesize Speech (Custom API)
`POST /api/v1/tts` or `POST /synthesize`

**Request Body:**
```json
{
  "text": "Tervehdys! Tämä on ääni-synteesi testi.",
  "voice": "voice_fi",
  "language": "fi",
  "speed": 1.0,
  "num_step": 32,
  "guidance_scale": 2.0,
  "response_format": "wav"
}
```

**Response:** Binary audio stream (`audio/wav`, `audio/mpeg`, `audio/flac`, or `audio/ogg`).

---

### 2. OpenAI Compatible Endpoint
`POST /v1/audio/speech`

**Request Body:**
```json
{
  "model": "omnivoice",
  "input": "Hello world, this is zero-shot voice cloning.",
  "voice": "voice_fi",
  "response_format": "mp3",
  "speed": 1.0
}
```

---

### 3. List Voices
`GET /api/v1/voices`

**Response:**
```json
{
  "count": 1,
  "voices": [
    {
      "voice_id": "voice_fi",
      "audio_path": "storage/voices/voice_fi.wav",
      "has_transcript": true,
      "transcript": "Tämä on suomenkielinen ääninäyte...",
      "settings": {
        "language": "fi",
        "speed": 1.0,
        "num_step": 32,
        "guidance_scale": 2.0
      }
    }
  ]
}
```

---

### 4. Reload Voice Catalog
`POST /api/v1/voices/reload`

Rescans `storage/voices/` for newly added audio, transcript, or config files without restarting the server.

---

### 5. Health Check
`GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "whisper_loaded": false,
  "cuda_available": true,
  "voices_count": 1
}
```

---

## Running with Docker

```bash
docker build -t omnivoice-tts-server .
docker run --gpus all -p 8000:8000 -v $(pwd)/storage/voices:/app/storage/voices omnivoice-tts-server
```
