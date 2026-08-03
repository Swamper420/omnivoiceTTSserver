import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import torch

from app.config import config
from app.schemas import (
    TTSRequest,
    OpenAITTSRequest,
    VoiceListResponse,
    HealthResponse,
)
from app.voice_manager import voice_manager
from app.model_manager import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("omnivoice_server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing OmniVoice TTS API Server...")
    logger.info(f"Voices directory: {config.voices_dir}")
    
    # Reload voices from storage/voices
    voice_manager.reload_voices()
    
    # Preload model in background or on startup if configured
    try:
        model_manager.load_model()
    except Exception as e:
        logger.error(f"Initial model loading failed: {e}. Model will attempt loading on first request.")
        
    yield
    logger.info("Shutting down OmniVoice TTS Server...")

app = FastAPI(
    title="OmniVoice TTS Server",
    description="High-performance zero-shot voice cloning TTS API powered by OmniVoice",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        device=config.device,
        whisper_loaded=False,  # Explicitly verification that Whisper is never loaded
        cuda_available=torch.cuda.is_available(),
        voices_count=len(voice_manager.voices)
    )

@app.get("/api/v1/voices", response_model=VoiceListResponse, tags=["Voices"])
@app.get("/v1/voices", response_model=VoiceListResponse, tags=["Voices"])
async def list_voices():
    voices_data = voice_manager.list_voices()
    return VoiceListResponse(
        count=len(voices_data),
        voices=voices_data
    )

@app.post("/api/v1/voices/reload", tags=["Voices"])
async def reload_voices():
    voices = voice_manager.reload_voices()
    model_manager.prompt_cache.clear()
    return {
        "status": "success",
        "message": f"Reloaded {len(voices)} voice(s) successfully.",
        "voices": list(voices.keys())
    }

@app.post("/api/v1/tts", tags=["TTS"])
@app.post("/synthesize", tags=["TTS"])
async def synthesize_speech(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")

    voice_meta = voice_manager.get_voice(request.voice)
    if not voice_meta:
        available = list(voice_manager.voices.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{request.voice}' not found in storage/voices. Available voices: {available}"
        )

    try:
        audio_bytes, mime_type = await model_manager.synthesize(
            voice_meta=voice_meta,
            text=request.text,
            language=request.language,
            speed=request.speed,
            num_step=request.num_step,
            guidance_scale=request.guidance_scale,
            response_format=request.response_format or "wav",
            seed=request.seed
        )
        return Response(content=audio_bytes, media_type=mime_type)
    except Exception as e:
        logger.error(f"Speech synthesis error for text '{request.text[:30]}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech(request: OpenAITTSRequest):
    """
    OpenAI TTS API compatible endpoint (POST /v1/audio/speech).
    """
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    voice_meta = voice_manager.get_voice(request.voice)
    if not voice_meta:
        # If voice requested was generic e.g. "alloy", fallback to first available or error
        available = list(voice_manager.voices.keys())
        if available:
            voice_meta = voice_manager.voices[available[0]]
            logger.warning(f"Voice '{request.voice}' not found. Falling back to default voice '{voice_meta.voice_id}'.")
        else:
            raise HTTPException(status_code=404, detail=f"No voices available in storage/voices.")

    try:
        audio_bytes, mime_type = await model_manager.synthesize(
            voice_meta=voice_meta,
            text=request.input,
            speed=request.speed,
            response_format=request.response_format or "mp3"
        )
        return Response(content=audio_bytes, media_type=mime_type)
    except Exception as e:
        logger.error(f"OpenAI TTS endpoint synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=False)
