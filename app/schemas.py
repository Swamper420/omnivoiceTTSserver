from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text content to synthesize into speech", example="Tervehdys! Tämä on ääni-synteesi testi.")
    voice: str = Field("voice_fi", description="Voice ID or name stored in storage/voices", example="voice_fi")
    language: Optional[str] = Field(None, description="Target language code (e.g. 'fi', 'en', 'zh'). Uses voice setting or default if omitted.")
    speed: Optional[float] = Field(None, description="Speech speed factor (1.0 = normal speed)", example=1.0)
    num_step: Optional[int] = Field(None, description="Number of diffusion generation steps (e.g. 16-40)", example=32)
    guidance_scale: Optional[float] = Field(None, description="CFG guidance scale for prompt adherence", example=2.0)
    response_format: Optional[str] = Field("wav", description="Audio container format ('wav', 'mp3', 'ogg', 'flac')", example="wav")
    disable_chunking: Optional[bool] = Field(False, description="Bypass/disable text chunking and synthesize full text as one single segment", example=False)
    seed: Optional[int] = Field(None, description="Random seed for deterministic generation")

class OpenAITTSRequest(BaseModel):
    model: str = Field("omnivoice", description="Model name (e.g., 'tts-1', 'omnivoice')")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice name matching voice_*.wav in storage/voices")
    response_format: Optional[str] = Field("mp3", description="Audio format ('mp3', 'wav', 'ogg', 'flac')")
    speed: Optional[float] = Field(1.0, description="Playback speed")

class VoiceConfigOverride(BaseModel):
    language: Optional[str] = None
    speed: Optional[float] = None
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_params: Optional[Dict[str, Any]] = None

class VoiceInfo(BaseModel):
    voice_id: str
    audio_path: str
    has_transcript: bool
    transcript: str
    settings: Dict[str, Any]

class VoiceListResponse(BaseModel):
    count: int
    voices: List[VoiceInfo]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    whisper_loaded: bool = False
    cuda_available: bool
    voices_count: int
