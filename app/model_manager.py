import asyncio
import concurrent.futures
import copy
import gc
import io
import logging
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import soundfile as sf
import torch

from app.config import config
from app.voice_manager import VoiceMetadata

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.prompt_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        # Single persistent thread for all PyTorch/CUDA execution to guarantee CUDA context thread safety
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omnivoice_model"
        )

    def get_torch_dtype(self) -> torch.dtype:
        dt = config.dtype.lower()
        if dt == "float16" or dt == "fp16":
            return torch.float16
        elif dt == "bfloat16" or dt == "bf16":
            return torch.bfloat16
        return torch.float32

    def load_model(self) -> None:
        if self.is_loaded and self.model is not None:
            return

        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested in config but PyTorch reports CUDA is unavailable. Falling back to CPU.")
            device = "cpu"

        dtype = self.get_torch_dtype()
        logger.info(f"Loading OmniVoice model '{config.model_name_or_path}' on device '{device}' with dtype '{dtype}'...")
        logger.info("Explicitly disabling Whisper ASR load (load_asr=False).")

        try:
            from omnivoice import OmniVoice
            
            # CRITICAL: Always set load_asr=False so Whisper is NEVER loaded into memory!
            self.model = OmniVoice.from_pretrained(
                config.model_name_or_path,
                device_map=device,
                dtype=dtype,
                load_asr=False
            )
            self.is_loaded = True
            logger.info("OmniVoice model successfully loaded!")
        except Exception as e:
            logger.error(f"Failed to load OmniVoice model: {e}", exc_info=True)
            self.model = None
            self.is_loaded = False
            raise RuntimeError(f"OmniVoice model loading error: {e}")

    async def get_or_create_prompt(self, voice_meta: VoiceMetadata) -> Optional[Any]:
        """
        Creates or retrieves cached VoiceClonePrompt if supported by OmniVoice,
        otherwise returns None so direct ref_audio and ref_text are passed to generate.
        """
        voice_id = voice_meta.voice_id
        if voice_id in self.prompt_cache:
            return self.prompt_cache[voice_id]

        if not self.model:
            return None

        # Check if model supports create_voice_clone_prompt
        if hasattr(self.model, "create_voice_clone_prompt") and voice_meta.transcript:
            try:
                logger.info(f"Pre-computing voice clone prompt for voice '{voice_id}'...")
                loop = asyncio.get_running_loop()
                def _create_prompt():
                    with torch.inference_mode():
                        return self.model.create_voice_clone_prompt(
                            ref_audio=str(voice_meta.audio_path),
                            ref_text=voice_meta.transcript
                        )
                prompt = await loop.run_in_executor(self._executor, _create_prompt)
                self.prompt_cache[voice_id] = prompt
                return prompt
            except Exception as e:
                logger.warning(f"Could not pre-compute voice clone prompt for '{voice_id}': {e}. Fallback to direct ref_audio.")
                return None
        return None

    def _generate_audio(self, gen_kwargs: dict) -> np.ndarray:
        with torch.inference_mode():
            output_audio = self.model.generate(**gen_kwargs)
        
        audio_data = output_audio[0] if isinstance(output_audio, (tuple, list)) else output_audio
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            audio_data = np.asarray(audio_data, dtype=np.float32)

        if audio_data.ndim > 1:
            audio_data = np.squeeze(audio_data)

        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
        return audio_data

    def _cleanup_cuda(self) -> None:
        """Helper to safely empty CUDA memory cache and force garbage collection after synthesis."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.debug(f"CUDA memory cleanup warning: {e}")

    async def synthesize(
        self,
        voice_meta: VoiceMetadata,
        text: str,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        response_format: str = "wav",
        seed: Optional[int] = None
    ) -> Tuple[bytes, str]:
        """
        Synthesizes text using specified voice_meta and generation options.
        Returns tuple of (audio_bytes, mime_type).
        """
        loop = asyncio.get_running_loop()

        if not self.is_loaded or self.model is None:
            await loop.run_in_executor(self._executor, self.load_model)

        # Merge parameters with hierarchy: Request explicit > Voice JSON/YAML > Server defaults
        voice_settings = voice_meta.settings or {}
        defaults = config.default_gen_config

        final_language = language or voice_settings.get("language") or defaults.get("language", "en")
        final_speed = speed if speed is not None else voice_settings.get("speed", defaults.get("speed", 1.0))
        final_num_step = num_step if num_step is not None else voice_settings.get("num_step", defaults.get("num_step", 32))
        final_guidance_scale = guidance_scale if guidance_scale is not None else voice_settings.get("guidance_scale", defaults.get("guidance_scale", 2.0))

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        async with self._lock:
            # Check prompt cache
            cached_prompt = await self.get_or_create_prompt(voice_meta)

            # Normalize whitespace
            clean_text = re.sub(r'[\r\n\t]+', ' ', text).strip()
            if not clean_text:
                clean_text = text

            logger.info(f"Synthesizing speech for voice '{voice_meta.voice_id}' ({len(clean_text)} chars): '{clean_text[:60]}...'")

            gen_kwargs = {
                "text": clean_text,
                "speed": float(final_speed),
                "num_step": int(final_num_step),
                "guidance_scale": float(final_guidance_scale)
            }

            if final_language:
                gen_kwargs["language"] = final_language

            if cached_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = cached_prompt
            else:
                gen_kwargs["ref_audio"] = str(voice_meta.audio_path)
                if voice_meta.transcript:
                    gen_kwargs["ref_text"] = voice_meta.transcript

            try:
                combined_audio = await loop.run_in_executor(self._executor, lambda kw=gen_kwargs: self._generate_audio(kw))
            finally:
                # Flush CUDA memory cache post-synthesis
                await loop.run_in_executor(self._executor, self._cleanup_cuda)

            combined_audio = np.ascontiguousarray(combined_audio, dtype=np.float32)

        # Format output audio
        sr = 24000  # Default sampling rate for OmniVoice
        audio_bytes, mime_type = self._encode_audio(combined_audio, sr, response_format)
        return audio_bytes, mime_type

    def _encode_audio(self, audio_data: np.ndarray, samplerate: int, fmt: str) -> Tuple[bytes, str]:
        import shutil
        import subprocess

        fmt = fmt.lower().strip()
        
        # Ensure float32 audio data is finite and normalized within [-1.0, 1.0] to prevent codec clipping/overflow
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

        # 1. Always generate crash-free in-memory WAV first
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, samplerate, format="WAV")
        buffer.seek(0)
        wav_bytes = buffer.read()

        if fmt in {"wav", "wave"}:
            return wav_bytes, "audio/wav"

        elif fmt in {"flac"}:
            try:
                flac_buf = io.BytesIO()
                sf.write(flac_buf, audio_data, samplerate, format="FLAC")
                flac_buf.seek(0)
                return flac_buf.read(), "audio/flac"
            except Exception as e:
                logger.warning(f"FLAC encoding failed ({e}), returning WAV.")
                return wav_bytes, "audio/wav"

        elif fmt in {"ogg", "opus", "vorbis"}:
            # CRITICAL: libsndfile.so in PySoundFile has a C-level segmentation fault when encoding OGG directly on Linux.
            # To guarantee process safety and prevent core dumps, convert the clean WAV bytes via external ffmpeg pipe or pydub.
            if shutil.which("ffmpeg"):
                try:
                    cmd = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-y", "-f", "wav", "-i", "pipe:0",
                        "-c:a", "libvorbis", "-q:a", "4",
                        "-f", "ogg", "pipe:1"
                    ]
                    proc = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=15)
                    if proc.returncode == 0 and proc.stdout:
                        return proc.stdout, "audio/ogg"
                    else:
                        logger.warning(f"ffmpeg OGG encoding stderr: {proc.stderr.decode('utf-8', errors='ignore')}")
                except Exception as e:
                    logger.warning(f"ffmpeg OGG subprocess failed: {e}")

            # Fallback to pydub if installed
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
                out_buf = io.BytesIO()
                seg.export(out_buf, format="ogg", codec="libvorbis")
                out_buf.seek(0)
                return out_buf.read(), "audio/ogg"
            except Exception as e:
                logger.warning(f"pydub OGG conversion failed: {e}")

            logger.warning("OGG conversion tools unavailable or failed. Falling back to WAV output.")
            return wav_bytes, "audio/wav"

        elif fmt in {"mp3"}:
            if shutil.which("ffmpeg"):
                try:
                    cmd = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-y", "-f", "wav", "-i", "pipe:0",
                        "-c:a", "libmp3lame", "-q:a", "2",
                        "-f", "mp3", "pipe:1"
                    ]
                    proc = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=15)
                    if proc.returncode == 0 and proc.stdout:
                        return proc.stdout, "audio/mpeg"
                except Exception as e:
                    logger.warning(f"ffmpeg MP3 subprocess failed: {e}")

            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                sf.write(tmp_path, audio_data, samplerate, format="MP3")
                mp3_bytes = tmp_path.read_bytes()
                return mp3_bytes, "audio/mpeg"
            except Exception as e:
                logger.warning(f"Soundfile MP3 format export failed ({e}), returning WAV output.")
                return wav_bytes, "audio/wav"
            finally:
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

        # Default fallback to WAV
        return wav_bytes, "audio/wav"

model_manager = ModelManager()
