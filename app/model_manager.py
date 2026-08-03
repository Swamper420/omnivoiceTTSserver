import asyncio
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


def split_text_into_chunks(text: str, max_chars: int = 100, max_sentences: int = 2) -> List[str]:
    """
    Splits input text into chunks containing at most `max_sentences` sentences
    and at most `max_chars` characters per chunk.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Split text into sentences (preserving punctuation)
    raw_sentences = re.split(r'(?<=[.!?\n])\s+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    # 2. Break down any sentence exceeding max_chars into smaller clause/word sub-units
    units: List[Tuple[str, bool]] = []  # List of (sub_unit_text, is_sentence_end)
    for s in sentences:
        if len(s) <= max_chars:
            units.append((s, True))
        else:
            # Sub-split long sentence by secondary punctuation like commas, semicolons, colons, dashes
            clauses = re.split(r'(?<=[,;:\-—])\s+', s)
            clauses = [c.strip() for c in clauses if c.strip()]

            sub_units: List[str] = []
            for clause in clauses:
                if len(clause) <= max_chars:
                    sub_units.append(clause)
                else:
                    # Sub-split long clause by word boundaries
                    words = clause.split()
                    curr_words: List[str] = []
                    curr_len = 0
                    for w in words:
                        needed = len(w) if not curr_words else len(w) + 1
                        if curr_len + needed <= max_chars:
                            curr_words.append(w)
                            curr_len += needed
                        else:
                            if curr_words:
                                sub_units.append(" ".join(curr_words))
                            # Handle single word exceeding max_chars
                            while len(w) > max_chars:
                                sub_units.append(w[:max_chars])
                                w = w[max_chars:]
                            curr_words = [w]
                            curr_len = len(w)
                    if curr_words:
                        sub_units.append(" ".join(curr_words))

            for idx, sub_unit in enumerate(sub_units):
                is_end = (idx == len(sub_units) - 1)
                units.append((sub_unit, is_end))

    # 3. Group units into chunks satisfying max_chars and max_sentences
    chunks: List[str] = []
    curr_chunk: List[str] = []
    curr_len = 0
    curr_sentences = 0

    for unit_text, is_sentence_end in units:
        unit_len = len(unit_text)
        space_len = 1 if curr_chunk else 0

        would_exceed_chars = (curr_len + space_len + unit_len > max_chars)
        would_exceed_sentences = (curr_sentences >= max_sentences)

        if would_exceed_chars or would_exceed_sentences:
            if curr_chunk:
                chunks.append(" ".join(curr_chunk))
                curr_chunk = []
                curr_len = 0
                curr_sentences = 0

        curr_chunk.append(unit_text)
        curr_len += (1 if len(curr_chunk) > 1 else 0) + unit_len
        if is_sentence_end:
            curr_sentences += 1

    if curr_chunk:
        chunks.append(" ".join(curr_chunk))

    return chunks


class ModelManager:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.prompt_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

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
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(voice_meta.audio_path),
                    ref_text=voice_meta.transcript
                )
                self.prompt_cache[voice_id] = prompt
                return prompt
            except Exception as e:
                logger.warning(f"Could not pre-compute voice clone prompt for '{voice_id}': {e}. Fallback to direct ref_audio.")
                return None
        return None

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
        if not self.is_loaded or self.model is None:
            self.load_model()

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

            chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
            if not chunks:
                chunks = [text]

            logger.info(f"Synthesizing text for voice '{voice_meta.voice_id}' split into {len(chunks)} chunk(s): {chunks}")

            base_kwargs = {
                "speed": float(final_speed),
                "num_step": int(final_num_step),
                "guidance_scale": float(final_guidance_scale)
            }

            if final_language:
                base_kwargs["language"] = final_language

            if cached_prompt is not None:
                base_kwargs["voice_clone_prompt"] = cached_prompt
            else:
                base_kwargs["ref_audio"] = str(voice_meta.audio_path)
                if voice_meta.transcript:
                    base_kwargs["ref_text"] = voice_meta.transcript

            audio_chunks = []
            loop = asyncio.get_running_loop()

            for chunk_idx, chunk_text in enumerate(chunks):
                gen_kwargs = dict(base_kwargs)
                gen_kwargs["text"] = chunk_text

                logger.info(f"Generating chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_text)} chars): '{chunk_text}'")

                output_audio = await loop.run_in_executor(None, lambda kw=gen_kwargs: self.model.generate(**kw))

                audio_data = output_audio[0] if isinstance(output_audio, (tuple, list)) else output_audio
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().numpy()

                if audio_data.ndim > 1:
                    audio_data = np.squeeze(audio_data)

                audio_chunks.append(audio_data)

            if len(audio_chunks) == 1:
                combined_audio = audio_chunks[0]
            else:
                combined_audio = np.concatenate(audio_chunks)

        # Format output audio
        sr = 24000  # Default sampling rate for OmniVoice
        audio_bytes, mime_type = self._encode_audio(combined_audio, sr, response_format)
        return audio_bytes, mime_type

    def _encode_audio(self, audio_data: np.ndarray, samplerate: int, fmt: str) -> Tuple[bytes, str]:
        fmt = fmt.lower().strip()
        buffer = io.BytesIO()

        if fmt in {"wav", "wave"}:
            sf.write(buffer, audio_data, samplerate, format="WAV")
            mime_type = "audio/wav"
        elif fmt in {"flac"}:
            sf.write(buffer, audio_data, samplerate, format="FLAC")
            mime_type = "audio/flac"
        elif fmt in {"ogg", "opus"}:
            sf.write(buffer, audio_data, samplerate, format="OGG")
            mime_type = "audio/ogg"
        elif fmt in {"mp3"}:
            # Write wav buffer first, fallback to WAV if mp3 writer not available in soundfile
            try:
                sf.write(buffer, audio_data, samplerate, format="MP3")
                mime_type = "audio/mpeg"
            except Exception:
                logger.warning("Soundfile MP3 format write failed, falling back to WAV output.")
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, samplerate, format="WAV")
                mime_type = "audio/wav"
        else:
            sf.write(buffer, audio_data, samplerate, format="WAV")
            mime_type = "audio/wav"

        buffer.seek(0)
        return buffer.read(), mime_type

model_manager = ModelManager()
