import asyncio
import concurrent.futures
import io
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import soundfile as sf

from app.config import config
from app.voice_manager import VoiceMetadata

logger = logging.getLogger(__name__)


class ModelManager:
    """GGUF backend: shells out to the omnivoice.cpp `omnivoice-tts` CLI.

    Models are the `Serveurperso/OmniVoice-GGUF` pair:
    `omnivoice-base-{quant}.gguf` + `omnivoice-tokenizer-{quant}.gguf`.
    Change size with `model.quant` in config.yaml (or MODEL_QUANT env).
    """

    def __init__(self):
        self.is_loaded = False
        self.model_path: Optional[Path] = None
        self.codec_path: Optional[Path] = None
        self.binary: Optional[str] = None
        self.prompt_cache: Dict[str, Any] = {}  # kept so /voices/reload clear() still works; CLI holds no state
        self._lock = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omnivoice_gguf"
        )

    def resolve_binary(self) -> str:
        candidates = [config.tts_binary, "omnivoice-tts",
                      "/app/omnivoice.cpp/build/bin/omnivoice-tts",
                      "/app/omnivoice.cpp/build/omnivoice-tts",
                      "build/bin/omnivoice-tts", "build/omnivoice-tts"]
        for c in candidates:
            found = shutil.which(c) or (c if Path(c).is_file() else None)
            if found:
                return found
        raise RuntimeError(
            "omnivoice-tts binary not found. Build omnivoice.cpp "
            "(./buildcuda.sh) or set model.binary / TTS_BINARY.")

    def ensure_models(self) -> Tuple[Path, Path]:
        from huggingface_hub import hf_hub_download
        models_dir = config.models_dir
        base = models_dir / config.model_base_file
        codec = models_dir / config.model_tokenizer_file
        for f in (base, codec):
            if f.is_file():
                continue
            logger.info(f"Downloading {f.name} from {config.model_repo_id} ({config.model_quant})...")
            dl = hf_hub_download(repo_id=config.model_repo_id, filename=f.name,
                                 local_dir=str(models_dir))
            logger.info(f"Downloaded {dl}")
        return base, codec

    def load_model(self) -> None:
        if self.is_loaded and self.model_path and self.codec_path:
            return
        self.binary = self.resolve_binary()
        self.model_path, self.codec_path = self.ensure_models()
        logger.info(f"GGUF ready: base={self.model_path.name} codec={self.codec_path.name} via {self.binary}")
        self.is_loaded = True

    async def get_or_create_prompt(self, voice_meta: VoiceMetadata) -> Optional[Any]:
        return None  # ponytail: CLI takes --ref-wav per call; no embedding cache to keep

    def _run_tts_cli(self, text: str, voice_meta: VoiceMetadata,
                     language: Optional[str], num_step: int,
                     seed: Optional[int]) -> bytes:
        ref_text_file = None
        tmp_ref_text = None
        try:
            cmd = [self.binary,
                   "--model", str(self.model_path),
                   "--codec", str(self.codec_path)]
            if language:
                cmd += ["--lang", language]
            if num_step:
                cmd += ["--steps", str(int(num_step))]
            if seed is not None:
                cmd += ["--seed", str(int(seed))]
            if voice_meta.audio_path and Path(voice_meta.audio_path).is_file():
                cmd += ["--ref-wav", str(voice_meta.audio_path)]
                if voice_meta.transcript_path and Path(voice_meta.transcript_path).is_file():
                    ref_text_file = str(voice_meta.transcript_path)
                elif voice_meta.transcript:
                    tmp_ref_text = tempfile.NamedTemporaryFile(
                        suffix=".txt", mode="w", encoding="utf-8", delete=False)
                    tmp_ref_text.write(voice_meta.transcript)
                    tmp_ref_text.close()
                    ref_text_file = tmp_ref_text.name
                if ref_text_file:
                    cmd += ["--ref-text", ref_text_file]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
                out_path = out.name
            cmd += ["-o", out_path]
            proc = subprocess.run(cmd, input=text.encode("utf-8"),
                                  capture_output=True, timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[-2000:])
            return Path(out_path).read_bytes()
        finally:
            if tmp_ref_text:
                Path(tmp_ref_text.name).unlink(missing_ok=True)
            if "out_path" in locals():
                Path(out_path).unlink(missing_ok=True)

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
        loop = asyncio.get_running_loop()
        if not self.is_loaded:
            await loop.run_in_executor(self._executor, self.load_model)

        voice_settings = voice_meta.settings or {}
        defaults = config.default_gen_config
        final_language = language or voice_settings.get("language") or defaults.get("language", "en")
        final_num_step = num_step if num_step is not None else voice_settings.get("num_step", defaults.get("num_step", 32))
        if speed is not None or "speed" in voice_settings:
            logger.warning("speed has no effect on the GGUF CLI backend; ignoring.")
        if guidance_scale is not None or "guidance_scale" in voice_settings:
            logger.warning("guidance_scale has no CLI flag (library default 2.0); ignoring.")

        clean_text = re.sub(r'[\r\n\t]+', ' ', text).strip() or text
        logger.info(f"Synthesizing (GGUF {config.model_quant}) voice '{voice_meta.voice_id}' "
                    f"({len(clean_text)} chars): '{clean_text[:60]}...'")

        async with self._lock:
            wav_bytes = await loop.run_in_executor(
                self._executor, lambda: self._run_tts_cli(
                    clean_text, voice_meta, final_language, int(final_num_step), seed))

        # CLI always emits 24 kHz mono WAV; convert only if another format asked
        if (response_format or "wav").lower().strip() in {"wav", "wave"}:
            return wav_bytes, "audio/wav"
        buf = io.BytesIO(wav_bytes)
        audio_data, sr = sf.read(buf, dtype="float32", always_2d=False)
        return self._encode_audio(np.ascontiguousarray(audio_data, dtype=np.float32), sr, response_format)

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
