import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from app.config import config

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

class VoiceMetadata:
    def __init__(
        self,
        voice_id: str,
        audio_path: Path,
        transcript_path: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        self.voice_id = voice_id
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.config_path = config_path
        self.transcript: str = ""
        self.settings: Dict[str, Any] = {}
        self.load_resources()

    def load_resources(self) -> None:
        # Load transcript text file
        if self.transcript_path and self.transcript_path.exists():
            try:
                self.transcript = self.transcript_path.read_text(encoding="utf-8").strip()
                logger.info(f"Loaded transcript for voice '{self.voice_id}': '{self.transcript[:50]}...'")
            except Exception as e:
                logger.error(f"Error reading transcript file {self.transcript_path}: {e}")
                self.transcript = ""
        else:
            logger.warning(f"No transcript file found for voice '{self.voice_id}' at expected path.")

        # Load per-voice configuration settings (JSON or YAML)
        if self.config_path and self.config_path.exists():
            try:
                if self.config_path.suffix.lower() == ".json":
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        self.settings = json.load(f)
                elif self.config_path.suffix.lower() in {".yaml", ".yml"}:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        self.settings = yaml.safe_load(f) or {}
                logger.info(f"Loaded custom configuration for voice '{self.voice_id}': {self.settings}")
            except Exception as e:
                logger.error(f"Error loading config file {self.config_path}: {e}")
                self.settings = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "audio_path": str(self.audio_path),
            "has_transcript": bool(self.transcript),
            "transcript": self.transcript,
            "settings": self.settings,
        }

class VoiceManager:
    def __init__(self, voices_dir: Optional[Path] = None):
        self.voices_dir = voices_dir or config.voices_dir
        self.voices: Dict[str, VoiceMetadata] = {}
        self.reload_voices()

    def reload_voices(self) -> Dict[str, VoiceMetadata]:
        logger.info(f"Scanning voices directory: {self.voices_dir}")
        new_voices: Dict[str, VoiceMetadata] = {}

        if not self.voices_dir.exists():
            self.voices_dir.mkdir(parents=True, exist_ok=True)

        for file in self.voices_dir.iterdir():
            if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
                voice_id = file.stem  # e.g., 'voice_fi' from 'voice_fi.wav'
                
                # Check for corresponding transcript file (e.g. voice_fi.txt)
                transcript_path = self.voices_dir / f"{voice_id}.txt"
                if not transcript_path.exists():
                    transcript_path = None

                # Check for companion configuration file (voice_fi.json or voice_fi.yaml)
                config_path = self.voices_dir / f"{voice_id}.json"
                if not config_path.exists():
                    config_path = self.voices_dir / f"{voice_id}.yaml"
                if not config_path.exists():
                    config_path = self.voices_dir / f"{voice_id}.yml"
                if not config_path.exists():
                    config_path = None

                voice_meta = VoiceMetadata(
                    voice_id=voice_id,
                    audio_path=file,
                    transcript_path=transcript_path,
                    config_path=config_path
                )
                new_voices[voice_id] = voice_meta

        self.voices = new_voices
        logger.info(f"Discovered {len(self.voices)} voice(s): {list(self.voices.keys())}")
        return self.voices

    def get_voice(self, voice_id: str) -> Optional[VoiceMetadata]:
        # Handle matching with or without file extension
        clean_id = Path(voice_id).stem
        if clean_id in self.voices:
            return self.voices[clean_id]
        return None

    def list_voices(self) -> List[Dict[str, Any]]:
        return [v.to_dict() for v in self.voices.values()]

voice_manager = VoiceManager()
