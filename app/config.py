import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

class AppConfig:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self._data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

    @property
    def host(self) -> str:
        return os.getenv("HOST", self._data.get("server", {}).get("host", "0.0.0.0"))

    @property
    def port(self) -> int:
        return int(os.getenv("PORT", self._data.get("server", {}).get("port", 8000)))

    @property
    def model_name_or_path(self) -> str:
        return os.getenv("MODEL_NAME", self._data.get("model", {}).get("name_or_path", "k2-fsa/OmniVoice"))

    @property
    def device(self) -> str:
        return os.getenv("DEVICE", self._data.get("model", {}).get("device", "cuda"))

    @property
    def dtype(self) -> str:
        return os.getenv("DTYPE", self._data.get("model", {}).get("dtype", "float16"))

    @property
    def load_asr(self) -> bool:
        # User requested to NEVER load whisper / ASR under any circumstances
        return False

    @property
    def voices_dir(self) -> Path:
        raw_path = os.getenv("VOICES_DIR", self._data.get("storage", {}).get("voices_dir", "storage/voices"))
        path = Path(raw_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def prompt_cache_dir(self) -> Path:
        raw_path = os.getenv("PROMPT_CACHE_DIR", self._data.get("storage", {}).get("prompt_cache_dir", "storage/cache"))
        path = Path(raw_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def default_gen_config(self) -> Dict[str, Any]:
        return self._data.get("default_gen_config", {
            "language": "en",
            "speed": 1.0,
            "num_step": 32,
            "guidance_scale": 2.0,
            "response_format": "wav"
        })

config = AppConfig()
