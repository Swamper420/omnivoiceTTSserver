import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

# ponytail: single quant knob maps to both GGUF filenames; per-file
# overrides only if a mix (e.g. Q4 base + F32 tokenizer) ever matters.
VALID_QUANTS = ("F32", "BF16", "Q8_0", "Q4_K_M")

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
    def model_repo_id(self) -> str:
        return os.getenv("MODEL_REPO_ID", self._data.get("model", {}).get("repo_id", "Serveurperso/OmniVoice-GGUF"))

    @property
    def model_quant(self) -> str:
        q = os.getenv("MODEL_QUANT", self._data.get("model", {}).get("quant", "Q4_K_M"))
        q = str(q).upper()
        return q if q in VALID_QUANTS else "Q4_K_M"

    @property
    def model_base_file(self) -> str:
        return os.getenv("MODEL_BASE_FILE",
            self._data.get("model", {}).get("base_file", f"omnivoice-base-{self.model_quant}.gguf"))

    @property
    def model_tokenizer_file(self) -> str:
        return os.getenv("MODEL_TOKENIZER_FILE",
            self._data.get("model", {}).get("tokenizer_file", f"omnivoice-tokenizer-{self.model_quant}.gguf"))

    @property
    def models_dir(self) -> Path:
        raw_path = os.getenv("MODELS_DIR", self._data.get("model", {}).get("models_dir", "models"))
        path = Path(raw_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tts_binary(self) -> str:
        return os.getenv("TTS_BINARY", self._data.get("model", {}).get("binary", "omnivoice-tts"))

    @property
    def device(self) -> str:
        return os.getenv("DEVICE", self._data.get("model", {}).get("device", "cuda"))

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
