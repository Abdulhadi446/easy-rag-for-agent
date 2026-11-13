"""Configuration loading for easy-rag.

Resolution order (last wins):
    1. built-in defaults
    2. ``$XDG_CONFIG_HOME/easy-rag/config.json``
    3. ``.env`` in the current working directory (or a parent, up to the project root)
    4. process environment (``EASYRAG_*`` and common provider keys)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "easy-rag"

DEFAULT_MODEL = "openrouter/free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 180.0

API_KEY_ENV_VARS = ("EASYRAG_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


def config_dir() -> Path:
    """Where non-secret settings live."""
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return Path(os.environ.get("EASYRAG_CONFIG_DIR", root / APP_NAME))


def data_dir() -> Path:
    """Where the Chroma database lives."""
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return Path(os.environ.get("EASYRAG_DATA_DIR", root / APP_NAME))


def default_db_path() -> Path:
    return data_dir() / "chroma"


def parse_dotenv(path: Path) -> dict[str, str]:
    """Tiny ``.env`` reader: ``KEY=value`` lines, ``#`` comments, optional quotes."""
    values: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from *start* looking for a `.env` (stops after the filesystem root)."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _as_int(value: object, fallback: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return fallback


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return fallback


@dataclass
class Settings:
    """Everything the app needs to talk to a model and a vector store."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    db_path: Path = field(default_factory=default_db_path)
    collection: str = "documents"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT
    extra_headers: dict[str, str] = field(default_factory=dict)
    env_file: Path | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, env_file: Path | None = None, cwd: Path | None = None) -> "Settings":
        settings = cls()

        config_file = config_dir() / "config.json"
        if config_file.is_file():
            try:
                settings._apply(config_file.read_text(encoding="utf-8") and json.loads(config_file.read_text()))
            except (OSError, json.JSONDecodeError):
                pass

        dotenv = env_file if env_file is not None else find_dotenv(cwd)
        if dotenv and dotenv.is_file():
            settings.env_file = dotenv
            settings._apply(parse_dotenv(dotenv))

        settings._apply({key: value for key, value in os.environ.items() if value})
        settings._finalise()
        return settings

    def _apply(self, values: dict[str, object]) -> None:
        mapping = {
            "EASYRAG_API_KEY": "api_key",
            "OPENROUTER_API_KEY": "api_key",
            "OPENAI_API_KEY": "api_key",
            "EASYRAG_BASE_URL": "base_url",
            "OPENROUTER_BASE_URL": "base_url",
            "OPENAI_BASE_URL": "base_url",
            "EASYRAG_MODEL": "model",
            "MODEL": "model",
            "EASYRAG_DB": "db_path",
            "EASYRAG_COLLECTION": "collection",
            "EASYRAG_CHUNK_SIZE": "chunk_size",
            "EASYRAG_CHUNK_OVERLAP": "chunk_overlap",
            "EASYRAG_TOP_K": "top_k",
            "EASYRAG_MAX_TOKENS": "max_tokens",
            "EASYRAG_TEMPERATURE": "temperature",
            "EASYRAG_TIMEOUT": "timeout",
            # plain config.json keys
            "api_key": "api_key",
            "base_url": "base_url",
            "model": "model",
            "db_path": "db_path",
            "collection": "collection",
            "chunk_size": "chunk_size",
            "chunk_overlap": "chunk_overlap",
            "top_k": "top_k",
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "timeout": "timeout",
            "extra_headers": "extra_headers",
        }
        for key, attribute in mapping.items():
            if key not in values:
                continue
            value = values[key]
            if value in (None, ""):
                continue
            if attribute in {"chunk_size", "chunk_overlap", "top_k", "max_tokens"}:
                setattr(self, attribute, _as_int(value, getattr(self, attribute)))
            elif attribute in {"temperature", "timeout"}:
                setattr(self, attribute, _as_float(value, getattr(self, attribute)))
            elif attribute == "db_path":
                self.db_path = Path(str(value)).expanduser()
            else:
                setattr(self, attribute, value)

    def _finalise(self) -> None:
        if not self.api_key:
            for name in API_KEY_ENV_VARS:
                if os.environ.get(name):
                    self.api_key = os.environ[name]
                    break
        self.base_url = self.base_url.rstrip("/")
        if self.model.startswith("openrouter/") and "openrouter.ai" not in self.base_url:
            # keep the id, the endpoint decides.
            pass
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 5)
        self.db_path = Path(self.db_path).expanduser()

    # ------------------------------------------------------------------ helpers
    @property
    def configured(self) -> bool:
        """True when a key and a model are present."""
        return bool(self.api_key and self.model)

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["api_key"] = "set" if self.api_key else "missing"
        data["db_path"] = str(self.db_path)
        data["env_file"] = str(self.env_file) if self.env_file else None
        return data

    def save(self, path: Path | None = None) -> Path:
        """Persist non-secret settings to the config file."""
        target = path or (config_dir() / "config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "base_url": self.base_url,
            "model": self.model,
            "db_path": str(self.db_path),
            "collection": self.collection,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if self.extra_headers:
            data["extra_headers"] = self.extra_headers
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return target

    def update(self, **values: object) -> None:
        """Set known fields from strings (used by the CLI)."""
        valid = {item.name for item in fields(self)}
        cleaned = {key: value for key, value in values.items() if key in valid}
        self._apply(cleaned)
        self._finalise()
