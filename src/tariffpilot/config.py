"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("TP_ROOT", Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    mode: str = "offline"  # offline | gemini | replay
    gemini_api_key: str = ""
    model_flash: str = "gemini-flash-latest"
    model_lite: str = "gemini-flash-lite-latest"
    embed_model: str = "gemini-embedding-001"
    rpm: int = 8
    rpd: int = 200
    api_keys: tuple[str, ...] = ()
    api_rate_per_min: int = 30
    state: Path | None = None

    @property
    def prompts_dir(self) -> Path: return self.root / "prompts"
    @property
    def skills_dir(self) -> Path: return self.root / "skills"
    @property
    def policy_path(self) -> Path: return self.root / "policy" / "policy.yaml"
    @property
    def data_dir(self) -> Path: return self.root / "data"
    @property
    def cache_dir(self) -> Path: return self.state_dir / "llm_cache"
    @property
    def state_dir(self) -> Path: return self.state or (self.root / ".state")
    @property
    def audit_db(self) -> Path: return self.state_dir / "audit.db"

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        keys = tuple(k.strip() for k in e.get("TP_API_KEYS", "").split(",") if k.strip())
        return cls(
            root=Path(e.get("TP_ROOT", ROOT)),
            mode=e.get("TP_MODE", "offline"),
            gemini_api_key=e.get("GEMINI_API_KEY", ""),
            model_flash=e.get("GEMINI_MODEL_FLASH", "gemini-flash-latest"),
            model_lite=e.get("GEMINI_MODEL_LITE", "gemini-flash-lite-latest"),
            embed_model=e.get("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            rpm=int(e.get("TP_RPM", 8)),
            rpd=int(e.get("TP_RPD", 200)),
            api_keys=keys,
            api_rate_per_min=int(e.get("TP_API_RATE_PER_MIN", 30)),
            state=Path(e["TP_STATE_DIR"]) if e.get("TP_STATE_DIR") else None,
        )
