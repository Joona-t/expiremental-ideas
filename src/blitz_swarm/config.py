"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_seed_paths() -> list[Path]:
    artifact = _workspace_root() / "compass_artifact_wf-f6ac8f4e-2c3f-47d2-8611-8dc2eb1032cc_text_markdown.md"
    return [artifact] if artifact.exists() else []


class AppSettings(BaseModel):
    """Static configuration for the local Blitz Swarm runtime."""

    workspace_root: Path = Field(default_factory=_workspace_root)
    state_dir: Path = Field(default_factory=lambda: _workspace_root() / ".blitz" / "state")
    run_dir: Path = Field(default_factory=lambda: _workspace_root() / ".blitz" / "runs")
    redis_url: str = "redis://127.0.0.1:6379/0"
    sqlite_path: Path = Field(default_factory=lambda: _workspace_root() / ".blitz" / "state" / "blitz_swarm.db")
    vector_dir: Path = Field(default_factory=lambda: _workspace_root() / ".blitz" / "state" / "lancedb")
    openai_api_base: str | None = None
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    default_max_rounds: int = 4
    retention_hot_limit: int = 250
    snapshot_interval: int = 2
    seed_corpus_paths: list[Path] = Field(default_factory=_default_seed_paths)

    @classmethod
    def from_env(cls) -> "AppSettings":
        workspace_root = _workspace_root()
        state_dir = Path(os.getenv("BLITZ_STATE_DIR", workspace_root / ".blitz" / "state"))
        run_dir = Path(os.getenv("BLITZ_RUN_DIR", workspace_root / ".blitz" / "runs"))
        return cls(
            workspace_root=workspace_root,
            state_dir=state_dir,
            run_dir=run_dir,
            redis_url=os.getenv("BLITZ_REDIS_URL", "redis://127.0.0.1:6379/0"),
            sqlite_path=Path(os.getenv("BLITZ_SQLITE_PATH", state_dir / "blitz_swarm.db")),
            vector_dir=Path(os.getenv("BLITZ_VECTOR_DIR", state_dir / "lancedb")),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            default_max_rounds=int(os.getenv("BLITZ_MAX_ROUNDS", "4")),
            retention_hot_limit=int(os.getenv("BLITZ_RETENTION_HOT_LIMIT", "250")),
            snapshot_interval=int(os.getenv("BLITZ_SNAPSHOT_INTERVAL", "2")),
            seed_corpus_paths=_default_seed_paths(),
        )

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
