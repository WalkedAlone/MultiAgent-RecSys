"""Centralised configuration.

Two kinds of configuration exist in this project:

1. **Runtime secrets / endpoints** (``.env``) are read with
   :class:`pydantic_settings.BaseSettings` via :class:`DeepSeekSettings`.
2. **Experiment configuration** (``configs/*.yaml``) is loaded once, validated
   with Pydantic models and exposed as a typed :class:`AppConfig`.

No API key is ever stored in YAML or Python source code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ENV_FILE = PROJECT_ROOT / ".env"

DeviceLiteral = Literal["auto", "cpu", "cuda"]


class _Section(BaseModel):
    """Strict base model for config sections (typos are rejected)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------
class SampleConfig(_Section):
    enabled: bool = True
    max_users: int = Field(default=20_000, gt=0)
    max_reviews_per_item: int | None = Field(default=50, ge=1)


class FullConfig(_Section):
    enabled: bool = False
    max_reviews_per_item: int | None = Field(default=None, ge=1)


class NegativeSamplingConfig(_Section):
    enabled: bool = True
    num_negatives: int = Field(default=4, ge=1, le=100)
    seed: int = Field(default=42, ge=0)


class DataPaths(_Section):
    raw_reviews: str = "data/raw/reviews"
    raw_metadata: str = "data/raw/metadata"
    raw_benchmark: str = "data/raw/benchmark"
    processed: str = "data/processed"
    splits: str = "data/splits"
    features: str = "data/features"


class DownloadConfig(_Section):
    chunk_size_bytes: int = Field(default=1_048_576, ge=65_536)
    segment_size_bytes: int = Field(default=33_554_432, ge=1_048_576)
    connections: int = Field(default=16, ge=1, le=32)
    max_retries: int = Field(default=5, ge=0, le=20)
    timeout_seconds: float = Field(default=60.0, gt=0)
    resume: bool = True
    verify_gzip: bool = True


class DataConfig(_Section):
    mode: Literal["sample", "full"] = "sample"
    random_seed: int = Field(default=42, ge=0)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    full: FullConfig = Field(default_factory=FullConfig)
    min_user_interactions: int = Field(default=5, ge=1)
    min_item_interactions: int = Field(default=5, ge=1)
    paths: DataPaths = Field(default_factory=DataPaths)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    negative_sampling: NegativeSamplingConfig = Field(
        default_factory=NegativeSamplingConfig
    )

    @model_validator(mode="after")
    def _mode_matches_enabled_flag(self) -> "DataConfig":
        if self.mode == "sample" and not self.sample.enabled:
            raise ValueError("data.sample.enabled must be true when data.mode is 'sample'")
        if self.mode == "full" and not self.full.enabled:
            raise ValueError("data.full.enabled must be true when data.mode is 'full'")
        return self


# ---------------------------------------------------------------------------
# model.yaml
# ---------------------------------------------------------------------------
class EmbeddingConfig(_Section):
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = Field(default=64, ge=1)
    normalize: bool = True
    device: DeviceLiteral = "auto"


class TwoTowerConfig(_Section):
    user_embedding_dim: int = Field(default=64, ge=8)
    item_embedding_dim: int = Field(default=64, ge=8)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=2048, ge=1)
    epochs: int = Field(default=20, ge=1)
    patience: int = Field(default=3, ge=1)
    num_negatives: int = Field(default=4, ge=1)


class LightGCNConfig(_Section):
    embedding_dim: int = Field(default=64, ge=8)
    num_layers: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=4096, ge=1)
    epochs: int = Field(default=50, ge=1)
    patience: int = Field(default=5, ge=1)
    num_negatives: int = Field(default=1, ge=1)


class DeepFMConfig(_Section):
    embedding_dim: int = Field(default=16, ge=4)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=1024, ge=1)
    epochs: int = Field(default=10, ge=1)
    patience: int = Field(default=2, ge=1)


class DINConfig(_Section):
    embedding_dim: int = Field(default=16, ge=4)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    attention_hidden_dim: int = Field(default=64, ge=4)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=512, ge=1)
    epochs: int = Field(default=10, ge=1)
    patience: int = Field(default=2, ge=1)


class ModelConfig(_Section):
    device: DeviceLiteral = "auto"
    seed: int = Field(default=42, ge=0)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    two_tower: TwoTowerConfig = Field(default_factory=TwoTowerConfig)
    lightgcn: LightGCNConfig = Field(default_factory=LightGCNConfig)
    deepfm: DeepFMConfig = Field(default_factory=DeepFMConfig)
    din: DINConfig = Field(default_factory=DINConfig)


# ---------------------------------------------------------------------------
# retrieval.yaml
# ---------------------------------------------------------------------------
class ItemCFConfig(_Section):
    enabled: bool = True
    top_k: int = Field(default=100, ge=1, le=1000)
    similarity: Literal["cosine"] = "cosine"
    max_history: int = Field(default=50, ge=1, le=500)


class TwoTowerRetrievalConfig(_Section):
    enabled: bool = True
    top_k: int = Field(default=100, ge=1, le=1000)
    index_path: str = "indexes/two_tower.faiss"
    item_ids_path: str = "indexes/two_tower_item_ids.json"


class LightGCNRetrievalConfig(_Section):
    enabled: bool = True
    top_k: int = Field(default=100, ge=1, le=1000)
    item_embeddings_path: str = "indexes/lightgcn_item_embeddings.npy"
    item_ids_path: str = "indexes/lightgcn_item_ids.json"


class FusionConfig(_Section):
    strategy: Literal["quota", "rrf", "quota_rrf"] = "quota"
    total_candidates: int = Field(default=300, ge=1)
    quota_per_channel: int = Field(default=100, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    score_normalization: Literal["min_max", "zscore", "none"] = "min_max"
    # Reliability weights per recall channel. Missing channels default to 1.0.
    channel_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "itemcf": 0.47,
            "two_tower": 1.0,
            "lightgcn": 0.92,
            "popularity": 0.5,
        }
    )


class RetrievalFiltersConfig(_Section):
    exclude_interacted: bool = True
    exclude_disliked: bool = True


class PopularityRetrievalConfig(_Section):
    enabled: bool = True
    # Bounded global popularity list used before falling back to the remainder.
    pool_size: int = Field(default=2000, ge=100)


class RetrievalConfig(_Section):
    itemcf: ItemCFConfig = Field(default_factory=ItemCFConfig)
    two_tower: TwoTowerRetrievalConfig = Field(default_factory=TwoTowerRetrievalConfig)
    lightgcn: LightGCNRetrievalConfig = Field(default_factory=LightGCNRetrievalConfig)
    popularity: PopularityRetrievalConfig = Field(
        default_factory=PopularityRetrievalConfig
    )
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    filters: RetrievalFiltersConfig = Field(default_factory=RetrievalFiltersConfig)


# ---------------------------------------------------------------------------
# ranking.yaml
# ---------------------------------------------------------------------------
class DINRankingConfig(_Section):
    checkpoint_path: str = "checkpoints/din_best.pt"
    batch_size: int = Field(default=256, ge=1)
    max_history_length: int = Field(default=50, ge=1)


class DeepFMRankingConfig(_Section):
    checkpoint_path: str = "checkpoints/deepfm_best.pt"
    batch_size: int = Field(default=1024, ge=1)


class RerankConfig(_Section):
    enabled: bool = True
    top_k: int = Field(default=10, ge=1, le=100)
    enable_mmr: bool = True
    mmr_lambda: float = Field(default=0.9, ge=0.0, le=1.0)
    similarity: Literal["category", "embedding"] = "category"
    max_per_category: int = Field(default=5, ge=1)
    max_per_brand: int | None = Field(default=None, ge=1)


class RankingFiltersConfig(_Section):
    exclude_purchased: bool = True
    apply_budget_constraint: bool = True


class RankingConfig(_Section):
    model: Literal["din", "deepfm"] = "din"
    din: DINRankingConfig = Field(default_factory=DINRankingConfig)
    deepfm: DeepFMRankingConfig = Field(default_factory=DeepFMRankingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    filters: RankingFiltersConfig = Field(default_factory=RankingFiltersConfig)


# ---------------------------------------------------------------------------
# agent.yaml
# ---------------------------------------------------------------------------
class AgentLLMConfig(_Section):
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    json_max_attempts: int = Field(default=2, ge=1, le=5)
    tool_max_rounds: int = Field(default=5, ge=1, le=20)


class WorkflowConfig(_Section):
    max_critic_retries: int = Field(default=2, ge=0, le=5)
    max_total_steps: int = Field(default=20, ge=1, le=100)
    enable_checkpointing: bool = True
    checkpoint_db_path: str = "data/processed/agent_checkpoints.sqlite"


class MemoryConfig(_Section):
    short_term_max_turns: int = Field(default=10, ge=1)
    profile_db_path: str = "data/processed/user_profiles.sqlite"
    chroma_dir: str = "indexes/chroma_memory"
    semantic_top_k: int = Field(default=5, ge=1)


class RAGConfig(_Section):
    review_index_dir: str = "indexes/review_vectors"
    collection_name: str = "product_reviews"
    top_k: int = Field(default=5, ge=1)
    rerank_top_n: int = Field(default=3, ge=1)


class AgentConfig(_Section):
    llm: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


# ---------------------------------------------------------------------------
# Aggregate app config
# ---------------------------------------------------------------------------
class AppConfig(BaseModel):
    """Validated view over all YAML configuration files."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    data: DataConfig
    model: ModelConfig
    retrieval: RetrievalConfig
    ranking: RankingConfig
    agent: AgentConfig
    project_root: Path = PROJECT_ROOT

    def resolve_path(self, relative: str | Path) -> Path:
        """Resolve a config path against the project root."""
        path = Path(relative)
        return path if path.is_absolute() else self.project_root / path


# ---------------------------------------------------------------------------
# DeepSeek settings (.env)
# ---------------------------------------------------------------------------
class DeepSeekSettings(BaseSettings):
    """DeepSeek connection settings loaded from environment / ``.env``.

    Field names map to environment variables with the ``DEEPSEEK_`` prefix:
    ``api_key`` -> ``DEEPSEEK_API_KEY``, ``base_url`` -> ``DEEPSEEK_BASE_URL``,
    ``model`` -> ``DEEPSEEK_MODEL``, and so on.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        env_prefix="DEEPSEEK_",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: SecretStr | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # System/environment proxies are ignored by default.  Set DEEPSEEK_PROXY
    # only when an explicit, working HTTP proxy is required.
    proxy: str | None = None

    @field_validator("proxy", mode="before")
    @classmethod
    def _blank_proxy_is_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("proxy")
    @classmethod
    def _valid_proxy(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("DEEPSEEK_PROXY must start with http:// or https://")
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: Any) -> Any:
        if value is None or isinstance(value, SecretStr):
            return value
        text = str(value).strip()
        if not text or text.lower() in {
            "your_api_key",
            "your-api-key",
            "replace_me",
            "changeme",
        }:
            return None
        return text

    @field_validator("model")
    @classmethod
    def _model_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("DEEPSEEK_MODEL must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("DEEPSEEK_BASE_URL must start with http:// or https://")
        return value

    @property
    def has_api_key(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value().strip())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
_SECTION_FILES: dict[str, str] = {
    "data": "data.yaml",
    "model": "model.yaml",
    "retrieval": "retrieval.yaml",
    "ranking": "ranking.yaml",
    "agent": "agent.yaml",
}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration file must contain a mapping: {path}")
    return payload


def load_app_config(
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Load and validate every YAML config file.

    Args:
        config_dir: Directory containing the five YAML files. Relative paths
            are resolved against ``project_root``.
        project_root: Project root used to resolve relative data paths.

    Raises:
        ConfigurationError: if a file is missing, malformed or fails validation.
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    cfg_dir = Path(config_dir) if config_dir is not None else root / "configs"
    if not cfg_dir.is_absolute():
        cfg_dir = root / cfg_dir

    sections: dict[str, Any] = {}
    for section, filename in _SECTION_FILES.items():
        mapping = _load_yaml_mapping(cfg_dir / filename)
        if section not in mapping:
            raise ConfigurationError(
                f"{filename} must contain a top-level '{section}' key"
            )
        value = mapping[section]
        if not isinstance(value, dict):
            raise ConfigurationError(f"'{section}' in {filename} must be a mapping")
        sections[section] = value

    try:
        return AppConfig(**sections, project_root=root)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {cfg_dir}: {exc}") from exc


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Return a cached, validated application config."""
    return load_app_config()


def get_deepseek_settings(env_file: str | Path | None = None) -> DeepSeekSettings:
    """Build DeepSeek settings from environment variables and ``.env``.

    Passing ``env_file`` overrides the default project ``.env`` path (useful in
    tests and deployment).
    """
    kwargs: dict[str, Any] = {}
    if env_file is not None:
        kwargs["_env_file"] = str(env_file)
    return DeepSeekSettings(**kwargs)
