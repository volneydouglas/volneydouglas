"""
Configuration settings for the Identity Framing experiment.
API keys are loaded from environment variables with placeholder fallbacks.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    provider: str = "anthropic"
    model_id: str = "claude-sonnet-4-20250514"
    temperature: float = 1.0
    max_tokens: int = 1024


@dataclass
class ExperimentConfig:
    rounds_per_session: int = 10
    sessions_per_condition: int = 30
    random_seed: int = 42
    output_dir: str = "results"
    data_dir: str = "data"


@dataclass
class Settings:
    # API keys — set via environment variables before running
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "ANTHROPIC_API_KEY", "sk-ant-PLACEHOLDER-set-your-key"
        )
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_API_KEY", "sk-PLACEHOLDER-set-your-key"
        )
    )

    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @property
    def has_valid_keys(self) -> bool:
        return "PLACEHOLDER" not in self.anthropic_api_key

    def validate(self) -> None:
        if not self.has_valid_keys:
            raise ValueError(
                "API keys not configured. Set ANTHROPIC_API_KEY environment variable.\n"
                "  export ANTHROPIC_API_KEY='your-key-here'"
            )
