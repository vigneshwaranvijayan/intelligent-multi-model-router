from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from multirouter.domain import ModelSpec, RoutingPolicy


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIROUTER_", env_file=".env", extra="ignore")

    config_dir: Path = Path("configs")
    router: str = "utility"
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=15.0, gt=0)
    failure_threshold: int = Field(default=3, ge=1)
    circuit_reset_seconds: float = Field(default=30.0, gt=0)
    health_check_interval_seconds: float = Field(default=10.0, gt=0)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


def load_model_specs(config_dir: Path) -> list[ModelSpec]:
    raw = _read_yaml(config_dir / "models.yaml")
    items = raw.get("models")
    if not isinstance(items, list) or not items:
        raise ValueError("models.yaml must contain a non-empty 'models' list")

    specs: list[ModelSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping")
        specs.append(
            ModelSpec(
                id=str(item["id"]),
                endpoint=str(item["endpoint"]).rstrip("/"),
                max_context_tokens=int(item["max_context_tokens"]),
                privacy_levels=frozenset(item.get("privacy_levels", ["public"])),
                capabilities=frozenset(item.get("capabilities", ["chat"])),
                input_cost_per_1k=float(item.get("input_cost_per_1k", 0.0)),
                output_cost_per_1k=float(item.get("output_cost_per_1k", 0.0)),
                quality_prior=float(item.get("quality_prior", 0.5)),
                latency_prior_ms=float(item.get("latency_prior_ms", 1000.0)),
                task_affinity={str(k): float(v) for k, v in item.get("task_affinity", {}).items()},
            )
        )
    return specs


def load_policies(config_dir: Path) -> dict[str, RoutingPolicy]:
    raw = _read_yaml(config_dir / "policies.yaml")
    items = raw.get("policies")
    if not isinstance(items, dict) or not items:
        raise ValueError("policies.yaml must contain a non-empty 'policies' mapping")

    return {
        name: RoutingPolicy(
            name=name,
            quality_weight=float(item["quality_weight"]),
            latency_weight=float(item["latency_weight"]),
            cost_weight=float(item["cost_weight"]),
            reliability_weight=float(item["reliability_weight"]),
            minimum_quality=float(item.get("minimum_quality", 0.0)),
            maximum_latency_ms=(
                float(item["maximum_latency_ms"])
                if item.get("maximum_latency_ms") is not None
                else None
            ),
        )
        for name, item in items.items()
    }


class ConfigBundle(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    settings: AppSettings
    models: list[ModelSpec]
    policies: dict[str, RoutingPolicy]


def load_bundle(settings: AppSettings | None = None) -> ConfigBundle:
    effective = settings or AppSettings()
    return ConfigBundle(
        settings=effective,
        models=load_model_specs(effective.config_dir),
        policies=load_policies(effective.config_dir),
    )
