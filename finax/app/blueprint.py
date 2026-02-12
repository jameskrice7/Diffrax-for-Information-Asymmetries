"""Blueprint objects for building a Finax-powered analytics app."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataSourceConfig:
    name: str
    kind: str
    refresh_seconds: int = 60


@dataclass(frozen=True)
class ModelServiceConfig:
    model_name: str
    endpoint: str
    supports_jumps: bool = True
    supports_regime_switching: bool = True


@dataclass
class AppBlueprint:
    """Serializable description of the app architecture to implement later."""

    app_name: str
    data_sources: list[DataSourceConfig] = field(default_factory=list)
    model_services: list[ModelServiceConfig] = field(default_factory=list)

    def register_data_source(self, source: DataSourceConfig) -> None:
        self.data_sources.append(source)

    def register_model_service(self, service: ModelServiceConfig) -> None:
        self.model_services.append(service)

    def to_dict(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "data_sources": [s.__dict__ for s in self.data_sources],
            "model_services": [m.__dict__ for m in self.model_services],
        }
