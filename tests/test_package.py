"""Package-level contracts: imports, exports, errors, metadata."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import pytest

import finax
from finax.errors import (
    DataValidationError,
    FinaxError,
    MissingDependencyError,
    ShapeError,
    require,
)

SUBMODULES = [
    "finax.core",
    "finax.data",
    "finax.diagnostics",
    "finax.evaluation",
    "finax.inference",
    "finax.infrastructure",
    "finax.microstructure",
    "finax.models",
    "finax.processes",
    "finax.utils",
    "finax.visualization",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", SUBMODULES)
def test_all_exports_resolve(name):
    """Every name in __all__ must actually exist. Catches stale exports."""
    module = importlib.import_module(name)
    for export in getattr(module, "__all__", []):
        assert hasattr(module, export), f"{name}.{export} is exported but missing"


def test_lazy_attribute_access():
    assert finax.models is not None
    assert finax.microstructure is not None


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = finax.does_not_exist


def test_version_is_pep440():
    import re

    assert re.match(r"^\d+\.\d+\.\d+", finax.__version__)


def test_dir_lists_submodules():
    listing = dir(finax)
    assert "models" in listing and "microstructure" in listing


class TestErrors:
    def test_hierarchy(self):
        assert issubclass(MissingDependencyError, FinaxError)
        assert issubclass(MissingDependencyError, ImportError)
        assert issubclass(DataValidationError, FinaxError)
        assert issubclass(DataValidationError, ValueError)
        assert issubclass(ShapeError, DataValidationError)

    def test_require_returns_module(self):
        assert require("json").__name__ == "json"

    def test_require_error_names_the_extra(self):
        """A missing optional dep must say what it is and how to install it."""
        pytest.importorskip  # noqa: B018 - documents intent
        with pytest.raises(MissingDependencyError) as info:
            require("definitely_not_a_real_module", purpose="testing")
        message = str(info.value)
        assert "definitely_not_a_real_module" in message
        assert "testing" in message
        assert "pip install" in message

    def test_require_error_maps_known_modules_to_extras(self):
        """Modules that belong to an extra should name the extra, not the module."""
        from finax.errors import _EXTRAS

        assert _EXTRAS["statsmodels"] == "stats"
        assert _EXTRAS["matplotlib"] == "viz"
        assert _EXTRAS["pandas"] == "data"

    def test_shape_errors_are_catchable_as_value_error(self):
        from finax.core import fill_forward

        with pytest.raises(ValueError):
            fill_forward(jnp.ones(3))


class TestRemovedModules:
    """0.1.0 modules that were removed should stay removed."""

    @pytest.mark.parametrize(
        "name",
        [
            "finax.app",
            "finax.nlp",
            "finax.research",
            "finax.modeling",
            "finax.data.eikon",
        ],
    )
    def test_module_is_gone(self, name):
        with pytest.raises(ImportError):
            importlib.import_module(name)


class TestInfrastructure:
    def test_device_summary_keys(self):
        summary = finax.infrastructure.device_summary()
        assert set(summary) == {
            "platform",
            "device_count",
            "devices",
            "default_backend",
            "x64_enabled",
        }

    def test_seed_everything_is_reproducible(self):
        import jax.random as jr

        a = jr.normal(finax.infrastructure.seed_everything(7), (5,))
        b = jr.normal(finax.infrastructure.seed_everything(7), (5,))
        assert bool(jnp.array_equal(a, b))

    def test_reproducibility_report(self):
        report = finax.infrastructure.reproducibility_report()
        assert report["finax"] == finax.__version__
        assert "jax" in report["versions"]

    def test_best_platform(self):
        assert finax.infrastructure.best_platform() in ("cpu", "gpu", "tpu")

    def test_config_roundtrip(self, tmp_path):
        path = tmp_path / "config.json"
        finax.infrastructure.save_config({"lr": 0.01, "steps": 100}, path)
        assert finax.infrastructure.load_config(path) == {"lr": 0.01, "steps": 100}

    def test_load_config_missing_file(self):
        with pytest.raises(DataValidationError, match="not found"):
            finax.infrastructure.load_config("/nonexistent/config.json")

    def test_load_config_unknown_format(self, tmp_path):
        path = tmp_path / "config.xml"
        path.write_text("<config/>")
        with pytest.raises(DataValidationError, match="Unsupported"):
            finax.infrastructure.load_config(path)


class TestLogging:
    def test_namespaced_logger(self):
        assert finax.utils.get_logger("models").name == "finax.models"
        assert finax.utils.get_logger().name == "finax"

    def test_no_duplicate_prefix(self):
        assert finax.utils.get_logger("finax.models").name == "finax.models"

    def test_importing_does_not_configure_root_logging(self):
        """A library must not hijack the application's logging setup."""
        import logging

        assert not logging.getLogger().handlers or all(
            not isinstance(h, logging.StreamHandler) or h.name != "finax"
            for h in logging.getLogger().handlers
        )
