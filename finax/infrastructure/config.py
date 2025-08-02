"""Configuration utilities for Finax."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
