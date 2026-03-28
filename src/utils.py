"""Shared utility functions for IO and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a Python dictionary as formatted JSON."""
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(input_path: Path) -> dict[str, Any]:
    """Load and return JSON content as a dictionary."""
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {input_path}")
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_model(model: Any, output_path: Path) -> None:
    """Persist a trained model artifact using joblib."""
    ensure_directory(output_path.parent)
    joblib.dump(model, output_path)


def load_saved_model(model_path: Path) -> Any:
    """Load a previously saved model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. Train the model first."
        )
    return joblib.load(model_path)


def print_section(title: str) -> None:
    """Print a compact section header for command-line summaries."""
    print(f"\n{'=' * 8} {title} {'=' * 8}")
