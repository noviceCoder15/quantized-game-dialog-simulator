"""File I/O: .env loading, corpus, model checkpoints and benchmark results."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .core import GPT, CharTokenizer, GPTConfig, clean_text


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_corpus(path: str) -> str:
    """Read the dialogue corpus and run it through the text preprocessing step."""
    corpus_file = Path(path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    return clean_text(corpus_file.read_text(encoding="utf-8"))


def save_checkpoint(path: str, model: GPT, tokenizer: CharTokenizer) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.cfg.to_dict(),
            "chars": tokenizer.chars,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str):
    """Return (model, tokenizer) from a checkpoint written by save_checkpoint."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no model at {path} - run `python -m app.cli train` first"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, CharTokenizer(checkpoint["chars"])


def save_results(path: str, results: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")
