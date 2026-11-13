"""Core logic for the Quantized Game Dialogue Simulator.

Text cleaning, a character tokenizer, a small GPT, and size / timing helpers.
Nothing in this module reads or writes files, so it is easy to test.
"""
from __future__ import annotations

import io
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Text preprocessing and tokenization
# --------------------------------------------------------------------------

_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",      # curly single quotes
    "\u201c": '"', "\u201d": '"',      # curly double quotes
    "\u2013": "-", "\u2014": "-",      # en / em dash
    "\u2026": "...",                   # ellipsis
    "\u00a0": " ",                     # non-breaking space
}


def clean_text(text: str) -> str:
    """Normalise punctuation, collapse whitespace and drop blank lines."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line) + "\n"


class CharTokenizer:
    """Character-level tokenizer: every distinct character is one token."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # Characters never seen in training are dropped so prompts cannot crash.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# --------------------------------------------------------------------------
# A small GPT (decoder-only transformer)
# --------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer(
            "mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # Causal mask: a token may only look at itself and earlier tokens.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection
        x = x + self.mlp(self.ln2(x))    # residual connection
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --------------------------------------------------------------------------
# Size math and timing
# --------------------------------------------------------------------------

BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "int8": 1}


def count_parameters(model: nn.Module) -> int:
    """Number of float parameters (use on the fp32 model, not a quantized one)."""
    return sum(p.numel() for p in model.parameters())


def theoretical_size_mb(n_params: int, dtype: str) -> float:
    """Weight storage if every parameter were stored as `dtype`."""
    return n_params * BYTES_PER_PARAM[dtype] / 1024**2


def serialized_size_mb(model: nn.Module) -> float:
    """Real size of the model: bytes of its serialized state_dict."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024**2


def time_fn(fn, warmup: int = 1, runs: int = 5) -> float:
    """Average seconds per call. Warm-up calls are run first and not timed."""
    for _ in range(warmup):
        fn()
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        total += time.perf_counter() - start
    return total / runs
