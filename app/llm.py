"""Training, generation, evaluation and quantization for the dialogue GPT."""
from __future__ import annotations

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qconfig,
    prepare,
    quantize_dynamic,
)

from .core import time_fn

# PyTorch warns about reduce_range inside its own default qconfig; it is harmless.
warnings.filterwarnings("ignore", message=".*reduce_range.*")


# --------------------------------------------------------------------------
# Data batches
# --------------------------------------------------------------------------

def get_batch(data, block_size, batch_size, generator=None):
    """Random (input, target) batch; target is the input shifted by one token."""
    if len(data) <= block_size + 1:
        raise ValueError("corpus is too short for this block size")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def make_calibration_batches(data, block_size, n_batches=16, batch_size=8, seed=0):
    """Fixed sample batches used by static quantization to learn activation ranges."""
    generator = torch.Generator().manual_seed(seed)
    return [get_batch(data, block_size, batch_size, generator)[0] for _ in range(n_batches)]


# --------------------------------------------------------------------------
# Training, evaluation, generation
# --------------------------------------------------------------------------

def train(model, data, steps, batch_size, lr, log_every=50, seed=1337):
    """Train with AdamW on random batches. Returns the list of step losses."""
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(1, steps + 1):
        x, y = get_batch(data, model.cfg.block_size, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step {step:>5}/{steps}  loss {loss.item():.4f}")
    model.eval()
    return losses


@torch.no_grad()
def evaluate_loss(model, data, batches=10, batch_size=16, seed=0):
    """Average cross-entropy on fixed batches (same batches for every variant)."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, model.cfg.block_size, batch_size, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=None, seed=None):
    """Sample `max_new_tokens` tokens after `idx` (shape [1, T])."""
    model.eval()
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature=0.8, top_k=10, seed=None):
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = generate(model, idx, max_new_tokens, temperature, top_k, seed)
    return tokenizer.decode(out[0].tolist())


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------

def select_engine() -> str:
    """Pick a quantized backend this machine supports (fbgemm = x86, qnnpack = ARM)."""
    supported = torch.backends.quantized.supported_engines
    for name in ("fbgemm", "x86", "qnnpack"):
        if name in supported:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(f"no quantized backend available (supported: {supported})")


def quantize_dynamic_model(model):
    """Dynamic PTQ: Linear weights -> int8 now, activations quantized at run time."""
    select_engine()
    return quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8, inplace=False)


class StaticLinear(nn.Module):
    """quant -> Linear -> dequant, so a single Linear can be statically quantized
    while the rest of the transformer (attention maths, LayerNorm, softmax,
    embeddings) stays in float32."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.quant = QuantStub()
        self.linear = linear
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.linear(self.quant(x)))


def _wrap_linears(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, StaticLinear(child))
        else:
            _wrap_linears(child)


def quantize_static_model(model, calibration_batches):
    """Static PTQ: observers record activation ranges on sample data (calibration),
    then Linear layers are converted to int8 with fixed scale / zero_point."""
    engine = select_engine()
    quantized = copy.deepcopy(model).eval()
    _wrap_linears(quantized)
    qconfig = get_default_qconfig(engine)
    # Only the wrapped Linears get a qconfig; LayerNorm / Embedding / GELU stay float.
    for module in quantized.modules():
        if isinstance(module, StaticLinear):
            module.qconfig = qconfig
    prepare(quantized, inplace=True)
    with torch.no_grad():
        for x in calibration_batches:     # calibration pass
            quantized(x)
    convert(quantized, inplace=True)
    return quantized


def build_variant(name, model, calibration_batches=None):
    """Return the model as 'fp32', 'dynamic' or 'static'."""
    if name == "fp32":
        return model.eval()
    if name == "dynamic":
        return quantize_dynamic_model(model)
    if name == "static":
        if not calibration_batches:
            raise ValueError("static quantization needs calibration batches")
        return quantize_static_model(model, calibration_batches)
    raise ValueError(f"unknown variant '{name}' (use fp32, dynamic or static)")


# --------------------------------------------------------------------------
# Speed measurements
# --------------------------------------------------------------------------

def tokens_per_second(model, tokenizer, prompt, n_tokens, warmup=1, runs=3, seed=0):
    """Generation throughput. One untimed warm-up run, then the average of `runs`."""
    def run():
        generate_text(model, tokenizer, prompt, n_tokens, seed=seed)

    return n_tokens / time_fn(run, warmup=warmup, runs=runs)


def linear_microbenchmark(d_in, d_out, rows, warmup=3, runs=20):
    """Same Linear op with fp32 weights vs dynamically quantized int8 weights."""
    select_engine()
    fp32 = nn.Sequential(nn.Linear(d_in, d_out)).eval()
    int8 = quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8, inplace=False)
    x = torch.randn(rows, d_in)
    with torch.no_grad():
        t_fp32 = time_fn(lambda: fp32(x), warmup=warmup, runs=runs)
        t_int8 = time_fn(lambda: int8(x), warmup=warmup, runs=runs)
    return {
        "shape": f"{d_in}x{d_out}, {rows} rows",
        "fp32_ms": t_fp32 * 1000,
        "int8_ms": t_int8 * 1000,
        "speedup": t_fp32 / t_int8 if t_int8 > 0 else float("nan"),
    }
