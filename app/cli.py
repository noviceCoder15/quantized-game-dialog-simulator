"""Command line interface: train, generate, simulate.

    python -m app.cli train
    python -m app.cli generate --variant dynamic
    python -m app.cli simulate
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from . import core, llm, storage


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def format_table(headers, rows) -> str:
    widths = [
        max([len(str(h))] + [len(str(row[i])) for row in rows])
        for i, h in enumerate(headers)
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join([line, rule] + body)


def _load_data(corpus_path, tokenizer=None):
    text = storage.load_corpus(corpus_path)
    if tokenizer is None:
        tokenizer = core.CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return text, tokenizer, data


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_train(args) -> int:
    torch.manual_seed(args.seed)
    text, tokenizer, data = _load_data(args.corpus)
    if len(data) < 8:
        raise ValueError("corpus is too short to train on")

    block_size = max(2, min(args.block_size, len(data) // 2))
    config = core.GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = core.GPT(config)
    print(f"corpus : {len(text):,} characters, vocabulary {tokenizer.vocab_size}")
    print(f"model  : {core.count_parameters(model):,} parameters, block size {block_size}")

    llm.train(model, data, args.steps, args.batch_size, args.lr, seed=args.seed)
    storage.save_checkpoint(args.model, model, tokenizer)
    print(f"\nsaved {args.model} ({core.serialized_size_mb(model):.2f} MB)")

    print("\nsample dialogue:")
    print(llm.generate_text(model, tokenizer, "Guard:", 150, seed=args.seed))
    return 0


def cmd_generate(args) -> int:
    torch.manual_seed(args.seed)
    model, tokenizer = storage.load_checkpoint(args.model)
    calibration = None
    if args.variant == "static":
        _, _, data = _load_data(args.corpus, tokenizer)
        calibration = llm.make_calibration_batches(data, model.cfg.block_size)
    model = llm.build_variant(args.variant, model, calibration)
    text = llm.generate_text(
        model, tokenizer, args.prompt, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed,
    )
    print(f"[{args.variant}]")
    print(text)
    return 0


def cmd_simulate(args) -> int:
    torch.manual_seed(args.seed)
    model, tokenizer = storage.load_checkpoint(args.model)
    _, _, data = _load_data(args.corpus, tokenizer)
    n_params = core.count_parameters(model)
    calibration = llm.make_calibration_batches(data, model.cfg.block_size)

    # Build the three variants; a variant the machine cannot build is skipped.
    variants = {"fp32": model}
    skipped = {}
    for name in ("dynamic", "static"):
        try:
            variants[name] = llm.build_variant(name, model, calibration)
        except Exception as exc:  # unsupported backend / PyTorch version
            skipped[name] = f"{type(exc).__name__}: {exc}"

    # --- size, quality and generation speed for every variant ---------------
    rows, variant_results = [], {}
    base_size = core.serialized_size_mb(model)
    for name, variant in variants.items():
        size = core.serialized_size_mb(variant)
        loss = llm.evaluate_loss(variant, data)
        speed = llm.tokens_per_second(
            variant, tokenizer, args.prompt, args.tokens, runs=args.runs, seed=args.seed
        )
        variant_results[name] = {
            "size_mb": size,
            "size_ratio_vs_fp32": base_size / size,
            "loss": loss,
            "tokens_per_sec": speed,
        }
        rows.append([name, f"{size:.3f}", f"{base_size / size:.2f}x", f"{loss:.4f}", f"{speed:.1f}"])

    print(f"\nModel: {n_params:,} parameters   (quantized backend: "
          f"{torch.backends.quantized.engine})\n")
    print("Measured on this machine (CPU)")
    print(format_table(["variant", "size MB", "smaller", "loss", "tokens/sec"], rows))
    print("\nloss is measured on the training corpus: it shows how much quality "
          "quantization costs, not generalisation.")

    # --- theoretical weight storage ----------------------------------------
    theory = {d: core.theoretical_size_mb(n_params, d) for d in core.BYTES_PER_PARAM}
    print("\nTheoretical weight storage if every weight used one data type")
    print(format_table(["dtype", "size MB"], [[d, f"{s:.3f}"] for d, s in theory.items()]))
    print("Measured sizes are above the 4x ideal because embeddings, LayerNorm and "
          "biases stay float32.")

    # --- operation-level speed: fp32 weights vs int8 weights ---------------
    cfg = model.cfg
    micro = {
        "this model's MLP layer": llm.linear_microbenchmark(cfg.n_embd, 4 * cfg.n_embd, cfg.block_size),
        "larger reference layer": llm.linear_microbenchmark(512, 2048, 64),
    }
    print("\nOne Linear operation: fp32 weights vs int8 weights")
    print(format_table(
        ["layer", "shape", "fp32 ms", "int8 ms", "speedup"],
        [[k, v["shape"], f"{v['fp32_ms']:.3f}", f"{v['int8_ms']:.3f}", f"{v['speedup']:.2f}x"]
         for k, v in micro.items()],
    ))

    for name, reason in skipped.items():
        print(f"\nskipped '{name}': {reason}")

    storage.save_results(args.results, {
        "parameters": n_params,
        "quantized_engine": torch.backends.quantized.engine,
        "settings": {"prompt": args.prompt, "tokens": args.tokens, "runs": args.runs},
        "theoretical_weight_mb": theory,
        "variants": variant_results,
        "skipped": skipped,
        "linear_microbenchmark": micro,
    })
    print(f"\nresults written to {args.results}")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default=_env("GAME_CORPUS_PATH", "data/dialogue.txt"))
    common.add_argument("--model", default=_env("GAME_MODEL_PATH", "models/dialogue_gpt.pt"))
    common.add_argument("--seed", type=int, default=int(_env("GAME_SEED", "1337")))

    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Train a tiny dialogue GPT, quantize it and simulate its footprint.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", parents=[common], help="train the GPT on the corpus")
    train.add_argument("--steps", type=int, default=800)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--block-size", type=int, default=64)
    train.add_argument("--n-layer", type=int, default=2)
    train.add_argument("--n-head", type=int, default=4)
    train.add_argument("--n-embd", type=int, default=64)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--lr", type=float, default=2e-3)
    train.set_defaults(func=cmd_train)

    gen = sub.add_parser("generate", parents=[common], help="generate dialogue")
    gen.add_argument("--prompt", default="Guard:")
    gen.add_argument("--max-new-tokens", type=int, default=200)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=10)
    gen.add_argument("--variant", choices=["fp32", "dynamic", "static"], default="fp32")
    gen.set_defaults(func=cmd_generate)

    sim = sub.add_parser("simulate", parents=[common],
                         help="compare size and speed of fp32 vs quantized models")
    sim.add_argument("--prompt", default="Guard:")
    sim.add_argument("--tokens", type=int, default=100, help="tokens generated per timing run")
    sim.add_argument("--runs", type=int, default=3, help="timed runs to average")
    sim.add_argument("--results", default=_env("GAME_RESULTS_PATH", "results/simulation.json"))
    sim.set_defaults(func=cmd_simulate)

    return parser


def main(argv=None) -> int:
    storage.load_env()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
