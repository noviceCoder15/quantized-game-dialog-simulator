# Quantized Game Dialogue Simulator

Train a tiny GPT on game dialogue, quantize it (dynamic and static post-training
quantization), then measure what quantization does to model size, output quality
and generation speed. CPU only, command line, no API keys.

## Run it

From this folder, in a terminal inside the workbench (PyTorch is already installed there):

```bash
pip install -r requirements.txt      # only needed if torch/pytest are missing
pytest                               # 1. check everything works
python -m app.cli train              # 2. train on data/dialogue.txt (about a minute on CPU)
python -m app.cli generate           # 3. sample dialogue from the fp32 model
python -m app.cli generate --variant dynamic
python -m app.cli generate --variant static
python -m app.cli simulate           # 4. size / quality / speed comparison
```

Useful options:

```bash
python -m app.cli train --steps 1500 --n-embd 96 --n-layer 3
python -m app.cli generate --prompt "Wizard:" --temperature 1.0 --top-k 20
python -m app.cli simulate --tokens 200 --runs 5
```

Defaults can be changed in a `.env` file (copy `.env.example`).
`simulate` also writes `results/simulation.json`.

## Milestones

| Milestone | Where |
|---|---|
| Dialogue corpus | `data/dialogue.txt` |
| Preprocessing and tokenization | `app/core.py`: `clean_text`, `CharTokenizer` |
| GPT architecture and training | `app/core.py`: `GPT`; `app/llm.py`: `train` |
| Sample dialogue | `app/llm.py`: `generate`; `python -m app.cli generate` |
| Dynamic and static PTQ | `app/llm.py`: `quantize_dynamic_model`, `quantize_static_model` |
| Model size comparison | `app/core.py`: `serialized_size_mb`, `theoretical_size_mb` |
| Simulated inference speed | `app/llm.py`: `tokens_per_second`, `linear_microbenchmark` |

## What `simulate` reports

1. **Measured sizes** of fp32, dynamic and static models (bytes of the saved `state_dict`),
   plus loss and generation speed (tokens/sec) for each.
2. **Theoretical weight storage** at fp32, fp16 and int8 (parameters x bytes per weight).
3. **One Linear operation** with fp32 weights vs int8 weights, on this model's MLP layer and on a
   larger 512x2048 reference layer.

## Things to know

- **Measured size is less than 4x smaller.** Only Linear weights become int8. Embeddings,
  LayerNorm and biases stay float32, and a tiny model has proportionally more of those.
- **Static quantization covers the Linear layers only.** Each Linear is wrapped as
  `quant -> Linear -> dequant` and calibrated on sample batches. Attention maths, softmax,
  LayerNorm, GELU and embeddings stay float32, because eager-mode static quantization does not
  handle them well. If your machine has no supported quantized backend, `simulate` skips that
  variant and says why.
- **Small layers may not get faster.** For tiny matrices, call overhead dominates, so int8 can
  be no faster, or slower, than fp32. The larger reference layer shows where int8 starts to
  pay off. Speed numbers depend on your CPU.
- **Loss is measured on the training corpus.** It tells you how much quality quantization
  costs, not how well the model generalises. A corpus this small is memorised quickly.
- Timing runs one untimed warm-up first, then averages several runs. Sampling is seeded, so
  every variant generates from the same random numbers.

## Layout

```
app/core.py      tokenizer, GPT, size math, timing helper (no file I/O)
app/llm.py       training, generation, quantization, speed measurements
app/storage.py   .env, corpus, checkpoints, results JSON
app/cli.py       train / generate / simulate commands
data/            dialogue corpus
tests/           unit tests and an end-to-end smoke test
```

## CI

`.github/workflows/ci.yml` installs CPU-only PyTorch and runs `pytest`. GitHub only runs
workflows found in the repository root, so if this project lives in a subfolder, copy the file
to the root `.github/workflows/` and set `working-directory` to the project path.
