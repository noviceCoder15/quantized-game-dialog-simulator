# Quantized Game Dialogue Simulator

This project guides game developers in building a small, quantized GPT for in-game dialogue. You will simulate its memory footprint and generation speed, providing crucial insights for resource-constrained devices.

## What you are building

**Goal.** You will build a small, quantized GPT model trained on game dialogue, demonstrating its memory footprint and generation speed for resource-constrained devices.

**Done when.** The project is complete when you have trained a small GPT on custom game dialogue, applied quantization, and measured its impact on model size and simulated performance.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small game dialogue text corpus.**  
      Hint: Apply text preprocessing and tokenization techniques for your corpus.
- [ ] **2. Implement a basic GPT architecture and train it on the corpus.**  
      Hint: Use the `Build GPT from Scratch` drill for model architecture and training.
- [ ] **3. Generate sample dialogue from the trained GPT.**  
      Hint: Employ the generation methods from the `Build GPT from Scratch` drill.
- [ ] **4. Apply dynamic and static post-training quantization to the model.**  
      Hint: Refer to the `Shrink a Neural Network with Quantisation` drill for quantization methods.
- [ ] **5. Measure and compare the original and quantized model sizes.**  
      Hint: Utilize the `Shrink a Neural Network with Quantisation` drill's techniques for model size comparison.
- [ ] **6. Simulate inference speed by comparing operations on original versus quantized weights.**  
      Hint: Adapt the simulation approaches from the `Shrink a Neural Network with Quantisation` drill for performance comparison.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
