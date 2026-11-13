import json

import pytest

pytest.importorskip("torch")

from app import cli  # noqa: E402


def test_train_generate_simulate_end_to_end(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Guard: Halt!\nHero: Hello there.\n" * 20, encoding="utf-8")
    model_path = tmp_path / "model.pt"
    results_path = tmp_path / "results.json"
    paths = ["--corpus", str(corpus), "--model", str(model_path)]

    train_args = ["train", *paths, "--steps", "5", "--batch-size", "4", "--block-size", "16",
                  "--n-layer", "1", "--n-head", "2", "--n-embd", "16"]
    assert cli.main(train_args) == 0
    assert model_path.exists()

    assert cli.main(["generate", *paths, "--variant", "dynamic", "--max-new-tokens", "10"]) == 0

    simulate_args = ["simulate", *paths, "--tokens", "5", "--runs", "1",
                     "--results", str(results_path)]
    assert cli.main(simulate_args) == 0

    results = json.loads(results_path.read_text())
    assert "fp32" in results["variants"]
    assert "dynamic" in results["variants"]
    assert results["variants"]["dynamic"]["size_mb"] < results["variants"]["fp32"]["size_mb"] * 1.5


def test_generate_without_a_model_reports_an_error(tmp_path):
    missing = tmp_path / "nope.pt"
    assert cli.main(["generate", "--model", str(missing)]) == 1
