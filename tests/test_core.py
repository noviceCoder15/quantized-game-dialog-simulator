import pytest

torch = pytest.importorskip("torch")

from app import core, llm  # noqa: E402


def tiny_model(vocab_size=20, n_embd=16):
    torch.manual_seed(0)
    cfg = core.GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=n_embd)
    return core.GPT(cfg).eval()


def test_clean_text_normalises_and_drops_blank_lines():
    raw = "Hero:  \u201cHi\u201d \u2014 there\u2019s a  gap\n\n\n  Guard: ok  \n"
    assert core.clean_text(raw) == 'Hero: "Hi" - there\'s a gap\nGuard: ok\n'


def test_tokenizer_round_trip():
    tok = core.CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    assert tok.vocab_size == len(set("hello world"))


def test_tokenizer_drops_unknown_characters():
    tok = core.CharTokenizer.from_text("abc")
    assert tok.decode(tok.encode("a?b!c")) == "abc"


def test_theoretical_int8_is_a_quarter_of_fp32():
    fp32 = core.theoretical_size_mb(1_000_000, "fp32")
    assert core.theoretical_size_mb(1_000_000, "int8") == pytest.approx(fp32 / 4)
    assert core.theoretical_size_mb(1_000_000, "fp16") == pytest.approx(fp32 / 2)


def test_forward_shapes_and_loss():
    model = tiny_model()
    x = torch.randint(0, 20, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, 20)
    assert loss.item() > 0


def test_model_is_causal():
    model = tiny_model()
    x = torch.randint(0, 20, (1, 12))
    changed = x.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 20
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(changed)
    # Changing the last token must not change predictions for earlier positions.
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_generate_returns_requested_length():
    model = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = llm.generate(model, idx, max_new_tokens=7, seed=1)
    assert out.shape == (1, 10)


def test_generation_is_reproducible_with_a_seed():
    model = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    first = llm.generate(model, idx, 10, seed=5)
    second = llm.generate(model, idx, 10, seed=5)
    assert torch.equal(first, second)


def test_dynamic_quantization_shrinks_the_model():
    model = tiny_model(n_embd=64)  # big enough that int8 savings beat fixed overhead
    quantized = llm.quantize_dynamic_model(model)
    assert core.serialized_size_mb(quantized) < core.serialized_size_mb(model)
