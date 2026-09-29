"""Greedy speculative decoding must reproduce autoregressive output token-for-token.

Uses tiny randomly initialised Qwen2 models in float64 on CPU (no downloads),
so any mismatch is a logic bug rather than fp16 kernel non-determinism.
"""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.speculative_decode import HybridSpecDecoder

MODES = ["specdec", "sa_only", "hybrid_fixed", "hybrid_dynamic"]


def _tiny(seed, layers):
    cfg = Qwen2Config(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).double().eval()


def _perturbed_copy(model, scale, seed):
    """A 'draft' that mostly agrees with the target, so drafts are partially accepted."""
    import copy
    d = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in d.parameters():
            prm.add_(scale * prm.std() * torch.randn(prm.shape, generator=g, dtype=prm.dtype))
    return d


@pytest.fixture(scope="module", params=["independent_draft", "close_draft"])
def decoder(request):
    target = _tiny(0, 3)
    draft = _tiny(1, 1) if request.param == "independent_draft" else _perturbed_copy(target, 0.1, 7)
    return HybridSpecDecoder(target, None, draft_model=draft, device="cpu")


PROMPTS = [
    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],                       # repetitive: SA fires a lot
    [5, 9, 11, 7, 30, 2, 17, 40, 8],                          # generic
    [7],                                                      # single-token prompt
]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("mode", MODES)
def test_greedy_matches_autoregressive(decoder, prompt, mode):
    ref, _ = decoder.generate_ids(prompt, max_new_tokens=40, mode="autoregressive", temperature=0.0)
    out, m = decoder.generate_ids(prompt, max_new_tokens=40, mode=mode, temperature=0.0)
    assert out == ref
    assert len(out) == 40


def test_one_target_forward_per_step(decoder):
    out, m = decoder.generate_ids(PROMPTS[0], max_new_tokens=40, mode="sa_only", temperature=0.0)
    steps = len(m.source_history)
    assert m.target_forwards == steps + 1                     # +1 for prefill
    assert m.tokens_per_target_forward >= 1.0


def test_sampled_modes_run(decoder):
    g = torch.Generator().manual_seed(0)
    for mode in MODES:
        out, _ = decoder.generate_ids(PROMPTS[0], max_new_tokens=30, mode=mode, temperature=1.0, generator=g)
        assert len(out) == 30


def test_close_draft_gets_partial_acceptance():
    target = _tiny(0, 3)
    d = HybridSpecDecoder(target, None, draft_model=_perturbed_copy(target, 0.1, 7), device="cpu")
    ref, _ = d.generate_ids(PROMPTS[1], max_new_tokens=60, mode="autoregressive")
    out, m = d.generate_ids(PROMPTS[1], max_new_tokens=60, mode="specdec")
    assert out == ref
    assert 0.1 < m.draft_acceptance_rate < 1.0      # accept AND reject paths both exercised
