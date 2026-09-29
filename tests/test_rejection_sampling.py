"""Statistical tests: speculative sampling must emit tokens distributed exactly as p."""
import torch

from src.speculative_decode import expected_tokens_per_step, greedy_verify, rejection_sample

V, TRIALS, TOL = 6, 40_000, 0.015


def _first_token_dist(p, q_row, draft, trials=TRIALS, seed=0):
    """Empirical distribution of the first emitted token for a single draft."""
    g = torch.Generator().manual_seed(seed)
    p_rows = torch.stack([p, p])                     # row 1 = bonus row (irrelevant here)
    q_rows = None if q_row is None else q_row.unsqueeze(0)
    counts = torch.zeros(V)
    for _ in range(trials):
        n_acc, nxt = rejection_sample(p_rows, [draft], q_rows, g)
        counts[draft if n_acc == 1 else nxt] += 1
    return counts / trials


def _tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def test_deterministic_draft_is_exact():
    torch.manual_seed(1)
    p = torch.softmax(torch.randn(V), 0)
    for draft in range(V):
        emp = _first_token_dist(p, None, draft, seed=draft)
        assert _tv(emp, p) < TOL, (draft, emp, p)


def test_draft_model_is_exact():
    torch.manual_seed(2)
    p = torch.softmax(torch.randn(V), 0)
    q = torch.softmax(torch.randn(V), 0)
    g = torch.Generator().manual_seed(3)
    counts = torch.zeros(V)
    for _ in range(TRIALS):
        draft = torch.multinomial(q, 1, generator=g).item()     # draft ~ q
        n_acc, nxt = rejection_sample(torch.stack([p, p]), [draft], q.unsqueeze(0), g)
        counts[draft if n_acc == 1 else nxt] += 1
    assert _tv(counts / TRIALS, p) < TOL


def test_v1_residual_was_biased():
    """Documents the v1 bug: resampling from full p after rejecting x over-weights x."""
    torch.manual_seed(1)
    p = torch.softmax(torch.randn(V), 0)
    x = 0
    v1_px = p[x] + (1 - p[x]) * p[x]                  # accept + re-draw x from full p
    assert v1_px - p[x] > 0.05                        # clearly biased
    emp = _first_token_dist(p, None, x)
    assert abs(emp[x] - p[x]) < TOL                   # v2 is not


def test_greedy_verify():
    rows = torch.tensor([[0, 5, 1.], [9, 0, 0], [0, 0, 7]])
    assert greedy_verify(rows, [1, 0]) == (2, 2)      # all accepted, bonus = argmax(row 2)
    assert greedy_verify(rows, [1, 2]) == (1, 0)      # second rejected, correction = 0
    assert greedy_verify(rows, [0, 0]) == (0, 1)


def test_expected_tokens():
    assert expected_tokens_per_step(0.0, 5) == 1.0
    assert expected_tokens_per_step(1.0, 5) == 6.0
    assert abs(expected_tokens_per_step(0.5, 2) - 1.75) < 1e-9


def test_controller_routes_to_ar_and_reprobes():
    from src.speculative_decode import DynamicLengthController as DLC
    c = DLC(initial_draft_len=4)
    assert c.choose_draft_len() == 4                  # explores before it has data
    for _ in range(DLC.MIN_DRAFT_OBS):
        c.update("draft", accepted=0, proposed=4)     # draft model is useless here
    c.observe_target_step(1.0)
    c.observe_draft_forward(0.5)
    picks = [c.choose_draft_len() for _ in range(DLC.PROBE_EVERY)]
    assert picks[:-1] == [0] * (DLC.PROBE_EVERY - 1)  # routes to AR...
    assert picks[-1] == DLC.MIN_DRAFT_LEN             # ...but periodically re-probes
