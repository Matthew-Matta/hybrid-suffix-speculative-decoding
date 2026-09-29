"""
Hybrid Speculative Decoding Engine (v2)
=======================================
Five decoding modes:
  - autoregressive : one target forward per token, no speculation
  - specdec        : draft-model speculation, fixed draft length
  - sa_only        : suffix-automaton speculation (zero forward passes to draft)
  - hybrid_fixed   : SA when it matches, otherwise the draft model, fixed length
  - hybrid_dynamic : SA / draft model / AR chosen per step by a cost-aware controller

Correctness
-----------
Verification uses the speculative sampling rule of Leviathan et al. (2023) and
Chen et al. (2023). For draft token x with draft distribution q and target
distribution p:

    accept with probability  min(1, p(x) / q(x))
    on rejection, sample from  normalize(max(0, p - q))

For suffix-automaton drafts q is one-hot at x, so acceptance is p(x) and the
residual is p with x zeroed out and renormalized. In both cases the emitted
token is distributed exactly as p. Greedy decoding (temperature = 0) accepts a
draft iff it equals the target argmax.

Execution model
---------------
Invariant: the target KV cache holds every committed token except the last one,
``pending``. Each step feeds ``[pending] + drafts`` in ONE target forward:
row i of the logits verifies drafts[i], and row n yields the bonus token. After
acceptance the cache is cropped to the accepted prefix and the sampled token
becomes the new ``pending``. So every step - speculative or autoregressive -
costs exactly one target forward.

The draft model is synced lazily: its KV cache only catches up on committed
tokens when the controller actually routes to it.
"""

from __future__ import annotations

import time
from typing import Literal

import torch
import torch.nn.functional as F

from .suffix_automaton import DualSuffixAutomaton
from .utils import GenerationMetrics, MetricsTracker

Mode = Literal["autoregressive", "specdec", "sa_only", "hybrid_fixed", "hybrid_dynamic"]


# ---------------------------------------------------------------------------
# Sampling primitives (pure functions - unit tested in tests/)
# ---------------------------------------------------------------------------

def process_logits(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    """Turn logits [..., V] into the sampling distribution (temperature + top-p).

    The SAME function is used for the target, the draft model and plain AR
    sampling, so p and q are always comparable.
    """
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        cum = sorted_p.cumsum(dim=-1)
        sorted_p = sorted_p.masked_fill((cum - sorted_p) > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def greedy_verify(logit_rows: torch.Tensor, draft_tokens: list[int]) -> tuple[int, int]:
    """Greedy verification. logit_rows: [n+1, V].

    Returns (num_accepted, next_token). One device->host sync for the whole step.
    """
    argmax = logit_rows.argmax(dim=-1).tolist()
    for i, tok in enumerate(draft_tokens):
        if tok != argmax[i]:
            return i, argmax[i]
    return len(draft_tokens), argmax[len(draft_tokens)]


def rejection_sample(
    p_rows: torch.Tensor,
    draft_tokens: list[int],
    q_rows: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Exact speculative sampling.

    Args:
        p_rows:       [n+1, V] target distributions (row n is the bonus row).
        draft_tokens: n proposed tokens.
        q_rows:       [n, V] draft distributions, or None for deterministic
                      (suffix-automaton) drafts, i.e. q one-hot at the draft.
    Returns:
        (num_accepted, next_token) where next_token is the residual sample on
        rejection, or the bonus sample if every draft was accepted.
    """
    dev = p_rows.device
    for i, x in enumerate(draft_tokens):
        p = p_rows[i]
        if q_rows is None:
            accept_prob = p[x]
        else:
            qx = q_rows[i, x]
            accept_prob = torch.clamp(p[x] / qx, max=1.0) if qx > 0 else torch.tensor(1.0, device=dev)
        u = torch.rand((), device=dev, generator=generator)
        if u < accept_prob:
            continue
        # Rejected: sample from the residual normalize(max(0, p - q)).
        if q_rows is None:
            residual = p.clone()
            residual[x] = 0.0           # (p - onehot(x))_+  ==  p with x removed
        else:
            residual = (p - q_rows[i]).clamp_min(0.0)
        z = residual.sum()
        if z <= 0:                      # only reachable if p == q (then we'd never reject)
            residual, z = p, p.sum()
        tok = torch.multinomial(residual / z, 1, generator=generator).item()
        return i, tok
    tok = torch.multinomial(p_rows[len(draft_tokens)], 1, generator=generator).item()
    return len(draft_tokens), tok


def expected_tokens_per_step(alpha: float, k: int) -> float:
    """E[tokens emitted per verify step] with i.i.d. per-token acceptance alpha
    and k drafts: (1 - alpha^(k+1)) / (1 - alpha). Includes the bonus/residual token."""
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def _crop_kv(kv, keep: int):
    """Crop a KV cache to its first `keep` positions (works across transformers versions)."""
    if kv is None:
        return None
    if hasattr(kv, "crop"):
        cur = kv.get_seq_length()
        if cur > keep:
            kv.crop(-(cur - keep))      # negative = drop that many tokens (4.x and 5.x)
        return kv
    return tuple((k[:, :, :keep, :], v[:, :, :keep, :]) for k, v in kv)


# ---------------------------------------------------------------------------
# Dynamic length / source controller
# ---------------------------------------------------------------------------

class DynamicLengthController:
    """
    Per-source rolling acceptance tracking plus a cost model.

    SA drafts cost ~0 to produce, so their length follows a simple threshold
    rule (grow above HIGH_THRESH, shrink below LOW_THRESH) that trims wasted
    verification work when acceptance is poor.

    Draft-model drafts are NOT free. Their length is chosen to maximise
        E[tokens per step] / (1 + k * c)
    where c = (draft forward time) / (target step time) is measured online.
    If the best achievable ratio is <= 1, plain autoregressive decoding wins
    and the controller routes to AR.
    """

    WINDOW_SIZE = 20
    MIN_DRAFT_LEN = 2
    MAX_DRAFT_LEN = 10
    HIGH_THRESH = 0.8
    LOW_THRESH = 0.4
    EMA = 0.9

    def __init__(self, initial_draft_len: int = 4, initial_cost_ratio: float = 0.5) -> None:
        self._draft_len: dict[str, int] = {"SA": initial_draft_len, "draft": initial_draft_len}
        self._window: dict[str, list[tuple[int, int]]] = {"SA": [], "draft": []}
        self._t_target: float | None = None      # seconds per target step
        self._t_draft: float | None = None       # seconds per draft-model forward
        self._init_cost = initial_cost_ratio

    # --- acceptance tracking ---------------------------------------------
    def update(self, source: str, accepted: int, proposed: int) -> None:
        if source not in self._window or proposed == 0:
            return
        window = self._window[source]
        window.append((proposed, accepted))
        if len(window) > self.WINDOW_SIZE:
            window.pop(0)
        rate = self.get_estimated_rate(source)
        if rate > self.HIGH_THRESH:
            self._draft_len[source] = min(self._draft_len[source] + 1, self.MAX_DRAFT_LEN)
        elif rate < self.LOW_THRESH:
            self._draft_len[source] = max(self._draft_len[source] - 1, self.MIN_DRAFT_LEN)

    def get_draft_length(self, source: str) -> int:
        return self._draft_len.get(source, 4)

    def get_estimated_rate(self, source: str) -> float:
        """Rolling acceptance rate; 0.5 neutral prior before any data."""
        window = self._window.get(source, [])
        total_p = sum(p for p, _ in window)
        total_a = sum(a for _, a in window)
        return total_a / total_p if total_p > 0 else 0.5

    # --- cost tracking -----------------------------------------------------
    def observe_target_step(self, seconds: float) -> None:
        self._t_target = seconds if self._t_target is None else self.EMA * self._t_target + (1 - self.EMA) * seconds

    def observe_draft_forward(self, seconds: float) -> None:
        self._t_draft = seconds if self._t_draft is None else self.EMA * self._t_draft + (1 - self.EMA) * seconds

    def cost_ratio(self) -> float:
        if self._t_target and self._t_draft:
            return self._t_draft / self._t_target
        return self._init_cost

    def best_draft_plan(self) -> tuple[int, float]:
        """Return (k, speedup_vs_AR) maximising E[tokens]/(1 + k*c) for the draft model."""
        alpha = self.get_estimated_rate("draft")
        c = self.cost_ratio()
        best_k, best = 0, 1.0
        for k in range(1, self.MAX_DRAFT_LEN + 1):
            s = expected_tokens_per_step(alpha, k) / (1.0 + k * c)
            if s > best:
                best_k, best = k, s
        return best_k, best


# ---------------------------------------------------------------------------
# Core decoder
# ---------------------------------------------------------------------------

class HybridSpecDecoder:
    """Unified speculative decoding engine supporting all five modes (batch size 1)."""

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model=None,
        draft_tokenizer=None,
        device: str = "cuda",
    ) -> None:
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer or target_tokenizer
        self.device = device
        self.target_model.eval()
        if self.draft_model is not None:
            self.draft_model.eval()

    def _sync(self) -> None:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def warmup(self, prompt: str = "Hello", n_warmup: int = 3) -> None:
        """Throwaway forwards so CUDA kernel/JIT setup isn't timed."""
        ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            for _ in range(n_warmup):
                self.target_model(ids, use_cache=False)
                if self.draft_model is not None:
                    self.draft_model(ids, use_cache=False)
        self._sync()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: str, **kwargs) -> tuple[str, GenerationMetrics]:
        """Tokenize, generate, decode. See generate_ids for arguments."""
        ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        new_ids, metrics = self.generate_ids(ids, eos_token_id=self.target_tokenizer.eos_token_id, **kwargs)
        return self.target_tokenizer.decode(new_ids, skip_special_tokens=True), metrics

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        mode: Mode = "hybrid_dynamic",
        num_draft_tokens: int = 4,
        sa_threshold: int = 2,
        sa_max_draft_len: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], GenerationMetrics]:
        """Generate up to max_new_tokens after prompt_ids. Returns (new_token_ids, metrics)."""
        assert len(prompt_ids) >= 1
        tracker = MetricsTracker()
        greedy = temperature == 0.0
        uses_sa = mode in ("sa_only", "hybrid_fixed", "hybrid_dynamic")
        uses_draft = mode in ("specdec", "hybrid_fixed", "hybrid_dynamic") and self.draft_model is not None
        dev = self.device

        self._sync()
        t_start = time.perf_counter()

        dsa = DualSuffixAutomaton()
        if uses_sa:
            dsa.build_from_prompt(list(prompt_ids))
        dlc = DynamicLengthController(initial_draft_len=num_draft_tokens)

        # --- Prefill: target KV holds all prompt tokens except the last ('pending').
        committed: list[int] = list(prompt_ids)
        t_kv = None
        if len(committed) > 1:
            out = self.target_model(torch.tensor([committed[:-1]], device=dev), use_cache=True)
            t_kv = out.past_key_values
            tracker.count_target_forward()
        pending = committed[-1]

        d_kv = None          # draft-model KV cache
        d_synced = 0         # number of committed tokens whose KV is in d_kv
        n_prompt = len(prompt_ids)
        first_token_recorded = False

        while len(committed) - n_prompt < max_new_tokens:
            remaining = max_new_tokens - (len(committed) - n_prompt)
            max_k = remaining - 1            # each step also emits one sampled token

            # ---------------- choose source + draft length -----------------
            source, k = "autoregressive", 0
            sa_drafts: list[int] = []
            if uses_sa and max_k > 0:
                sa_len = sa_max_draft_len
                if mode == "hybrid_dynamic":
                    sa_len = min(dlc.get_draft_length("SA"), sa_max_draft_len)
                context = committed[-64:]
                sa_drafts, sa_match = dsa.query(context, max_draft_len=sa_len, temperature=temperature)
                if sa_match >= sa_threshold and sa_drafts:
                    source, k = "SA", min(len(sa_drafts), max_k)
            if source == "autoregressive" and uses_draft and max_k > 0:
                if mode == "hybrid_dynamic":
                    best_k, gain = dlc.best_draft_plan()
                    if best_k > 0:
                        source, k = "draft", min(best_k, max_k)
                else:
                    source, k = "draft", min(num_draft_tokens, max_k)

            # ---------------- produce drafts -------------------------------
            drafts: list[int] = []
            q_rows = None
            if source == "SA":
                drafts = sa_drafts[:k]
            elif source == "draft":
                t0 = time.perf_counter()
                catch_up = committed[d_synced:]          # always includes `pending`
                out = self.draft_model(torch.tensor([catch_up], device=dev), past_key_values=d_kv, use_cache=True)
                d_kv, logit = out.past_key_values, out.logits[0, -1]
                d_synced = len(committed)
                q_list = []
                for i in range(k):
                    if greedy:
                        tok = int(logit.argmax())
                    else:
                        q = process_logits(logit, temperature, top_p)
                        tok = torch.multinomial(q, 1, generator=generator).item()
                        q_list.append(q)
                    drafts.append(tok)
                    if i < k - 1:
                        out = self.draft_model(torch.tensor([[tok]], device=dev), past_key_values=d_kv, use_cache=True)
                        d_kv, logit = out.past_key_values, out.logits[0, -1]
                n_draft_fwd = k                          # catch-up + (k-1) single-token steps
                tracker.count_draft_forward(n_draft_fwd)
                self._sync()
                dlc.observe_draft_forward((time.perf_counter() - t0) / n_draft_fwd)
                if not greedy:
                    q_rows = torch.stack(q_list)

            # ---------------- ONE target forward: [pending] + drafts -------
            t0 = time.perf_counter()
            base = len(committed) - 1                    # tokens currently in t_kv
            inp = torch.tensor([[pending] + drafts], device=dev)
            out = self.target_model(inp, past_key_values=t_kv, use_cache=True)
            t_kv = out.past_key_values
            rows = out.logits[0]                         # [n+1, V]
            tracker.count_target_forward()

            if greedy:
                n_acc, nxt = greedy_verify(rows, drafts)
            else:
                p_rows = process_logits(rows, temperature, top_p)
                n_acc, nxt = rejection_sample(p_rows, drafts, q_rows, generator)
            dlc.observe_target_step(time.perf_counter() - t0)   # .item()/.tolist() above already synced

            # ---------------- commit -----------------------------------------
            t_kv = _crop_kv(t_kv, base + 1 + n_acc)       # keep pending + accepted drafts
            if source == "draft":
                # draft KV holds committed + drafts[:k-1]; keep only the accepted ones
                d_synced = d_synced + min(n_acc, k - 1)
                d_kv = _crop_kv(d_kv, d_synced)

            new_tokens = drafts[:n_acc] + [nxt]
            if source == "autoregressive":
                tracker.record_draft_attempt("autoregressive", proposed=0, accepted=0, draft_len=0)
            else:
                tracker.record_draft_attempt(source, proposed=len(drafts), accepted=n_acc, draft_len=len(drafts))
                if mode == "hybrid_dynamic":
                    dlc.update(source, accepted=n_acc, proposed=len(drafts))

            hit_eos = False
            for tok in new_tokens:
                committed.append(tok)
                if uses_sa:
                    dsa.extend(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    hit_eos = True
                    break
            pending = committed[-1]

            if not first_token_recorded:
                self._sync()
                tracker.record_ttft(time.perf_counter() - t_start)
                first_token_recorded = True
            if hit_eos:
                break

        self._sync()
        total_time = time.perf_counter() - t_start
        new_ids = committed[n_prompt:]
        metrics = tracker.finalize(total_tokens=len(new_ids), total_time=total_time)
        return new_ids, metrics
