#!/usr/bin/python
from __future__ import annotations

"""Trainable ``<pause>``-token generative recommender -- the *training* track of PauseRec.

Completes "PauseRec: Implicit Reasoning for LLM-based Generative Recommendation"
(He et al., arXiv:2606.14142) for ``CONCEPT:KG-2.93``. The sibling module
:mod:`agent_utilities.knowledge_graph.retrieval.generative_recommender` adapts
PauseRec's *idea* at inference time with a fixed budget of deterministic latent
refinement steps. That captures the qualitative behaviour but **not** the actual
mechanism the paper introduces: a small run of ``<pause>`` token embeddings that
are **trainable parameters optimised by gradient descent on the next-item
objective** -- giving the model latent computation steps that bridge the
world-knowledge (history/text) side and the item (SID) side, with *no* rationale
supervision. This module implements that real mechanism in torch, at tiny / CPU
scale, so the parity claim is grounded in a trained model rather than a heuristic.

What PauseRec actually does (and what we reproduce):

* Items are generated from a learned table. A user's history is encoded, then a
  fixed number ``n_pause_tokens`` of *learned* pause vectors are inserted before
  the next-item readout. Those pause vectors are ordinary ``nn.Parameter`` rows
  with no input dependence -- the model is free to use them as scratch latent
  computation. They are optimised **only** by the next-item cross-entropy loss;
  there is no decoded rationale and no auxiliary objective (PauseRec's core
  design: implicit, not explicit, reasoning).
* Because the pause tokens add depth/capacity in a *non-linear* path between the
  history encoding and the item readout, they let the model capture next-item
  rules that depend on a *non-linear interaction* of the history -- precisely the
  regime where a plain linear readout struggles. We construct the toy task around
  exactly such a rule (a parity / XOR-like interaction of two latent history
  factors) so the with-vs-without-pause ablation is meaningful.

torch is an **optional** dependency: the import is guarded so this module imports
cleanly without torch (``is_available()`` then returns ``False`` and constructing
:class:`PauseTokenRecommender` raises). Everything is seeded (torch + numpy) so
results are deterministic and stable across runs. Models are intentionally tiny
(a few hundred parameters, a couple hundred training steps) and train in well
under a second on CPU.

Layer contract: a pure L2 retrieval helper. No I/O, no network, no upward
dependencies; numpy + (optional) torch only.
"""

from dataclasses import dataclass

import numpy as np

try:  # torch is an optional, heavy dependency -- guard like the embedding utils do.
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-free installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

__all__ = ["is_available", "TrainResult", "PauseTokenRecommender"]


def is_available() -> bool:
    """Return ``True`` iff torch is importable, so the trainable mechanism can run.

    The module always imports; callers gate on this before constructing a
    :class:`PauseTokenRecommender`, mirroring the optional-dependency pattern used
    throughout the framework's embedding utilities.
    """
    return _TORCH_AVAILABLE


@dataclass
class TrainResult:
    """Outcome of training a :class:`PauseTokenRecommender`.

    Attributes:
        final_loss: Mean next-item cross-entropy after the last epoch (lower is
            better; should fall well below ``initial_loss`` once the model learns).
        initial_loss: Mean next-item cross-entropy before any optimisation step.
        recall_at_k_with_pause: Recall@k on the training sequences with the
            learned pause tokens active -- the full PauseRec mechanism.
        recall_at_k_without_pause: Recall@k on the same sequences with the pause
            tokens *zeroed out* (ablation). The gap ``with - without`` is
            PauseRec's central claim: the latent pause computation helps.
        n_pause_tokens: Number of trainable pause vectors used (reflected back so
            callers/tests can confirm the configured budget).
    """

    final_loss: float
    initial_loss: float
    recall_at_k_with_pause: float
    recall_at_k_without_pause: float
    n_pause_tokens: int


class PauseTokenRecommender:
    """Trainable-pause-token generative recommender (CONCEPT:KG-2.93, PauseRec mechanism).

    A small torch model: a learned item-embedding table, a user-history encoder
    (mean of history item embeddings), and ``n_pause_tokens`` TRAINABLE pause
    vectors inserted before the next-item readout. The pause tokens give the model
    latent computation that bridges the history (text / world-knowledge side) and
    the item / SID side, optimised ONLY by the next-item cross-entropy objective --
    exactly PauseRec's design (no rationale supervision). torch-optional.

    The readout is the sum of two paths, mirroring "generate directly" vs.
    "generate after a run of pause tokens": a shallow *linear* map from the
    mean-pooled history to item logits, plus a pause path in which each pause
    vector gates a non-linear hidden unit (``relu(<history, pause_j>)``) whose
    contribution is added to the logits. Zeroing the pause vectors (the ablation)
    zeroes every hidden pre-activation, collapsing the model to the linear direct
    path -- which cannot represent a *non-linear interaction* of the history. So
    on a task whose next item depends on exactly such an interaction the ablated
    model ranks worse, reproducing the paper's finding that the latent pause steps
    help.
    """

    def __init__(
        self, n_items: int, dim: int = 16, n_pause_tokens: int = 4, seed: int = 0
    ) -> None:
        """Construct a tiny trainable pause-token recommender.

        Args:
            n_items: Size of the item catalog (number of generable items).
            dim: Width of the item-embedding table and the history encoding.
            n_pause_tokens: Number of trainable ``<pause>`` vectors inserted
                before the next-item readout. ``0`` reduces the model to a plain
                history-only readout (the degenerate no-pause baseline).
            seed: Seeds torch and numpy so construction and training are fully
                deterministic across runs.

        Raises:
            RuntimeError: If torch is not available (gate on :func:`is_available`).
            ValueError: On non-positive ``n_items`` / ``dim`` or negative
                ``n_pause_tokens``.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover - torch-free installs only
            raise RuntimeError(
                "torch is required for PauseTokenRecommender; check is_available()"
            )
        if n_items < 2:
            raise ValueError(f"n_items must be >= 2, got {n_items}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_pause_tokens < 0:
            raise ValueError(f"n_pause_tokens must be >= 0, got {n_pause_tokens}")

        self.n_items = n_items
        self.dim = dim
        self.n_pause_tokens = n_pause_tokens
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Learned item-embedding table: rows are item generators / scorers.
        self._item_emb = nn.Embedding(n_items, dim)
        # n_pause_tokens TRAINABLE pause vectors -- input-independent scratch space.
        # nn.Parameter (not Embedding) because there is exactly one shared run of
        # pause tokens, used for every prediction, exactly as PauseRec inserts a
        # fixed <pause> run before SID generation.
        self._pause = nn.Parameter(torch.zeros(max(n_pause_tokens, 0), dim))
        if n_pause_tokens > 0:
            nn.init.normal_(self._pause, std=0.3)

        # Readout is the SUM of two paths to item logits, mirroring PauseRec's
        # "direct generation vs. generation after a run of pause tokens":
        #
        #   * Direct path (``_direct``): a *linear* map history -> item logits.
        #     This is the shallow, no-extra-computation route. It alone cannot
        #     represent a non-linear (XOR/parity) interaction of the history.
        #   * Pause path: each pause vector gates a non-linear hidden unit. Unit
        #     ``j`` fires on ``relu(<history, pause_j>)`` and contributes its row
        #     of ``_pause_out`` to the logits. This is the latent computation the
        #     pause tokens unlock -- the depth that lets the model solve the
        #     non-linear rule. Zeroing the pause vectors zeroes every hidden
        #     pre-activation, so the entire non-linear branch vanishes and the
        #     model collapses to the linear direct path: a faithful ablation of
        #     "remove the pause tokens" that genuinely costs accuracy.
        self._direct = nn.Linear(dim, n_items)
        self._pause_out = nn.Linear(max(n_pause_tokens, 1), n_items, bias=False)

        self._modules_list = [self._item_emb, self._direct, self._pause_out]
        nn.init.zeros_(self._direct.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _encode_history(self, history: list[int]) -> torch.Tensor:
        """Mean-pool the item embeddings of a history prefix into one vector.

        An empty history yields a zero vector of width ``dim`` so the readout is
        always well-defined (cold-start).
        """
        if not history:
            return torch.zeros(self.dim)
        idx = torch.tensor(history, dtype=torch.long)
        return self._item_emb(idx).mean(dim=0)

    def _logits(self, history: list[int], *, use_pause: bool) -> torch.Tensor:
        """Next-item logits for a history: linear direct path + pause-token path.

        Args:
            history: Item-id prefix (the context).
            use_pause: When ``False`` the learned pause vectors are replaced by
                zeros, which zeroes every pause hidden pre-activation and so
                removes the entire non-linear branch -- the ablation that strips
                out the latent pause computation while every other weight stays
                identical. The model then falls back to the linear direct path.
        """
        hist = self._encode_history(history).unsqueeze(0)  # (1, dim)
        return self._logits_from_hist(hist, use_pause=use_pause).squeeze(0)

    def _logits_from_hist(self, hist: torch.Tensor, *, use_pause: bool) -> torch.Tensor:
        """Batched next-item logits from a ``(batch, dim)`` history matrix.

        Vectorises the two-path readout over a whole batch so a full-batch epoch
        is one set of tensor ops (not a python loop over examples), keeping
        training fast. ``use_pause`` carries the same ablation semantics as
        :meth:`_logits`.
        """
        logits = self._direct(hist)  # (batch, n_items) -- shallow linear route
        if self.n_pause_tokens > 0:
            pause = self._pause if use_pause else torch.zeros_like(self._pause)
            # Per example, each pause vector j gates relu(<history, pause_j>): the
            # latent computation a single forward "pause step" performs.
            hidden = torch.relu(hist @ pause.t())  # (batch, n_pause_tokens)
            logits = logits + self._pause_out(hidden)
        return logits

    def _encode_histories(self, histories: list[list[int]]) -> torch.Tensor:
        """Mean-pool a list of history prefixes into a ``(batch, dim)`` matrix."""
        return torch.stack([self._encode_history(h) for h in histories])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _all_parameters(self) -> list[torch.Tensor]:
        """Collect every trainable tensor (table, pause vectors, head) for the optimiser."""
        params: list[torch.Tensor] = []
        for module in self._modules_list:
            params.extend(module.parameters())
        if self.n_pause_tokens > 0:
            params.append(self._pause)
        return params

    def fit(
        self,
        sequences: list[list[int]],
        *,
        epochs: int = 200,
        lr: float = 0.05,
        k: int = 5,
    ) -> TrainResult:
        """Train the pause tokens and readout on the next-item objective.

        Each sequence is split into a context prefix (all items but the last) and
        a target (the last item). The model is trained with cross-entropy over the
        item vocabulary -- the *only* signal, with no rationale supervision, as in
        PauseRec. Training is full-batch and seeded, so it is deterministic.

        Args:
            sequences: Item-id histories; each needs length >= 2 (context+target).
            epochs: Number of full-batch gradient steps (kept small for speed).
            lr: Adam learning rate.
            k: Cutoff for the reported Recall@k ablation.

        Returns:
            A :class:`TrainResult` carrying the loss trajectory and the
            with-vs-without-pause Recall@k ablation that demonstrates the pause
            tokens help.

        Raises:
            ValueError: On an empty corpus, a too-short sequence, or ``k < 1``.
        """
        if not sequences:
            raise ValueError("fit() requires a non-empty sequence corpus")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        contexts: list[list[int]] = []
        targets: list[int] = []
        for seq in sequences:
            if len(seq) < 2:
                raise ValueError("each sequence needs length >= 2 (context+target)")
            if any(not (0 <= i < self.n_items) for i in seq):
                raise ValueError("sequence contains an out-of-range item id")
            contexts.append(list(seq[:-1]))
            targets.append(seq[-1])

        torch.manual_seed(self.seed)
        target_t = torch.tensor(targets, dtype=torch.long)
        optim = torch.optim.Adam(self._all_parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        def epoch_loss() -> torch.Tensor:
            hist = self._encode_histories(contexts)  # (batch, dim), re-pooled
            logits = self._logits_from_hist(hist, use_pause=True)
            return loss_fn(logits, target_t)

        with torch.no_grad():
            initial_loss = float(epoch_loss().item())

        final_loss = initial_loss
        for _ in range(epochs):
            optim.zero_grad()
            loss = epoch_loss()
            loss.backward()
            optim.step()
            final_loss = float(loss.item())

        recall_with = self._recall_at_k(contexts, targets, k=k, use_pause=True)
        recall_without = self._recall_at_k(contexts, targets, k=k, use_pause=False)
        return TrainResult(
            final_loss=final_loss,
            initial_loss=initial_loss,
            recall_at_k_with_pause=recall_with,
            recall_at_k_without_pause=recall_without,
            n_pause_tokens=self.n_pause_tokens,
        )

    def _recall_at_k(
        self, contexts: list[list[int]], targets: list[int], *, k: int, use_pause: bool
    ) -> float:
        """Fraction of contexts whose true target lands in the top-``k`` logits."""
        if not contexts:
            return 0.0
        kk = min(k, self.n_items)
        with torch.no_grad():
            hist = self._encode_histories(contexts)
            logits = self._logits_from_hist(hist, use_pause=use_pause)
            topk = torch.topk(logits, kk, dim=1).indices  # (batch, kk)
            tgt = torch.tensor(targets, dtype=torch.long).unsqueeze(1)
            hits = int((topk == tgt).any(dim=1).sum().item())
        return hits / len(contexts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(self, history: list[int], *, k: int = 5) -> list[int]:
        """Return the top-``k`` next-item ids for a history (pause tokens active).

        Args:
            history: The user's item-id prefix.
            k: Number of recommendations.

        Returns:
            ``k`` distinct valid item ids, best-first, scored by the trained model
            with the learned pause tokens in play.

        Raises:
            ValueError: If ``k < 1``.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        kk = min(k, self.n_items)
        with torch.no_grad():
            logits = self._logits(list(history), use_pause=True)
            return torch.topk(logits, kk).indices.tolist()
