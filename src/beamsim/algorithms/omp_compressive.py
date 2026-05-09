"""Compressive beam alignment via Orthogonal Matching Pursuit (OMP).

Based on: Marzi, Ramasamy, Madhow (2016) — "Compressive channel estimation
and tracking for large arrays in mm-wave picocells."

Idea: the beamspace channel is sparse — only a few (k, l) pairs carry
significant energy.  At each step one measurement y_(k,l) = w_k^H H f_l x
is collected.  Every ``measurements_per_solve`` steps the buffer of recent
measurements is stacked into a linear system  y = A s + n, where
    - y ∈ C^H  (H measurements)
    - A[i, :] = kron(f_l_i, conj(w_k_i)) ∈ C^{K*L}  — the sensing row for
      the i-th measurement, expressed in the flattened codebook outer-product
      (beamspace) basis.  This is equivalent to vec(w_k_i w_{k_i}^H H F) for
      a single-path channel.
    - s ∈ C^{K*L}  — sparse beamspace channel vector.

OMP greedy loop (rolled without sklearn):
  Initialise residual r = y.
  For t = 1..sparsity:
    i* = argmax |A^H r|          — column most correlated with residual
    Add i* to support set S.
    s_S = lstsq(A[:, S], y)      — regress y onto current support.
    r = y - A[:, S] @ s_S        — update residual.
  The OBP is argmax_{k,l} |s[k*L+l]|.
  Between solves, random (k, l) pairs populate the buffer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class OMPCompressive(Algorithm):
    """Compressive beam alignment via a rolled OMP solver.

    Parameters
    ----------
    measurements_per_solve:
        Number of measurements H collected before each OMP solve.
    sparsity:
        Maximum number of non-zero components in the beamspace channel (K_s).
    """

    name = "omp_compressive"

    def __init__(self, measurements_per_solve: int = 8, sparsity: int = 2) -> None:
        self._H_solve = measurements_per_solve
        self._sparsity = sparsity

    def reset(self, state: BPLMState, context: dict) -> None:
        self._buf_k: list[int] = []
        self._buf_l: list[int] = []
        self._obp_cache: tuple[int, int] | None = None
        self._pending: tuple[int, int] | None = None  # pair requested last step
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Harvest the observation from the pair we requested last step
        if self._pending is not None:
            pk, pl = self._pending
            if state.measured_at[pk, pl] >= 0:
                self._buf_k.append(pk)
                self._buf_l.append(pl)

        # When buffer is full, solve OMP and refresh cache
        if len(self._buf_k) >= self._H_solve:
            self._solve(state)
            self._buf_k.clear()
            self._buf_l.clear()

        # Choose next pair: probe random pair for the buffer
        k = int(self._rng.integers(0, state.K))
        l = int(self._rng.integers(0, state.L))
        self._pending = (k, l)

        # Return cached OBP if available; otherwise probe
        if self._obp_cache is not None:
            return self._obp_cache
        return k, l

    # ------------------------------------------------------------------
    def _solve(self, state: BPLMState) -> None:
        L = state.L
        H_meas = len(self._buf_k)
        if H_meas < 2:
            return

        # Build sensing matrix A and measurement vector y from buffered pairs.
        # Model: y[i] = w_{k_i}^H H f_{l_i} = kron(conj(w_{k_i}), f_{l_i})^T vec(H)
        # So A[i, :] = kron(conj(w_{k_i}), f_{l_i}) and s = vec(H) ∈ C^{n_ue * n_bs}.
        # This is the standard vectorised channel CS formulation (Marzi 2016, Eq. 3).
        n_ue = state.ue_codebook.n_elements
        n_bs = state.bs_codebook.n_elements
        N = n_ue * n_bs  # dimension of vectorised channel

        A = np.zeros((H_meas, N), dtype=np.complex128)
        y_vec = np.array(
            [state.observations[k, l] for k, l in zip(self._buf_k, self._buf_l)],
            dtype=np.complex128,
        )

        for i, (k, l) in enumerate(zip(self._buf_k, self._buf_l)):
            w = state.ue_codebook.codeword(k)  # shape (n_ue,)
            f = state.bs_codebook.codeword(l)  # shape (n_bs,)
            A[i] = np.kron(w.conj(), f)  # kron: (n_ue*n_bs,)

        # Rolled OMP
        s_hat = self._omp(A, y_vec, self._sparsity)

        # The beamspace channel estimate is obtained by projecting s_hat
        # (vec(H) estimate) back through the codebook matrices.
        # Recovered beamspace gains: G[k,l] = |w_k^H unvec(s_hat) f_l|
        # = |(W^H @ unvec(s_hat) @ F)[k,l]|
        H_est = s_hat.reshape(n_ue, n_bs)
        W = state.ue_codebook.matrix  # (n_ue, K)
        F = state.bs_codebook.matrix  # (n_bs, L)
        gains = np.abs(W.conj().T @ H_est @ F)  # (K, L)

        best_flat = int(np.argmax(gains))
        self._obp_cache = (best_flat // L, best_flat % L)

    # ------------------------------------------------------------------
    @staticmethod
    def _omp(
        A: NDArray[np.complex128],
        y: NDArray[np.complex128],
        sparsity: int,
    ) -> NDArray[np.complex128]:
        """Greedy OMP: recover a ``sparsity``-sparse vector from y = A @ x + n."""
        N = A.shape[1]
        support: list[int] = []
        residual = y.copy()
        x_hat = np.zeros(N, dtype=np.complex128)

        for _ in range(min(sparsity, A.shape[0])):
            correlations = np.abs(A.conj().T @ residual)
            idx = int(np.argmax(correlations))
            if idx in support:
                break
            support.append(idx)
            A_s = A[:, support]
            coefs, *_ = np.linalg.lstsq(A_s, y, rcond=None)
            residual = y - A_s @ coefs
            x_hat[support] = coefs

        return x_hat
