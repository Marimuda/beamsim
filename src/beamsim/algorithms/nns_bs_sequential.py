"""NNS-BS-Sequential: NNS on UE dimension, round-robin on BS dimension.

Implements the variant described in report Sec. 6.5 / Fig. 6.7:
  - UE beam (k): standard NNS steepest-ascent hill-climb, exactly as in NNS.
  - BS beam (l): fixed sequential round-robin, advancing by ``bs_stride``
    each call: l = (l + bs_stride) % L.  The NNS logic still runs internally
    but its BS-dimension choice is overridden before returning.

This isolates the effect of losing BS-beam coordination (Algorithm 2, n=7)
on overall system performance.
"""

from __future__ import annotations

from beamsim.algorithms.base import Algorithm
from beamsim.algorithms.nns import NNS
from beamsim.bplm import BPLMState


class NNSBSSequential(Algorithm):
    """NNS on UE beams; sequential round-robin on BS beams.

    Parameters
    ----------
    bs_stride : int
        Step size for the BS round-robin scan.  Default 7 matches
        Algorithm 2 (predecessor thesis, Sec. 5.4.1).
    """

    name = "nns_bs_sequential"

    def __init__(self, bs_stride: int = 7) -> None:
        self._bs_stride = bs_stride
        self._nns = NNS(connectivity=4)

    def reset(self, state: BPLMState, context: dict) -> None:
        self._nns.reset(state, context)
        self._l: int = 0  # BS beam counter; starts at 0, first return is bs_stride

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Advance BS beam by stride (round-robin)
        self._l = (self._l + self._bs_stride) % state.L

        # Get UE beam from NNS (we ignore its BS choice)
        k, _l_nns = self._nns.select_next_mbp(state, m, context)

        return k, self._l
