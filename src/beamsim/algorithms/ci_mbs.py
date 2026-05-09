"""Multi-BS Context-Information tracking.

Mirrors the predecessor MATLAB simulator's ``updateCIMBS.m``: instead
of always pointing at a single fixed BS, the algorithm computes the
distance from the UE to every BS in the trial context, picks the
*closest* one, and then runs the standard sin-space CI match for that
BS.

The Python re-implementation differs from MATLAB's
``abs(atan2(...))`` half-plane fold by matching in :math:`\\sin\\theta`
space, which correctly resolves the ULA front/back ambiguity --- the
same correction applied in :class:`beamsim.algorithms.ContextInformation`.
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class ContextInformationMBS(Algorithm):
    """Closest-BS-aware CI: pick the closest BS, then sin-space CI match.

    Requires ``context["bs_list"]`` (a list of ``{"bs_xy", "bs_yaw"}``
    dictionaries) which the multi-BS runner provides automatically.
    Falls back to the single-BS context when ``bs_list`` is absent.
    """

    name = "ci_mbs"

    def reset(self, state: BPLMState, context: dict) -> None:
        pass

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        ue_pose = context["ue_pose_at"](m)
        ue_xy, ue_yaw = ue_pose[0], ue_pose[1]

        bs_list = context.get("bs_list")
        if bs_list:
            distances = [float(np.linalg.norm(np.asarray(bs["bs_xy"]) - ue_xy)) for bs in bs_list]
            best = int(np.argmin(distances))
            bs_xy = np.asarray(bs_list[best]["bs_xy"])
            bs_yaw = float(bs_list[best].get("bs_yaw", 0.0))
        else:
            # Single-BS fallback so CIMBS can also be used in unit tests
            # without a multi-BS scenario.
            bs_xy = np.asarray(context["bs_xy"])
            bs_yaw = float(context.get("bs_yaw", 0.0))

        aoa_world = np.arctan2(bs_xy[1] - ue_xy[1], bs_xy[0] - ue_xy[0])
        aod_world = np.arctan2(ue_xy[1] - bs_xy[1], ue_xy[0] - bs_xy[0])
        aoa_rel = float((aoa_world - ue_yaw + np.pi) % (2 * np.pi) - np.pi)
        aod_rel = float((aod_world - bs_yaw + np.pi) % (2 * np.pi) - np.pi)

        ue_sin = np.sin(state.ue_codebook.theta)
        bs_sin = np.sin(state.bs_codebook.theta)
        k = int(np.argmin(np.abs(ue_sin - np.sin(aoa_rel))))
        l = int(np.argmin(np.abs(bs_sin - np.sin(aod_rel))))
        return k, l
