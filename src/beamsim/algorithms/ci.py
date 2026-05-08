"""Context-Information tracking: noiseless geometry-driven beam selection."""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class ContextInformation(Algorithm):
    """Picks the codebook beam closest to the geometric LOS direction.

    Requires the trial context to expose `ue_pose_at(m)` and `bs_xy`,
    `bs_yaw`. Insensitive to measurement noise but vulnerable to
    LOS-blockage and NLOS-only operation.
    """

    name = "ci"

    def reset(self, state: BPLMState, context: dict) -> None:
        pass

    def select_next_mbp(self, state, m, context):
        ue_pose = context["ue_pose_at"](m)
        ue_xy, ue_yaw = ue_pose[0], ue_pose[1]
        bs_xy = context["bs_xy"]
        bs_yaw = context.get("bs_yaw", 0.0)
        aoa_world = np.arctan2(bs_xy[1] - ue_xy[1], bs_xy[0] - ue_xy[0])
        aod_world = np.arctan2(ue_xy[1] - bs_xy[1], ue_xy[0] - bs_xy[0])
        aoa_rel = _wrap_pi(aoa_world - ue_yaw)
        aod_rel = _wrap_pi(aod_world - bs_yaw)
        # Match in sin-space: ULA array response only depends on sin(theta),
        # so theta and pi-theta give the same array response (front/back
        # half-plane ambiguity). Picking by min |sin(theta_k) - sin(target)|
        # selects the codebook beam that genuinely matches the array
        # response, even if the target lies in the back half-plane.
        ue_sin = np.sin(state.ue_codebook.theta)
        bs_sin = np.sin(state.bs_codebook.theta)
        k = int(np.argmin(np.abs(ue_sin - np.sin(aoa_rel))))
        l = int(np.argmin(np.abs(bs_sin - np.sin(aod_rel))))
        return k, l


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
