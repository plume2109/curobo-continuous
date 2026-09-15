# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the relative-pose (loop-closure) cost between two tool frames."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Type

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_relative_pose import RelativePoseCost


@dataclass
class RelativePoseCostCfg(BaseCostCfg):
    """Configuration for :class:`RelativePoseCost`.

    Penalizes ``relpose(frame_a, frame_b) = T_a^{-1} T_b`` drifting from a fixed target
    (a rigid two-hand co-grasp). The target is optional at construction (defaults to identity)
    and is normally set live via :meth:`RelativePoseCost.update_target`.
    """

    #: Class type of the cost, used to build the cost from this config.
    class_type: Type[RelativePoseCost] = RelativePoseCost

    #: Tool frame the relative pose is measured FROM (the reference of the pair).
    frame_a: Optional[str] = None
    #: Tool frame the relative pose is measured TO.
    frame_b: Optional[str] = None

    #: Target translation of relpose(a, b) = R_a^T (t_b - t_a). None -> zeros.
    target_rel_position: Optional[List[float]] = None
    #: Target rotation (wxyz) of relpose(a, b) = R_a^T R_b. None -> identity [1,0,0,0].
    target_rel_quaternion: Optional[List[float]] = None

    #: Relative weighting of the translation term inside this cost.
    position_weight: float = 1.0
    #: Relative weighting of the rotation term inside this cost.
    rotation_weight: float = 1.0

    def clone(self) -> "RelativePoseCostCfg":
        return RelativePoseCostCfg(
            weight=self.weight.clone(),
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            frame_a=self.frame_a,
            frame_b=self.frame_b,
            target_rel_position=(
                list(self.target_rel_position) if self.target_rel_position is not None else None
            ),
            target_rel_quaternion=(
                list(self.target_rel_quaternion) if self.target_rel_quaternion is not None else None
            ),
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
        )
