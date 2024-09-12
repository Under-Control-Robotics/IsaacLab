# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


def compute_tanh_shift(k1, k2, y0):
    numerator = k1 * k2 + y0 * k1
    denominator = k1 * k2 - y0 * k2
    shift = 0.5 * torch.log(numerator / denominator)
    return shift

def asymetric_tanh(x: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, x_shift: torch.Tensor) -> torch.Tensor:
    e2x = torch.exp((x + x_shift) * 2.)
    tanh_pos = (e2x - 1.) / (e2x / k1 + 1. / k2)
    e2nx = torch.exp((-x - x_shift) * 2.)
    tanh_neg = (1. - e2nx) / (1. / k1 + e2nx / k2)
    return torch.where(x<0, tanh_pos, tanh_neg)

class JointPositionToLimitsAction(ActionTerm):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip the scaled actions to the range [-1, 1] and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position commands to the articulation's joints.
    """

    cfg: actions_cfg.JointPositionToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        # parse offset
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self.k1 = self._asset.data.joint_limits[:, self._joint_ids, 1]
        self.k2 = -self._asset.data.joint_limits[:, self._joint_ids, 0]
        self.x_shift = compute_tanh_shift(self.k1, self.k2, self._offset)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply affine transformations
        self._processed_actions = self._raw_actions * self._scale
        # rescale the position targets if configured
        # this is useful when the input actions are in the range [-1, 1]
        if self.cfg.rescale_to_limits:
            actions = asymetric_tanh(self._processed_actions, self.k1, self.k2, self.x_shift)
            self._processed_actions[:] = actions[:]

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    r"""Joint action term that applies exponential moving average (EMA) over the processed actions as the
    articulation's joints position commands.

    Exponential moving average (EMA) is a type of moving average that gives more weight to the most recent data points.
    This action term applies the processed actions as moving average position action commands.
    The moving average is computed as:

    .. math::

        \text{applied action} = \alpha \times \text{processed actions} + (1 - \alpha) \times \text{previous applied action}

    where :math:`\alpha` is the weight for the moving average, :math:`\text{processed actions}` are the
    processed actions, and :math:`\text{previous action}` is the previous action that was applied to the articulation's
    joints.

    In the trivial case where the weight is 1.0, the action term behaves exactly like
    the :class:`JointPositionToLimitsAction` class.

    On reset, the previous action is initialized to the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.EMAJointPositionToLimitsActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.EMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        # reset history to current joint positions
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations
        super().process_actions(actions)
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        self._processed_actions[:] = torch.clamp(
            ema_actions,
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]
