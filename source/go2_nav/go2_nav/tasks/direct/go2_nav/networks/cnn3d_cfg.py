# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlCNN3DModelCfg(RslRlMLPModelCfg):
    """Configuration for the 3D-CNN-based model."""

    @configclass
    class CNNCfg:
        """Configuration for the 3D-CNN network."""

        output_channels: list[int] = MISSING
        """List of output channels for each convolutional layer."""
        kernel_size: list[int] | int = MISSING
        """List of kernel sizes for each layer."""
        stride: list[int] | int = 1
        """List of strides for each layer."""
        padding: str = "none"
        """Padding type."""
        norm: str | list[str] = "none"
        """Normalization type."""
        activation: str = "elu"
        """Activation function."""
        max_pool: bool | list[bool] = False
        """Whether to apply max pooling."""
        global_pool: str = "none"
        """Global pooling type."""
        flatten: bool = True
        """Whether to flatten the output."""

    cnn_cfg: CNNCfg | dict[str, CNNCfg] = CNNCfg()
    """The configuration for the 3D-CNN network."""
