# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg


@configclass
class RslRlCNN3DModelCfg(RslRlCNNModelCfg):
    """Configuration for CNN + RNN model."""

    class_name: str = "CNN3DModel"
    """The model class name. Default is CNN3DModel."""

    

