# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg


@configclass
class RslRlCNNRNNModelCfg(RslRlCNNModelCfg):
    """Configuration for CNN + RNN model."""

    class_name: str = "CNNRNNModel"
    """The model class name. Default is CNNRNNModel. Either CNNRNNModel or CNNRNNSeqModel."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""
    

