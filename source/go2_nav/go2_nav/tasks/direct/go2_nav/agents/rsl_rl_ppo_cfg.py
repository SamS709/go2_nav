# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from go2_nav.tasks.direct.go2_nav.networks.cnn_rnn_cfg import RslRlCNNRNNModelCfg
from go2_nav.tasks.direct.go2_nav.networks.cnn3d_cfg import RslRlCNN3DModelCfg

CNN_RNN_MODEL = "go2_nav.tasks.direct.go2_nav.networks.cnn_rnn_model:CNNRNNModel"
CNN_RNN_SEQ_MODEL = "go2_nav.tasks.direct.go2_nav.networks.cnn_rnn_model:CNNRNNSeqModel"
CNN3D_MODEL = "go2_nav.tasks.direct.go2_nav.networks.cnn3d_model:CNN3DModel"


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_nav_planner"
    # Asymmetric actor-critic: student observations for actor, privileged teacher observations for critic.
    obs_groups = {
        "actor": ["student_proprio", "student_height_scan"],
        "critic": ["teacher_proprio", "teacher_height_scan"],
    }
    actor = RslRlCNNRNNModelCfg(
        class_name=CNN_RNN_MODEL,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNRNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        distribution_cfg=RslRlCNNRNNModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlCNNRNNModelCfg(
        class_name=CNN_RNN_MODEL,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNRNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    
@configclass
class PPORunnerSeqCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_nav_planner"
    # Asymmetric actor-critic: student observations for actor, privileged teacher observations for critic.
    obs_groups = {
        "actor": ["student_proprio", "student_height_scan"],
        "critic": ["teacher_proprio", "teacher_height_scan"],
    }
    actor = RslRlCNNRNNModelCfg(
        class_name=CNN_RNN_SEQ_MODEL,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNRNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        distribution_cfg=RslRlCNNRNNModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlCNNRNNModelCfg(
        class_name=CNN_RNN_SEQ_MODEL,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNRNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )