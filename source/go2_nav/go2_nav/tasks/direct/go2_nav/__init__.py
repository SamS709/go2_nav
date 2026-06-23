# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Go2-Nav-Direct-v0",
    entry_point=f"{__name__}.go2_nav_env:Go2NavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_nav_env_cfg:Go2NavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2NavTeacherPretrainRunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2NavDistillationRunnerCfg",
    },
)