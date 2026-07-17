---
description: Codebase for the workspace
---

Here are the useful codebases you can use to awnser the questions.


1. [NAVIGATION WORKSPACE] -> CURRENT WORKSPACE !!!

In this workspace, a navigation policy is trained using Isaaclab framework.
The key files are located in /mnt/D/dev/robotics/nvidia/isaaclab/go2_nav/source/go2_nav/go2_nav/tasks/direct/go2_nav
Especially:
- go2_nav_env_cfg.py: config for the training
- go2_nav_env.py: script for the training (observations, rewards, reset, ...)
- agents/rsl_rl_ppo_distillation_cfg.py: Go2NavTeacherPretrainRunnerCfg -> config for the agents (teacher and students).

The navigation_policy is trained to go from a point A to a point B. Its goal is to find the velocity to send as input of a pretrained locomotion_policy.

2. [LOCOMOTION WORKSPACE]

The locomotion policy was trained using Isaaclab framework to walk on rough terrains. It is has been deployed on real hardware and is working very well. No probem can be related to its training. All the problems related to the locomotion_policy is related to the way it is used, or the environment in which it is deployed.
The key files are located in /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar/source/go2_lidar/go2_lidar/tasks/direct/go2_lidar
Especially:
- go2_lidar_env_cfg.py: config for the training
- go2_lidar_env.py & go2_cnn_lidar_env.py: script for the training (observations, rewards, reset, ...)
- agents/rsl_rl_ppo_cfg.py: Go2LidarRoughCNNRNNSeqPPORunnerCfg -> config for the agent (Simple actor critic).

The locomotion policy is trained to follow input velocities and walk on rough terrain thanks to a RayCaster (lidar).

3. [ISAACLAB DOCUMENTATION]
Accessible via the optimized_rag tools. Useful to get informations about the IsaacLab source code and API.

3. [TRIMESH DOCUMENTATION]
Accessible via the optimized_rag tools. Useful to get informations about the trimesh source code and API.