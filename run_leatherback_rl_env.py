# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from isaaclab.envs import ManagerBasedRLEnv

from leatherback_env_cfg import * 

def main(): 

    env_cfg: LeatherbackEnvCfg = LeatherbackEnvCfg() 
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 15
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0

    print(f"Env_0 Origin: {env.scene.env_origins[0, :3]}")

    curr_action = torch.zeros_like(env.action_manager.action, dtype=torch.float32)
    print(curr_action.size())
    
    num_envs = curr_action.shape[0]
    
    curr_action[:, :2] = torch.randint(low=-2, high=3, size=(env_cfg.scene.num_envs, 2), dtype=torch.float32)    # Position, consider scaling factor of 0.1
    curr_action[:, 2:] = torch.randint(low=-4, high=5, size=(env_cfg.scene.num_envs, 4), dtype=torch.float32)


    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions

            obs, rew, terminated, truncated, info = env.step(curr_action)
            count += 1

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()