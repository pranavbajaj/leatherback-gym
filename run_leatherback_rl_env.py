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
import os
import numpy as np
from PIL import Image
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
    # print(curr_action.size())
    
    # curr_action[:, :2] = torch.randint(low=-2, high=3, size=(env_cfg.scene.num_envs, 2), dtype=torch.float32)    # Position, consider scaling factor of 0.1
    curr_action[:, 2:] = torch.randint(low=0, high=3, size=(env_cfg.scene.num_envs, 4), dtype=torch.float32)

    # stage = env.sim.stage
    # robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")

    # print("=== Robot Children ===")
    # for prim in stage.Traverse():
    #     if "Camera" in prim.GetName() or "camera" in prim.GetName():
    #         print(f"Found camera: {prim.GetPath()}")


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

            camera = env.scene["chase_camera"]
            rgb_data = camera.data.output["rgb"]  # (num_envs, H, W, 3)
            print(f"image data sizes: {rgb_data.size()}")

            save_dir = "camera_output"
            os.makedirs(save_dir, exist_ok=True)

            rgb_numpy = rgb_data.cpu().numpy()
            for i in range(min(4, rgb_numpy.shape[0])):  # Save first 4 envs
                img = rgb_numpy[i]
                
                # Convert to uint8 if normalized
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                # Save image
                Image.fromarray(img).save(f"{save_dir}/env_{i}_{count}_camera.png")
                
            print(f"Saved camera images to {save_dir}/")

            count += 1

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()