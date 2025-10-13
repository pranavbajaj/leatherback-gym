## Train agent to drive leatherback on track

import argparse
import os
import numpy as np
from PIL import Image
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the leatherback-track RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from isaaclab.envs import ManagerBasedRLEnv
from leatherback_env_cfg import * 
from agent.drive_agent import * 

def main(): 

    env_cfg: LeatherbackEnvCfg = LeatherbackEnvCfg() 
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 15
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    env_rewards = [[] for _ in range(args_cli.num_envs)]
    count = 0
    
    policy_agent = DriveAgent().to(args_cli.device)
    first_action = torch.zeros_like(env.action_manager.action, dtype=torch.float32).to(args_cli.device)

    
    obs, rew, terminated, truncated, info = env.step(first_action)
    
    while simulation_app.is_running(): 
        
        if count == 2000: 
            count = 0
            env.reset()
            print("-" * 80)
            print("[INFO]: Resetting enviroment...")
        
        actions = policy_agent.get_action(obs)
        obs, rew, terminated, truncated, info = env.step(actions)
        
        
        
        
        
        
        
        