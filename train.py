# ## Train agent to drive leatherback on track

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
from agent.memeory import * 

def main(): 

    env_cfg: LeatherbackEnvCfg = LeatherbackEnvCfg() 
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 15
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_buffer = EnvObsMemoryBuffer(args_cli.num_envs)
    action_buffer = EnvActMemoryBuffer(args_cli.num_envs)
    reward_buffer = EnvRewMemoryBuffer(args_cli.num_envs)

    batch_obs = {
        'image': [], 
        'steer': [],
        'vel': []
    }
    batch_actions = {
        'steer': [], 
        'vel': [] 
    } 

    batch_returns = []

    batch_size = 8
    
    policy_agent = DriveAgent(lr = 0.0001, gamma=0.99)
    policy_agent.policy = policy_agent.policy.to(args_cli.device)

    print("-" * 80)
    print("[INFO]: Resetting enviroment...")
    obs, _ = env.reset() 
    
    while simulation_app.is_running(): 
        
        
        # actions = policy_agent.get_action(obs)
        obs_img = obs["policy"]["camera_obs"].permute(0, 3, 1, 2)
        obs_s = obs["policy"]["steering_pos"]
        obs_v = obs["policy"]["wheel_vel"]

        with torch.no_grad(): 
            ## Get actions from policy 
            steer_action, vel_action = policy_agent.get_action(obs_img, obs_s, obs_v)
        ## Map action class to action values
        steer_vals, vel_vals = policy_agent.map_actions(steer_action, vel_action)

        ## Map actions values to env action values. 
        actions = torch.zeros_like(env.action_manager.action, device = env.device, dtype=torch.float32)
        actions[:, :2] = steer_vals
        actions[:, 2:] = vel_vals

        ## Take step in env 
        obs, rew, terminated, truncated, info = env.step(actions)

        ## Update the states 
        obs_buffer.add(obs_img, obs_s, obs_v)
        action_buffer.add(steer_action, vel_action)
        reward_buffer.add(rew)

        ## Check if any of the env are done. 
        done = terminated | truncated

        if torch.any(done): 
            print("In Done loop")
            for env_id in range(env.num_envs): 
                if done[env_id]: 
                    ep_obs_imgs, ep_obs_steers, ep_obs_vels = obs_buffer.get(env_id) 
                    ep_act_steer, ep_act_vel = action_buffer.get(env_id)
                    ep_returns = policy_agent.compute_return(reward_buffer.get(env_id))

                    ## Add to the training batch 
                    # Observations 
                    batch_obs['image'].extend(ep_obs_imgs)
                    print("Batch_obs size: ", np.shape(batch_obs))
                    batch_obs['steer'].extend(ep_obs_steers)
                    batch_obs['vel'].extend(ep_obs_vels)
                    # Actions
                    batch_actions['steer'].extend(ep_act_steer)
                    batch_actions['vel'].extend(ep_act_vel)
                    # Returns 
                    batch_returns.extend(ep_returns)


                    ## Reset buffer 
                    obs_buffer.reset(env_id)
                    action_buffer.reset(env_id)
                    reward_buffer.reset(env_id)

        # If the required batch of epds are recorded, run the train epoch.
        if len(batch_obs) > batch_size: 
            print("Entering the training loop")

            steer_dist, vel_dist = policy_agent.get_policy(batch_obs['image'], batch_obs['steer'], batch_obs['vel'])
            steer_log_probs = steer_dist.log_prob(batch_actions['steer'])
            vel_log_probs = vel_dist.log_prob(batch_actions['vel'])

            ## Backprop 
            policy_agent.update(steer_log_probs, vel_log_probs, batch_returns)
            print("Completed a training loop with reward: ", np.mean(batch_returns))

            ## Reset Batches 
            batch_obs['image'] = [] 
            batch_obs['steer'] = []
            batch_obs['vel'] = []
            batch_actions['steer'] = [] 
            batch_actions['vel'] = [] 
            batch_returns = []

            for env_id in range(env.num_envs): 
                obs_buffer.reset(env_id)
                action_buffer.reset(env_id)
                reward_buffer.reset(env_id)

            obs, _ = env.reset()
            print("-" * 80)
            print("[INFO]: Resetting enviroment...")

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()









        







    
        
        
        
        
        
        
        
        