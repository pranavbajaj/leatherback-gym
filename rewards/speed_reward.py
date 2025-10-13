import torch
import numpy as np
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

def speed_magnitude_reward_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward total horizontal speed magnitude in LOCAL frame.
    Good for encouraging movement while penalizing sideways sliding.
    """
    robot = env.scene["robot"]
    local_vel = robot.data.root_lin_vel_b
    
    # Forward and lateral velocities
    forward_speed = local_vel[:, 0]
    lateral_speed = local_vel[:, 1]
    
    # Reward forward speed, penalize lateral (sliding)
    # speed_reward = forward_speed - 0.5 * torch.abs(lateral_speed)

    # Not penalizing lateral_speed because I would like robot to learn to drift. 
    speed_reward = forward_speed
    
    return speed_reward