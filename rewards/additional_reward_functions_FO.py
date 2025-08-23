import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Custom reward functions for smooth driving
def steering_smoothness_reward(env) -> torch.Tensor:
    """Penalize rapid steering changes."""
    robot = env.scene["robot"]
    
    # Get steering joint velocities (rate of change)
    steering_joints = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]
    steering_vel = robot.data.joint_vel[:, robot.joint_names.index(steering_joints[0]):robot.joint_names.index(steering_joints[1])+1]
    
    # Penalize high angular velocity on steering
    steering_penalty = -torch.sum(torch.abs(steering_vel), dim=1)
    return steering_penalty


def wheel_acceleration_smoothness_reward(env) -> torch.Tensor:
    """Penalize rapid wheel velocity changes."""
    robot = env.scene["robot"]
    
    # Get current wheel velocities
    wheel_joints = ["Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right", 
                   "Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"]
    
    # Get indices for wheel joints
    wheel_indices = [robot.joint_names.index(joint) for joint in wheel_joints]
    current_wheel_vel = robot.data.joint_vel[:, wheel_indices]
    
    # Store previous velocities (initialize if not exists)
    if not hasattr(env, '_prev_wheel_vel'):
        env._prev_wheel_vel = current_wheel_vel.clone()
    
    # Calculate acceleration (change in velocity)
    wheel_acceleration = (current_wheel_vel - env._prev_wheel_vel) / env.sim.dt
    
    # Update stored velocities
    env._prev_wheel_vel = current_wheel_vel.clone()
    
    # Penalize high accelerations (both positive and negative)
    accel_penalty = -torch.sum(torch.abs(wheel_acceleration), dim=1) * 0.01  # Scale factor
    return accel_penalty


def forward_velocity_reward(env) -> torch.Tensor:
    """Reward forward velocity more than just wheel speed."""
    robot = env.scene["robot"]
    
    # Get robot's forward velocity in world frame
    # Assuming X is forward direction
    forward_vel = robot.data.root_lin_vel_w[:, 0]  # X component of velocity
    
    # Only reward positive forward velocity
    forward_reward = torch.clamp(forward_vel, min=0.0)
    return forward_reward


def wheel_slip_penalty(env) -> torch.Tensor:
    """Penalize wheel slip (difference between wheel speed and actual movement)."""
    robot = env.scene["robot"]
    
    # Get wheel velocities
    wheel_joints = ["Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right", 
                   "Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"]
    wheel_indices = [robot.joint_names.index(joint) for joint in wheel_joints]
    wheel_vel = robot.data.joint_vel[:, wheel_indices]
    
    # Average wheel speed (assuming wheel radius ~0.1m)
    wheel_radius = 0.1
    expected_forward_speed = torch.mean(wheel_vel, dim=1) * wheel_radius
    
    # Actual forward speed
    actual_forward_speed = robot.data.root_lin_vel_w[:, 0]
    
    # Penalize difference (slip)
    slip_penalty = -torch.abs(expected_forward_speed - actual_forward_speed)
    return slip_penalty

