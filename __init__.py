import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks
from .leatherback_env_cfg import LeatherbackEnvCfg


# Register the main environment
gym.register(
    id="Isaac-LeatherbackRaceTrack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "cfg": LeatherbackEnvCfg,
    },
    disable_env_checker=True,  # Add this to avoid gym warnings
)

# Testing variant with fewer environments
gym.register(
    id="Isaac-LeatherbackRaceTrack-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "cfg": LeatherbackEnvCfg,
        "cfg_overrides": {
            "scene.num_envs": 2,
            "scene.env_spacing": 15.0,  # Increase spacing for testing
        }
    },
    disable_env_checker=True,
)

# No-camera variant for faster training
class LeatherbackEnvNoCameraCfg(LeatherbackEnvCfg):
    """Leatherback environment without camera observations."""
    
    def __post_init__(self):
        super().__post_init__()
        # Remove camera observation
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import SceneEntityCfg
        
        # Create new observation config without camera
        @configclass
        class PolicyCfgNoCamera(ObsGroup):
            # Steering angle
            steering_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=["Knuckle__Upright__Front.*"])},
                scale=1.0/0.6
            )

            # Wheel velocities
            wheel_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=["Wheel.*"])},
                scale=1.0/30.0
            )

            # Local frame velocities (what the robot "feels")
            root_lin_vel = ObsTerm(
                func=mdp.base_lin_vel,  # Local frame
                params={"asset_cfg": SceneEntityCfg("robot")},
                scale=0.1
            )
            # Robot learns: [forward, lateral, vertical] speeds
            
            root_ang_vel = ObsTerm(
                func=mdp.base_ang_vel,  # Local frame
                params={"asset_cfg": SceneEntityCfg("robot")},
                scale=0.1
            )
            # Robot learns: [roll, pitch, yaw] rates
            
            # Previous actions (action history)
            last_action = ObsTerm(
                func=mdp.last_action,
            )
            
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        
        # Replace the policy observation config
        self.observations.policy = PolicyCfgNoCamera()

gym.register(
    id="Isaac-LeatherbackRaceTrack-NoCamera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": LeatherbackEnvNoCameraCfg},
    disable_env_checker=True,
)

# Custom wrapper (fixed)
from gymnasium import Wrapper
import torch

class LeatherbackEnvWrapper(Wrapper):
    """Custom wrapper for additional functionality."""
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add custom metrics (handle tensor rewards properly)
        if isinstance(reward, torch.Tensor):
            info["custom_metric"] = float(reward.mean().item())
        else:
            info["custom_metric"] = float(reward)
        
        return obs, reward, terminated, truncated, info

# Register wrapped version (fixed lambda)
gym.register(
    id="Isaac-LeatherbackRaceTrack-Wrapped-v0",
    entry_point="__main__:create_wrapped_env",
    disable_env_checker=True,
)

def create_wrapped_env():
    """Factory function for creating wrapped environment."""
    return LeatherbackEnvWrapper(gym.make("Isaac-LeatherbackRaceTrack-v0"))

__all__ = [
    "LeatherbackEnvCfg",
    "LeatherbackEnvNoCameraCfg",
    "LeatherbackEnvWrapper",
    "create_wrapped_env",
]

# Print registration confirmation
print("Leatherback environments registered:")
print("  - Isaac-LeatherbackRaceTrack-v0 (Main environment)")
print("  - Isaac-LeatherbackRaceTrack-Test-v0 (Testing with 4 envs)")
print("  - Isaac-LeatherbackRaceTrack-NoCamera-v0 (No camera observations)")
print("  - Isaac-LeatherbackRaceTrack-Wrapped-v0 (With custom wrapper)")