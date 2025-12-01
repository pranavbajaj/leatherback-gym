"""
Isaac Lab Robotics Interview Problem
=====================================

Problem: Multi-Robot Reward Computation System

You are building a reward computation system for a multi-robot RL training pipeline using Isaac Lab.
Your system needs to efficiently compute rewards for multiple environments running in parallel,
handling state normalization, distance calculations, and reward aggregation.

Context:
--------
- You have N parallel environments, each running a robot manipulation task
- Each environment has a robot with state: [joint_positions, joint_velocities, end_effector_pose]
- Goal: Compute rewards based on goal reaching, smoothness, and energy efficiency
- Must use torch.tensor for all numerical computations for GPU acceleration

Your Tasks:
-----------
1. Implement RewardComputer class with efficient tensor-based reward computation
2. Handle multiple environment setup with proper batch dimensioning
3. Implement state normalization using running statistics
4. Compute task-specific rewards (goal distance, smoothness, energy)
5. Implement environment reset logic with proper state initialization

Constraints:
- Use only PyTorch (no NumPy for computations)
- All operations must be vectorized (no loops over environments)
- Handle edge cases (division by zero, gradient flow)
- Ensure numerical stability
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for each environment"""
    num_envs: int  # Number of parallel environments
    num_joints: int  # Number of robot joints
    action_dim: int  # Dimension of action space
    state_dim: int  # Dimension of state space
    device: str = "cpu"  # torch device
    
    
class RobotState:
    """Container for robot state across multiple environments"""
    
    def __init__(self, batch_size: int, num_joints: int, device: str = "cpu"):
        self.batch_size = batch_size
        self.num_joints = num_joints
        self.device = device
        
        # TODO: Initialize joint position tensor [batch_size, num_joints]
        self.joint_pos = torch.zeros((batch_size, num_joints), dtype = torch.float32, device=device)
        
        # TODO: Initialize joint velocity tensor [batch_size, num_joints]
        self.joint_vel = torch.zeros((batch_size, num_joints), dtype = torch.float32, device=device)
        
        # TODO: Initialize end-effector position tensor [batch_size, 3]
        self.ee_pos = torch.zeros((batch_size, 3), dtype = torch.float32, device=device)
        
        # TODO: Initialize end-effector quaternion tensor [batch_size, 4]
        self.ee_quat = torch.zeros((batch_size, 4), dtype = torch.float32, device=device)
        self.ee_quat[:, 0] = 1.0
        
        # TODO: Initialize goal position tensor [batch_size, 3]
        self.goal_pos = torch.zeros((batch_size, 3), dtype = torch.float32, device=device)
        
    def get_state_vector(self) -> torch.Tensor:
        """
        TODO: Concatenate all states into single vector
        
        Should return tensor of shape [batch_size, state_dim] where state_dim includes:
        - joint_pos (num_joints)
        - joint_vel (num_joints)
        - ee_pos (3)
        - ee_quat (4)
        - goal_pos (3)
        """
        # HINT: Use torch.cat() along dimension -1
        return torch.cat([self.joint_pos, self.joint_vel, self.ee_pos, self.ee_quat, self.ee_quat, self.goal_pos], dim = -1)


class RunningNormalizer:
    """Tracks running mean and std for state normalization"""
    
    def __init__(self, dim: int, device: str = "cpu", epsilon: float = 1e-8):
        """
        TODO: Initialize running statistics
        
        Args:
            dim: Dimension of data to normalize
            device: torch device
            epsilon: Small constant for numerical stability
            
        Initialize:
        - self.mean as zeros tensor [dim]
        - self.var as ones tensor [dim]
        - self.count as zero scalar tensor
        - Store device and epsilon as instance variables
        """
        self.device = device
        self.epsilon = torch.tensor(epsilon, device = device)

        self.mean = torch.zeros((dim), dtype = torch.float32, device = device)
        self.var = torch.onse((dim), dtype = torch.float32, device = device)
        self.count = torch.tensor(0, dtype = torch.int, device = device)
        
    def update(self, x: torch.Tensor) -> None:
        """
        TODO: Update running statistics with new batch using Welford's algorithm
        
        Args:
            x: Batch of data with shape [batch_size, dim]
            
        Steps:
        1. Compute batch mean along dimension 0
        2. Compute batch variance along dimension 0 (use unbiased=False)
        3. Get batch count from first dimension
        4. Update total count
        5. Update running mean using: mean = mean + delta * batch_count / total_count
        6. Update running variance: var = (var * old_count + batch_var * batch_count) / total_count
        """
        
        batch_mean = torch.mean(x, dim = 0)
        batch_var = torch.var(x, dim = 0)
        batch_count = torch.tensor(x.size()[0], device = self.device)

        new_count = self.count + batch_count

        self.mean = self.mean + (batch_count / new_count) * batch_mean
        self.var = (self.var * self.count + batch_var * batch_count) / new_count
        self.count = new_count



        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Normalize data using current statistics
        
        Args:
            x: Input tensor [batch_size, dim]
            
        Returns:
            Normalized tensor: (x - mean) / sqrt(var + epsilon)
        """
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)



class RewardComputer:
    """Computes rewards for multi-environment robot tasks"""
    
    def __init__(self, config: EnvConfig):
        """
        TODO: Initialize reward computer
        
        Args:
            config: EnvConfig with environment parameters
            
        Store config and device, initialize:
        - Reward weights: goal_distance_weight, smoothness_weight, energy_weight
        - RunningNormalizer for state (dim=config.state_dim)
        - Previous actions tensor [num_envs, action_dim] initialized to zeros
        """
        self.config = config 
        self.device = config.device
        self.runningNormalizer = RunningNormalizer(config.state_dim, device = config.device)

        # TO______DO check below again 
        self.goal_distacne_weigth = 0.1
        self.smoothness_weight = 0.1
        self.energy_weight = 0.1
        self.prev_action = torch.zeros([config.num_envs, config.action_dim], dtype = torch.float32, device = config.device)
        
    def compute_rewards(
        self,
        robot_state: RobotState,
        actions: torch.Tensor,
        success_threshold: float = 0.05
    ) -> Dict[str, torch.Tensor]:
        """
        TODO: Compute rewards for batch of environments
        
        Args:
            robot_state: Current robot state for all environments
            actions: Actions executed [num_envs, action_dim]
            success_threshold: Distance threshold for goal reaching (meters)
            
        Returns:
            Dict containing:
                - 'total_reward': [num_envs] total reward
                - 'goal_distance_reward': [num_envs] goal distance reward component
                - 'smoothness_reward': [num_envs] action smoothness reward component
                - 'energy_reward': [num_envs] energy efficiency reward component
                - 'success': [num_envs] binary success flag
                - 'goal_distance_metric': [num_envs] actual distance to goal
        
        Steps:
        1. Assert action shape is correct

        
        2. GOAL DISTANCE REWARD:
           - Compute L2 distance from ee_pos to goal_pos using torch.norm(dim=-1)
           - Apply gaussian reward: torch.exp(-5.0 * distance)
           
        3. SMOOTHNESS REWARD:
           - Compute action difference from previous actions
           - Use torch.norm along action dimension
           - Apply negative penalty: -smoothness_norm
           
        4. ENERGY EFFICIENCY REWARD:
           - Compute norm of joint velocities
           - Apply negative penalty: -joint_vel_norm
           
        5. SUCCESS BONUS:
           - Check if distance < success_threshold
           - Add bonus reward of 10.0 when successful
           
        6. COMBINE REWARDS:
           - total_reward = weights * individual_rewards
           
        7. Update self.prev_actions for next step
        """
        
        # Assert action shape in correct 
        assert actions.shape == (self.config.num_envs, self.config.action_dim), f"Action Dim got {actions.shape}, but expected {(self.config.num_envs, self.config.action_dim)}"

        # Goal distance reward 
        ## Check_________ the gaussian reward 
        distance = torch.norm((robot_state.ee_pos, robot_state.goal_pos), dim = -1)
        goal_distance_reward = torch.exp(5 * distance)

        # Smoothness reward 
        smoothness_reward = -1.0 * torch.norm((actions - self.prev_action), dim = -1)

        # Energy efficiency reward 
        energy_efficiency_reward = -1.0 * torch.norm((robot_state.joint_vel), dim = -1)

        # Success Bonus 
        success_bonus = torch.zeros((self.config.num_envs), dtype=torch.float32, device=self.device)
        mask = distance < success_threshold
        success_bonus[mask] = 10.0

        
        # Combined Rewards 
        combined_rewards = self.goal_distacne_weigth * goal_distance_reward + self.smoothness_weight * smoothness_reward + self.energy_weight * energy_efficiency_reward + success_bonus

        # Update prev_actions 
        self.prev_action = actions

        # Returns:
        #     Dict containing:
        #         - 'total_reward': [num_envs] total reward
        #         - 'goal_distance_reward': [num_envs] goal distance reward component
        #         - 'smoothness_reward': [num_envs] action smoothness reward component
        #         - 'energy_reward': [num_envs] energy efficiency reward component
        #         - 'success': [num_envs] binary success flag
        #         - 'goal_distance_metric': [num_envs] actual distance to goal

        out_dict = {
            "total_reward": combined_rewards, 
            "goal_distance_reward": goal_distance_reward, 
            "smoothness_reward": smoothness_reward, 
            "energy_reward": energy_efficiency_reward, 
            "success": mask, 
            "goal_distance_metric": distance
        }

        return out_dict
        

    
    def normalize_state(self, robot_state: RobotState, update: bool = True) -> torch.Tensor:
        """
        TODO: Normalize state using running statistics
        
        Args:
            robot_state: Current robot state
            update: Whether to update running statistics
            
        Returns:
            Normalized state vector [num_envs, state_dim]
            
        Steps:
        1. Get state vector from robot_state
        2. Update normalizer if update=True
        3. Normalize using normalizer.normalize()
        """
        state_vector = robot_state.get_state_vector()

        if update: 
            self.runningNormalizer.update(state_vector)

        state_vector = self.runningNormalizer.normalize(state_vector)

        


class MultiEnvironmentManager:
    """Manages multiple parallel robot environments"""
    
    def __init__(self, config: EnvConfig):
        """
        TODO: Initialize environment manager
        
        Args:
            config: EnvConfig with environment parameters
            
        Initialize:
        - Store config and device
        - Create RobotState for batch_size=num_envs, num_joints=config.num_joints
        - Create RewardComputer with config
        - Episode tracking: episode_steps [num_envs], episode_rewards [num_envs]
        - Set max_episode_steps = 500
        """
        self.config = config
        self.device = config.device
        self.robot_state = RobotState(batch_size=config.num_envs, num_joints=config.num_joints)
        self.rewardComputer = RewardComputer(config=config)
        self.episode_steps = torch.zeros((config.num_envs), dtype=torch.int, device = self.device)
        self.episode_rewards = torch.zeros((config.num_envs), dtype=torch.float32, device = self.device)
        self.max_episode_steps = 500
        
    def reset_environments(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        TODO: Reset specified environments (or all if env_ids is None)
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all.
            
        Returns:
            Initial state observations dict with 'state' key
            
        Steps:
        1. If env_ids is None, create tensor [0, 1, ..., num_envs-1]
        2. Get num_reset from env_ids length
        3. Randomly initialize joint positions [num_reset, num_joints] 
           - Use torch.randn() * 0.5
        4. Zero out joint velocities at env_ids
        5. Random goal positions [num_reset, 3]
           - torch.randn() * 0.3 + [0.5, 0.0, 0.5]
        6. Initialize end-effector pose based on joint positions
           - ee_pos: use mean of joint_pos along joint dimension, expand to 3D
           - ee_quat: [1, 0, 0, 0] quaternion
        7. Reset episode tracking (steps, rewards to 0)
        8. Return observations via _get_observations()
        """
        # Check env ids, if None, reset all envs 
        if env_ids == None: 
            env_ids = torch.tensor([i for i in range(self.config.num_envs)]) 

        # Get num of env to reset 
        num_reset = env_ids.size()[0]

        # Reset robot joint pose to random value between 0-0.5 and joint vel to 0.0
        self.robot_state.joint_pos[env_ids] = torch.randn((num_reset, self.config.num_joints), dtype=torch.float32, device=self.device) * 0.5
        self.robot_state.joint_vel[env_ids] = torch.zeros((num_reset, self.config.num_joints), dtype=torch.float32, device=self.device)

        # Reset Goal positions 
        self.robot_state.goal_pos[env_ids] = torch.randn((num_reset, 3), dtype=torch.float32, device=self.device) * 0.3 + torch.tensor([0.5, 0.0, 0.5], dtype = torch.float32, device=self.device)

        # Initialized end-effector pos 
        # Check________Below
        self.robot_state.ee_pos[env_ids] = torch.mean(self.robot_state.joint_pos[env_ids], dim = -1).expand(num_reset, 3)
        # Init ee-quat 
        self.robot_state.ee_quat[env_ids] = torch.zeros((num_reset, 4), dtype=torch.float32, device = self.device)
        self.robot_state.ee_quat[env_ids][0] = 1.0

        # Reset episode tracking (steps, rewards to 0)
        self.episode_rewards[env_ids] = 0.0
        self.episode_steps[env_ids] = 0

        return self._get_observations() 



    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        TODO: Step all environments forward
        
        Args:
            actions: Actions for each environment [num_envs, action_dim]
            
        Returns:
            Tuple of (observations, rewards, dones)
            
        Steps:
        1. Assert actions shape is correct [num_envs, action_dim]
        
        2. UPDATE ROBOT STATE:
           - Set joint_vel = actions[:, :num_joints]
           - Update joint_pos += joint_vel * dt (dt=0.01)
           - Clamp joint_pos to [-1.0, 1.0]
           - Update ee_pos from first 3 joints: joint_pos[:, :3]
           
        3. COMPUTE REWARDS:
           - Call reward_computer.compute_rewards()
           - Extract total rewards
           - Accumulate to episode_rewards
           
        4. INCREMENT EPISODE TRACKING:
           - episode_steps += 1
           
        5. CHECK TERMINATION:
           - dones = episode_steps >= max_episode_steps
           
        6. AUTO-RESET:
           - Find done environment IDs
           - Call reset_environments() on those IDs
           
        7. RETURN:
           - observations from _get_observations()
           - rewards
           - dones
        """
        
        # Assert actions shape is correct [num_envs, action_dim]
        assert actions.shape == (self.config.num_envs, self.config.action_dim), f"Action shape got {actions.shape}, expected {(self.config.num_envs, self.config.action_dim)}" 

        # Update robot state
        self.robot_state.joint_vel = actions[:, :self.config.num_joints]
        self.robot_state.joint_pos += self.robot_state.joint_vel * 0.01 
        self.robot_state.joint_pos = torch.clamp(self.robot_state.joint_pos, min = -1.0, max = 1.0)
        # ToCheck______ee_post 
        self.robot_state.ee_pos = self.robot_state.joint_pos[:, :3]

        # Compute Rewards 
        self.rewardComputer.








    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        TODO: Get normalized observations for all environments
        
        Returns:
            Dict with 'state' key containing normalized state vector
            
        Use:
        - reward_computer.normalize_state() with update=True
        - Compute goal distance metric using compute_rewards
        """
        pass


# ============================================================================
# INTERVIEW QUESTIONS - Implement these tests to verify your solution
# ============================================================================

def test_basic_functionality():
    """Test 1: Basic tensor operations and shapes"""
    config = EnvConfig(
        num_envs=4,
        num_joints=6,
        action_dim=6,
        state_dim=29,  # 6*2 (joint pos/vel) + 3 (ee pos) + 4 (ee quat) + 3 (goal)
        device="cpu"
    )
    
    manager = MultiEnvironmentManager(config)
    obs = manager.reset_environments()
    
    # TODO: Verify and print:
    # 1. obs['state'] shape should be [4, 29]
    # 2. All values in obs['state'] should be finite (no NaN/Inf)
    # 3. robot_state.joint_pos shape should be [4, 6]
    # 4. Total number of tracked tensors
    
    print("✓ Test 1: Basic functionality")


def test_reward_computation():
    """Test 2: Verify reward values are in reasonable ranges"""
    config = EnvConfig(
        num_envs=8,
        num_joints=6,
        action_dim=6,
        state_dim=29,
        device="cpu"
    )
    
    manager = MultiEnvironmentManager(config)
    obs = manager.reset_environments()
    
    # TODO: Execute 5 random action steps and verify:
    # 1. rewards are finite (use torch.isfinite)
    # 2. rewards change based on actions (not all same)
    # 3. goal_distance_metric is in reasonable range
    # 4. success flags are boolean
    # 5. No error when computing rewards multiple times
    
    print("✓ Test 2: Reward computation")


def test_multiple_steps():
    """Test 3: Verify system stability over multiple steps"""
    config = EnvConfig(
        num_envs=16,
        num_joints=6,
        action_dim=6,
        state_dim=29,
        device="cpu"
    )
    
    manager = MultiEnvironmentManager(config)
    manager.reset_environments()
    
    # TODO: Run 100 steps with random actions and verify:
    # 1. No NaN/Inf values appear in observations or rewards
    # 2. Episode rewards accumulate correctly
    # 3. Environments reset properly when done (episode_steps goes back to 0)
    # 4. Running normalizer statistics update smoothly
    # 5. Success flag can be True at some point
    
    print("✓ Test 3: Multiple steps stability")


def test_vectorization_efficiency():
    """Test 4: Benchmark vectorized operations"""
    import time
    
    config = EnvConfig(
        num_envs=1024,
        num_joints=6,
        action_dim=6,
        state_dim=29,
        device="cpu"
    )
    
    manager = MultiEnvironmentManager(config)
    manager.reset_environments()
    
    # TODO: Measure and print:
    # 1. Time for single step() call (should be <100ms for 1024 envs on CPU)
    # 2. Time for 10 steps total
    # 3. Verify all 1024 environments process in parallel (no manual loops)
    
    print("✓ Test 4: Vectorization efficiency")


if __name__ == "__main__":
    # Run tests
    test_basic_functionality()
    test_reward_computation()
    test_multiple_steps()
    test_vectorization_efficiency()
    
    print("\n✓ All tests passed!")