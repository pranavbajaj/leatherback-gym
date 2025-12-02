import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical 
import gymnasium as gym 
from dataclasses import dataclass
from typing import List, Tuple, Dict 

env = gym.make("LunarLander-v3", render_mode="human")

device = ("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class EnvConfig: 
    """
    Config to vectorize env 
    """
    batch_size: int
    act_space: int
    obs_dim: int 
    device: str = "cpu"


# Action Policty 
class Policy(nn.Module): 
    """ 
    Policy: rl policy takes in obs input and gives actions 
    """
    def __init__(self, config: EnvConfig, hidden_dim: int): 
        super(Policy, self).__init__()

        # Setup device and config 
        self._config = config 
        self._device = config.device

        self.l1 = nn.Linear(config.obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, config.act_space)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x1 = F.relu(self.l1(x))
        x2 = self.softmax(self.l2(x1))

        return x2 
         



batch_rewards = [] 
batch_observations = []


# Configure env
env_config = EnvConfig(
    batch_size=64, 
    act_space=4,
    obs_dim=env.observation_space.shape[0]
)

# RL Policy 
ActionPolicty = Policy(config= env_config, hidden_dim=32)

# Start sim env 
observation, info = env.reset(seed=42)

for _ in range(1000): 

    # actions =   
    obs_tensor = torch.tensor(observation, dtype = torch.float32, device = device)
    obs_tensor = obs_tensor.reshape(1, -1)

    # Get action prob from policy
    action_prob = ActionPolicty(obs_tensor)

    action = env.action_space.sample()

    obesrvation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated: 
        observation, info = env.reset() 

    
    

env.close()