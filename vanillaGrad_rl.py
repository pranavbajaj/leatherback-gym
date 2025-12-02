import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical 
import gymnasium as gym 
from dataclasses import dataclass
from typing import List, Tuple, Dict 
import torch.optim as optim

env = gym.make("LunarLander-v3", render_mode="human")


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
         


def generateReturn(rewards: List[float], config: EnvConfig, gamma: float = 0.99) -> torch.Tensor: 
    
    """_summary_
    Generates Returns for an epoch. 
    Args: 
        rewards: list of reward for all time step in a epoch 
        cofig: env config 
    
    Returns:
        Epoch returns: returns for a time step is cumulative rewards from that time step till termination  
    """

    returns = [] 
    G = 0 
    for reward in reversed(rewards): 
        G = reward + gamma * G
        returns.insert(0, G)
        
    return torch.tensor(returns, dtype = torch.float32, device=config.device)


# Configure env
env_config = EnvConfig(
    batch_size=256, 
    act_space=4,
    obs_dim=env.observation_space.shape[0], 
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
)


# 
batch_returns = torch.tensor([], dtype=torch.float32, device=env_config.device)  
batch_observations = torch.tensor([], dtype = torch.float32, device=env_config.device)
batch_actions = torch.tensor([], dtype = torch.long, device=env_config.device)

# Epoch Rewards 
epoch_rewards = []
epoch_actions = torch.tensor([], dtype = torch.long, device=env_config.device) 
epoch_observation = torch.tensor([], dtype = torch.float32, device=env_config.device)
epoch_counter = 0  

# RL Policy 
ActionPolicy = Policy(config= env_config, hidden_dim=32).to(env_config.device)

# Start sim env 
observation, info = env.reset(seed=42)


optimizer = optim.Adam(ActionPolicy.parameters(), lr=0.001)


while epoch_counter < 1000: 

    # actions =   
    obs_tensor = torch.tensor(observation, dtype = torch.float32, device = env_config.device)
    obs_tensor = obs_tensor.reshape(1, -1)
    epoch_observation = torch.cat([epoch_observation, obs_tensor], dim = 0)

    # Get action prob from policy
    action_prob = ActionPolicy(obs_tensor)
    dist = Categorical(probs=action_prob)
    action_tensor = dist.sample()
    epoch_actions = torch.cat([epoch_actions, action_tensor], dim = 0)
    action = action_tensor.item() 
    
    # Take next step im simulation 
    observation, reward, terminated, truncated, info = env.step(action)
    epoch_rewards.append(reward)

    if terminated or truncated: 
        
        observation, info = env.reset() 
        
        epoch_returns = generateReturn(epoch_rewards, env_config)
        print("Max returns: ", epoch_returns[0])
        batch_returns = torch.cat([batch_returns, epoch_returns], dim = 0)
        batch_actions = torch.cat([batch_actions, epoch_actions], dim = 0)
        batch_observations = torch.cat([batch_observations, epoch_observation], dim = 0)
        
        epoch_rewards = []
        epoch_actions = torch.tensor([], dtype = torch.int, device=env_config.device) 
        epoch_observation = torch.tensor([], dtype = torch.float32, device=env_config.device)
                
                
        print("BatchUpdate")
        
        if batch_observations.size()[0] >= env_config.batch_size: 
            # Set gradiance to zero 
            optimizer.zero_grad() 
            
            # Updated epoch counter 
            epoch_counter += 1 
        
            
            batch_action_prob = ActionPolicy(batch_observations)
            batch_dist = Categorical(probs=batch_action_prob)
            batch_log_action_prob = batch_dist.log_prob(batch_actions)
            print("----->Max returns: ")
            # Normalize returns 
            batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.var() + 1e-8)
            
            batch_loss = -torch.mean(batch_log_action_prob * batch_returns)
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(ActionPolicy.parameters(), max_norm=0.5)
            optimizer.step()
            
            print("------>EpochUpdate: ", epoch_counter)
            
            # Reset the env
            batch_returns = torch.tensor([], dtype=torch.float32, device=env_config.device)  
            batch_observations = torch.tensor([], dtype = torch.float32, device=env_config.device)
            batch_actions = torch.tensor([], dtype = torch.long, device=env_config.device)
            
    

env.close()