import torch 
from typing import List, Tuple, Dict, Optional 
from dataclasses import dataclass 

@dataclass
class EnvConfig: 
    nums_env: int 
    action_dim: int 
    state_dim: int 
    space_dim: int = 1
    device: str = "cpu"
    epsilon: float = 1e-8
    momentum: float = 0.9
    


class Car1D:
    def __init__(self, config: EnvConfig): 

        """
        Args: EnvConfig file 
        Init: Env config 
            device 
            car pose 
            car velocity 
            goal pose 
        """

        # Config and device 
        self._config = config 
        self._device = config.device

        # Init car poses and car vel: torch.tensor() -> [nums_env, space_dim]  
        self.car_pose = torch.zeros(config.nums_env, config.space_dim, dtype=torch.float32, device=config.device)
        self.car_vel = torch.zeros(config.nums_env, config.space_dim, dtype=torch.float32, device=config.device)

        # Init goal position variable 
        self.goal_pos = torch.zeros(config.nums_env, config.space_dim, dtype=torch.float32, device=config.device)

    def step(self, actions: torch.Tensor, dt: float = 0.1) -> None: 

        """ 
        Args: 
            Actions: torch.Tensor -> [nums_env, action_dim]
            dt: float -> scalar
        
            Update the velocity and pose of the car 
            vel_new= actions 
            pose_new = pose_old + vel_new * dt 

        return: None  

        """

        self.car_vel = actions 
        self.car_pose += self.car_vel * dt 

    def get_observation_vector(self,) -> torch.Tensor: 
        
        """
        Args: 
            None 
        Return: 
            car observations: Torch.tensor() -> [nums_env, state_dim]
        """

        return torch.cat([self.car_pose, self.car_vel, self.goal_pos], dim = -1)




class RunningNormalizer: 

    def __init__(self, config: EnvConfig): 

        """
        Args: 
            config: EnvConfig 
        Init: 
            config 
            device 
            epsilon (numerical stability while normalization)
            momentum (for running average of mean)
            mean
            var
            count 
        """

        # Save config, device and epsilon  
        self._config = config 
        self._device = config.device
        self._epsilon = config.epsilon
        self.momentum = config.momentum

        # mean and variance 
        self.running_mean = torch.zeros(config.state_dim, dtype=torch.float32, device=self._device)
        self.running_var = torch.ones(config.state_dim, dtype=torch.float32, device=self._device)

        # Count 
        self.count = torch.tensor(0.0, torch.float32, device = config.device)


    def update(self, states: torch.Tensor) -> None: 

        """
        Args: 
            states: torch.Tensor -> [batch_size, state_dim]

        Calculate: 
            running_mean 
            running_var 
            update mean and var 
            update count 
        
        """

        batch_mean = torch.mean(states, dim = 0, keepdim = True)
        batch_var = torch.var(states, dim = 0, keepdim = True)

        delta = batch_mean - self.running_mean
        batch_count = states.size()[0]
        total_count = self.count + batch_count 

        self.running_mean = self.running_mean * self.momentum + (batch_count / total_count) * delta 
        self.running_var = (self.running_var * self.count + batch_var * batch_count) / total_count

        self.count = total_count

    def normalize(self, states: torch.Tensor, ifUpdate: bool = True) -> torch.Tensor: 
        """
        Args: 
            states: torch.Tensor -> [batch_size, state_dim]
            ifUpdate: bool -> True to update running mean and running var 
        Return: 
            Normalized states: torch.Tensor -> [batch_size, state_dim]
        """

        if ifUpdate: 
            self.update(states)

        return (states - self.mean) / torch.sqrt(self.var + self._epsilon)


