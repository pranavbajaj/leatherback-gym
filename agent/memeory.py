import numpy as np
import torch 
from typing import List, Dict, Tuple


class EnvObsMemoryBuffer: 

    def __init__(self, n_env: int = 1): 
        self.n_env: int = n_env  
        self.buffer: Dict[int, Dict[str, List]] = {
            env_id: {
                'img_obs': [],
                'steer_obs': [], 
                'vel_obs': []
            }
            for env_id in range(self.n_env)
        }

    def add(self, 
            obs_img: torch.Tensor, 
            obs_steer: torch.Tensor, 
            obs_val: torch.Tensor):

        for env_id in range(self.n_env): 
            self.buffer[env_id]['img_obs'].append(obs_img[env_id])
            self.buffer[env_id]['steer_obs'].append(obs_steer[env_id])
            self.buffer[env_id]['vel_obs'].append(obs_val[env_id])

    def reset(self, env_id: int): 

        self.buffer[env_id]['img_obs'] = [] 
        self.buffer[env_id]['steer_obs'] = [] 
        self.buffer[env_id]['vel_obs'] = [] 

    def get(self, env_id: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: 

        return self.buffer[env_id]['img_obs'], self.buffer[env_id]['steer_obs'], self.buffer[env_id]['vel_obs']
    

class EnvActMemoryBuffer: 
    def __init__(self, n_env: int = 1): 
        self.n_env = n_env
        self.buffer = {
            env_id: {
                'steer_act': [], 
                'vel_act': [] 
            }
            for env_id in range(self.n_env)
        }

    def add(self,
            steer_act: torch.Tensor, 
            vel_act: torch.Tensor): 
        
        for env_id in range(self.n_env): 
            self.buffer[env_id]['steer_act'].append(steer_act[env_id].item()) 
            self.buffer[env_id]['vel_act'].append(vel_act[env_id].item())

    def reset(self, env_id: int): 

        self.buffer[env_id]['steer_act'] = []
        self.buffer[env_id]['vel_act'] = [] 

    def get(self, env_id: int) -> Tuple[List[int], List[int]]: 
        return self.buffer[env_id]['steer_act'], self.buffer[env_id]['vel_act']
    


class EnvRewMemoryBuffer: 
    def __init__(self, n_env: int = 1): 
        self.n_env = n_env
        self.buffer = {
            env_id: {
                'rewards': []
            }
            for env_id in range(self.n_env)
        }

    def add(self,
            reward: torch.Tensor): 
        for env_id in range(self.n_env): 
            self.buffer[env_id]['rewards'].append(reward[env_id]) 

    def reset(self, env_id: int): 
        self.buffer[env_id]['rewards'] = []

    def get(self, env_id: int) -> List[torch.Tensor]: 
        return self.buffer[env_id]['rewards']
    

