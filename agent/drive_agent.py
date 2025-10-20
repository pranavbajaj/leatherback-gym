import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import models 
from torch.distributions import *


class PerceptionEncoder(nn.Module): 

    def __init__(self,): 
        super(PerceptionEncoder, self).__init__()

        self.resnet = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.conv1 = self.resnet.conv1
        
        self.conv2 = nn.Sequential(self.resnet.bn1,
                                   self.resnet.relu,
                                   self.resnet.maxpool,
                                   self.resnet.layer1)
        
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4
        

    def forward(self, x): 
        
        x1 = self.resnet(x)

        return x1
    
    
class SteerObsEncoder(nn.Module): 
    
    def __init__(self,): 
        super(SteerObsEncoder, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(2,128), 
                                nn.ReLU(), 
                                nn.Linear(128, 256), 
                                nn.ReLU(),
                                nn.Linear(256, 512)
                                )
        
    def forward(self, x): 
        return self.fc(x)
    
class VelocityObsEncoder(nn.Module): 
    
    def __init__(self,): 
        super(VelocityObsEncoder, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(4,128),
                                nn.ReLU(), 
                                nn.Linear(128,256),
                                nn.ReLU(),
                                nn.Linear(256,512)
                                )
        
    def forward(self, x): 
        return self.fc(x)
    

class ActionDecoder(nn.Module): 
    
    def __init__(self):
        super(ActionDecoder, self).__init__()
        
        self.fc_steer = nn.Sequential(nn.Linear(2024, 512),
                                nn.ReLU(), 
                                nn.Linear(512,128),
                                nn.ReLU(), 
                                nn.Linear(128,32),
                                nn.ReLU(), 
                                nn.Linear(32,3),
                                nn.Softmax()
                                )
        
        self.fc_vel = nn.Sequential(nn.Linear(2024, 512),
                                nn.ReLU(), 
                                nn.Linear(512,128),
                                nn.ReLU(), 
                                nn.Linear(128,32),
                                nn.ReLU(), 
                                nn.Linear(32,2),
                                nn.Softmax()
                                )
        
    def forward(self, x1, x2, x3): 
        x = torch.cat((x1, x2, x3), dim = 1)
        x_steer_out = self.fc_steer(x)
        x_val_out = self.fc_vel(x)
        return x_steer_out, x_val_out
        
        
class AgentNNetwork(nn.Module): 
    
    def __init__(self,): 
        super(AgentNNetwork, self).__init__()
        
        self.perceptionEncoder = PerceptionEncoder()
        self.steerEncoder = SteerObsEncoder() 
        self.velocityEncoder = VelocityObsEncoder() 
        self.actionDecoder = ActionDecoder() 
        
        
    def forward(self, x_img, x_s, x_v):
        
        x1 = self.perceptionEncoder(x_img)
        x2 = self.steerEncoder(x_s)
        x3 = self.velocityEncoder(x_v)        
        x_steer_out, x_val_out = self.actionDecoder(x1, x2, x3)

        return x_steer_out, x_val_out
        
        
class DriveAgent():
    def __init__(self, lr = 0.001, gamma = 0.99):
        self.policy = AgentNNetwork() 
        self.optimizer = optim.Adagrad(self.policy.parameters(), lr = lr)
        self.gamma = gamma 
        
    def get_policy(self, obs_img, obs_s, obs_v):
        
        steer_action_logit, vel_action_logit = self.policy(obs_img, obs_s, obs_v)

        steer_dist = Categorical(steer_action_logit)
        vel_dist = Categorical(vel_action_logit)

        return steer_dist, vel_dist
    
    def get_action(self, obs_img, obs_s, obs_v): 
        steer_dist, vel_dist = self.get_policy(obs_img, obs_s, obs_v)
        return steer_dist.sample(), vel_dist.sample()
    
    def map_actions(self, steer_acts, vel_acts): 
        
        ## Revise this. steer_acts and vel_acts are not int, they are torch.tensor array of ints  
        
        steer_vals = []
        vel_vals = []

        for i in range(len(steer_acts)): 
            steer_val = 0.0 
            if steer_acts[i] == 0: 
                steer_val = -0.75
            elif steer_acts[i] == 2: 
                steer_val = 0.75

            vel_val = 0.0
            if vel_acts[i] == 1: 
                vel_val = 0.75

            steer_vals.append(steer_val)
            vel_vals.append(vel_val)
        
        return torch.tensor(steer_vals, dtype=torch.float32).unsqueeze(-1), torch.tensor(vel_vals, dtype=torch.float32).unsqueeze(-1)

        
    def compute_return(self, rewards):
        Returns = [0]
        for i in range(len(rewards)-1, -1, -1): 
            Returns.append(rewards[i] + self.gamma * Returns[-1])
        Returns.reverse()
        return Returns[:-1] 
        
    def update(self, steer_log_probs, vel_log_probs, returns):
        policy_loss = -(steer_log_probs * returns).mean()-(vel_log_probs * returns).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return policy_loss.item()
    
    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
