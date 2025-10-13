import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import models 


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
        
        self.fc = nn.Sequential(nn.Linear(1,128), 
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
        
        self.fc = nn.Sequential(nn.Linear(2,128),
                                nn.ReLU(), 
                                nn.Linear(128,256),
                                nn.ReLU(),
                                nn.Linear(256,512)
                                )
        
    def forward(self, x): 
        return self.fc(x)
    

class ActionDecoder(nn.Module): 
    
    def __init__(self, ):
        super(ActionDecoder, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(2024, 512),
                                nn.ReLU(), 
                                nn.Linear(512,128),
                                nn.ReLU(), 
                                nn.Linear(128,32),
                                nn.ReLU(), 
                                nn.Linear(32,2),
                                nn.Sigmoid()
                                )
        
    def forward(self, x1, x2, x3): 
        x = torch.cat((x1, x2, x3), dim = 1)
        x_out = self.fc(x)
        return x_out
        
        
class AgentNNetwork(nn.Module): 
    
    def __init__(self, ): 
        super(AgentNNetwork, self).__init__()
        
        self.perceptionEncoder = PerceptionEncoder()
        self.steerEncoder = SteerObsEncoder() 
        self.velocityEncoder = VelocityObsEncoder() 
        self.actionDecoder = ActionDecoder() 
        
    def forward(self, x_img, x_s, x_v):
        
        x1 = self.perceptionEncoder(x_img)
        x2 = self.steerEncoder(x_s)
        x3 = self.velocityEncoder(x_v)
        
        return self.actionDecoder(x1, x2, x3)
        
        
class DriveAgent():
    def __init__(self, lr = 0.001, gamma = 0.99):
        self.policy = AgentNNetwork() 
        self.optimizer = optim.Adagrad(self.policy.parameters(), lr = lr)
        self.gamma = gamma 
        
    def get_action(self, env, obs_manager):
        
        obs_img = obs_manager["policy"]["camera_obs"]
        obs_s = obs_manager["policy"]["steering_pos"]
        obs_v = obs_manager["policy"]["wheel_vel"]
        
        actions = self.policy(obs_img, obs_s, obs_v)
        # Converting it isaaclab env formate 
        actions_env = torch.zeros_like(env.action_manager.action, dtype=torch.float32)
        actions_env[:,:2] = actions[:,0]
        actions_env[:,2:] = actions[:,1]
        
        return actions_env 
        
    def compute_return(self, rewards):
        Returns = [0]
        for i in range(len(rewards)-1, -1, -1): 
            Returns.append(rewards[i] + self.gamma * Returns[-1])
        Returns.reverse()
        return Returns[:-1] 
        
    def update(self, log_probs, returns):
        pass 
    