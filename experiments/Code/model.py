import torch
from torch import nn

# mine
class ResBlock(nn.Module):
    def __init__(self):
        self.seq = nn.Sequential(
            nn.Linear(),
        )
        self.skip = nn.Sequential(
            nn.Linear(),
        )
        self.act = nn.LeakyReLU(0.2)
    def forward(self,x):
        return self.act(self.seq(x)+self.skip(x))

class WeatherGenerator(nn.Module):
    def __init__(self):
        self.layer
    def forward(self, x):
        return self.act(self.seq(x)+self.skip(x))

#% copernicus paper
class ResIdentityCritic(nn.Module):
    def __init__(self,features):
        self.seq = nn.Sequential(
            nn.Conv2d(features,features,kernel_size = (1,1),stride =  1),
            nn.ReLU(),
            nn.ReplicationPad2d((1,0)),
            nn.Conv2d(features,features,kernel_size = (3,3), padding = (0,1),padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(features,features,kernel_size = (1,1))
        )
        self.act = nn.ReLU()
    def forward(self,x):
        return self.act(self.seq(x)+x)
class ResConvCritic(nn.Module):
    def __init__(self,inchan,outchan,stride=1):
        #2*inchan = outchan
        self.seq = nn.Sequential(
            nn.Conv2d(inchan,inchan,kernel_size = (1,1),stride = stride),
            nn.ReLU(),
            nn.ReplicationPad2d((1,0)),
            nn.Conv2d(inchan,inchan,kernel_size = (3,3), padding = (0,1),padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(inchan,outchan,kernel_size = (1,1)),
        )
        self.skip = nn.Sequential(
            nn.ReplicationPad2d((1,0)),
            nn.Conv2d(inchan,outchan,(3,3),stride = stride, padding = (0,1),padding_mode = 'circular')
        )
        self.act = nn.ReLU()
    def forward(self,x):
        return self.act(self.seq(x)+self.skip(x))
class WeatherCritic(nn.Module):
    def __init__(self):
        # input shape: (64,128,82)
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels = 82,out_channels = 128, kernel_size = (7,7),stride = 1), #,padding = (0,3),padding_mode = 'circular'),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size = (2,2)),
            ResConvCritic(128,256),
            ResIdentityCritic(256),
            ResIdentityCritic(256),
            ResConvCritic(256,512,stride=2),
            ResIdentityCritic(512,512),
            ResIdentityCritic(512),
            ResIdentityCritic(512),
            ResConvCritic(512,512,stride=2),
            ResIdentityCritic(512),
            ResIdentityCritic(512),
            ResIdentityCritic(512),
            ResIdentityCritic(512),
            nn.AvgPool2d(kernel_size = (2,2) ),
            nn.Flatten(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.Linear(100,1)
        )
    def forward(self,x):
        return self.seq(x)
class ResConvGen(nn.Module):
    def __init__(self,inchan, upsamp):
        
        layers = [
            nn.Batchnorm2d(inchan),
            nn.ReLU()
            
        ]
        self.seq1 = nn.Sequential(
            *layers
        )
    
class WeatherGen(nn.Module):
    def __init__(self):
        self.seq = nn.Sequential(
            nn.Linear(64,8*16*256),
            nn.Unflatten(1,(8,16,256)),
            ResConvGen(upsamp = True),
            ResConvGen(upsamp = True),
            ResConvGen(upsamp = True),
            ResConvGen(upsamp = False),
            ResConvGen(upsamp = False),
            nn.Conv2d(in_channels = 128,out_channels = 82, kernel_size =  (3,3), stride = 1,padding = (0,1),padding_mode = 'circular')
        )
    def forward(self,x):
        return self.seq(x)
# DeepClimateGAN