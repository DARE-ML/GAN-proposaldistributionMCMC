# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from Code.gan import VanillaTrain_MNIST


# %%
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
#from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Define model Generator and Discriminator

# %%
# -*- coding: utf-8 -*-
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# %% [markdown]
# # Set up hyper param

# %%
G = generator()
D = discriminator()

batch_size = 64
lr = 0.0002
goptim = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
doptim = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Load data
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


train = VanillaTrain_MNIST(
    epochs          = 100,
    goptim          =goptim,
    doptim          =doptim,
    generator       = G, 
    discriminator   = D, 
    dataloader      = train_loader,
    latentdim       = (128,)
)


# %%
train.train()


# %%
for i,(batch_d,batch_y) in enumerate(train_loader):
    print(batch_d.shape)
    break


# %%



