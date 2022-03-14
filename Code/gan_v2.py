from typing import Callable
from abc import ABC, abstractmethod
from torch import nn, optim, distributions

class GAN:
    def __init__(
        self,
        gen: nn.Module,
        dis: nn.Module,
        genoptim: optim.Optimizer,
        disoptim: optim.Optimizer,
        genStep: Callable[[nn.Module,optim.Optimizer],float], # input: net,optim, output loss, includes loss declaration, backwards, steps and zero grad
        disStep: Callable[[nn.Module,optim.Optimizer],float],
        trainloop: Callable[[genStep,disStep],None]
    ):
        ...
    def train(self):
        meta = self.trainingLoop(self.genStep,self.disStep)
        return meta

class Loop:
    def __init__():
        pass
    def startloop(self,k):
        pass
def togetherLoop():
    ...
def OneGenPerEpochLoop():
    ...
def KGenPerEpochLoop(k):
    ...

class TrainNetworkStep:
    def __init__(self,net,optimizer):
        pass
    def step(self):
        pass
def VanillaDiscriminatorStep():
    ...
def VanillaGeneratorStep():
    ...

def WassersteinCriticStep():
    ...
def WassersteinGeneratorStep():
    ...

def MixtureDensityGeneratorStep():
    ...
def MixtureDensityCriticStep():
    ...
