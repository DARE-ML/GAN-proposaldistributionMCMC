# reimplements metropolis hasting wrt the factory method
# goal is to combines mhgan and normal mh code 
import torch, math
from abc import ABC,abstractmethod
from typing import Callable
class MetropolisHastings:
    def __init__(self,
        target_func: Callable[torch.Tensor,torch.FloatTensor], # evaluates the intensity/(scaled probability) of a sample
                     #DensityFunction Class?
        accept_prob_func: Callable[...,torch.FloatTensor], # accept or reject new samples
        n_chains: int = 1, # how many chains to run in parallel
    ):
        self.target_func = target_func
        self.accept_prob_func = accept_prob_func
        self.sample_store = torch.zeroes()
        ...
    def SequentialChainSampling(
        self
    ):
        self.sample_store[0,:] = ... # initial sample
        
        for i in range(1,self.iters):
            ...
    def ParallelChainSampling(
        self
    ):
        ...
    def CombinedChainSampling(
        self
    ):
        ...