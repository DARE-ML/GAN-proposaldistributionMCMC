import typing
import torch
from torch import nn, distributions


from torch import nn, optim, distributions
class MHGAN_paper:
    def __init__(self,
        gen:nn.Module,
        dis:nn.Module,
        gen_latent: callable,
        device: str,
        dis_cal: callable = None
    ):
        self.gen = gen
        self.gen_latent = gen_latent
        self.dis = dis
        # n chains:
        self.nchains = self.batch_size = 100
        self.device = device
        self.dis_cal = dis_cal
    def accept_prob(self,propose_prob,last_prob):
        # a = min(1,(D(xk)^-1 -1)/(D(x')^-1 -1) )
        #print("inside ACCEPT")
        #print(propose_prob.shape,last_prob.shape)
        
        return torch.min(torch.ones(len(propose_prob),device=self.device),(1.0/last_prob - 1)/(1.0/propose_prob - 1)  )
    def performSample_vectorized(self,initial_real_batch = None):
        # vectorized code logic drawn from https://github.com/obryniarski/mh-gan-experiments/blob/master/mh.py
        # the uber_research repo is too messy to make sense from in short term.
        # it essentially can be considered as running N metropolis hasting chains together, where N is the batch_size of each sample from G
        
        #self.target.init_theta(batch_size)
        #x_last = target.last_sample()
        ITER = 100

        ## HACKY, NOT GENERALIZED !!!!!!!!
        all_samples = torch.zeros(ITER,self.batch_size,2)
        print(all_samples.shape)
        print(len(self.gen_latent(1)),self.gen_latent(1))
        x_last = self.gen(self.gen_latent(self.batch_size)) if not initial_real_batch else initial_real_batch
        last_prob = self.dis_cal(self.dis(x_last)).view(-1) if self.dis_cal else self.dis(x_last).view(-1)
        all_samples[0,:] = x_last
        for iter in range(1,ITER):
            print("ITER:",iter)
            x_prop = self.gen(self.gen_latent(self.batch_size))
            #print("X",x_prop.shape)
            u = torch.rand(self.batch_size).to(self.device)
            #print("U",u.shape)
            prop_prob = self.dis_cal(self.dis(x_prop)).view(-1) if self.dis_cal else self.dis(x_prop).view(-1)
            #print("XP",prop_prob.shape)
            
            with np.errstate(divide='ignore'):
                a = self.accept_prob(prop_prob, last_prob)
            #print(a.shape,u.shape)
            x_last = torch.where(
                #torch.cat([u.view(-1, 1), u.view(-1, 1)], dim=1) <= torch.cat([a.view(-1, 1), a.view(-1, 1)], dim=1),
                u.view(-1, 1) <= a.view(-1, 1),
                x_prop, x_last
            )
            all_samples[iter,:] = x_last

            last_prob = torch.where(
                u <= a,
                prop_prob,last_prob  
            )

            __accept_count = torch.sum(u.view(-1, 1) <= a.view(-1, 1))
            print("ITERATION ACCEPT:",iter,__accept_count)
        return all_samples
        