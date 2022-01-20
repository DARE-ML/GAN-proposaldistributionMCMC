import torch 
from abc import ABC
class TargetDistribution(ABC):
    def __init__(self):
        self.samples_store

def core():
    pass
def proposal():
    pass
def prior():
    pass
def posterior():
    pass

class MetropolisHasting:
    def __init__(self,totalsamples,distributionclass):
        self.totalsamples = totalsamples
        self.target = distributionclass(self.totalsamples)
        
    def sampler():
        theta_last = self.target.init_theta()
        self.target.log_likelihood(theta_last)
        for i in range(1,self.totalsamples):
            # generate proposal theta* \sim q(theta|theta_last)
            theta_prop = 1

            log_proposal_ratio = ( 
                self.target.proposal(theta_prop).log_prob(theta_last) - 
                self.target.proposal(theta_last).log_prob(theta_prop)
            )
            log_prior_ratio = (
                self.target.prior().log_prob(theta_prop) - 
                self.target.prior().log_prob(theta_last)
            )
            log_likelihood_ratio = (
                self.target.log_likelihood(theta_prop) - 
                self.target.log_likelihood_iter(i-1)               
            )
            # u \sim U(0,1)
            u = torch.rand(1)

            #accept reject
            # min(1, pi(w*|x)/pi(wi|x) *  q(wi|w*)/q(w*|wi) )    
            # min(1, pi(x|w*)/pi(x|wi) * p(w*)/p(wi) *  q(wi|w*)/q(w*|wi) )    
            
            try:
                alpha = min(1, torch.exp(log_likelihood_ratio + log_proposal_ratio + log_prior_ratio))
            except OverflowError as e:
                alpha = 1
            if u < alpha:
                #accept proposal
                pass
            else:
                #reject proposal
                pass