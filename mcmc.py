import torch 
from abc import ABC,abstractmethod
class TargetDistribution(ABC):
    def __init__(self,totalsamples,dimension):
        # samples
        self._samples_store = torch.zeros(totalsamples,dimension)
        self._hist_likelihoods = torch.zeros(totalsamples)
        self._last_sample_index = -1
        # meta statistics
        self._accepted_count = 0
    @abstractmethod
    def proposal_dist(self,condition):
        pass
    @abstractmethod
    def prior(self):
        pass
    @abstractmethod
    def log_likelihood(self,sample):
        pass
    
    def init_theta(self):
        pass
    def log_likelihood_iter(self,i):
        return self._hist_likelihoods[i]
    def last_sample(self):
        return self._samples_store[self._last_sample_index]
    def accept(self,newsample):
        self._last_sample_index += 1
        self._samples_store[self._last_sample_index] = newsample

        self.accepted_count += 1
    def reject(self):
        self._last_sample_index += 1
        self._samples_store[self._last_sample_index] = self._samples_store[self._last_sample_index-1]
        

class Rastrigin(TargetDistribution):
    def __init__(self,totalsamples,dimension=2, proposal_type = "unif"):
        super().__init__(totalsamples,dimension)
        ... #unif or normal 
        self.proposal_type = proposal_type
    def proposal_dist(self, lastsample):
        # assume all points equally likely.
        return torch.distributions.Uniform(low = -5.14, high = 5.14)
        # assume points from normal distriution with variance 4.
        return torch.distributions.Normal(loc = lastsample, scale = torch.Tensor([2,2]))
    def prior(self):
        # assume all points between -5.14, 5.14 as equally likely
        return torch.distributions.Uniform(low = -5.14, high = 5.14)
    def log_likelihood(self, sample):
        y = A*n+torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=0)
        return torch.log(-y)
class Regress1(TargetDistribution):
    pass
class Regress1ParallelTempering(TargetDistribution):
    pass
class Normal2D(TargetDistribution):
    def __init__(self,totalsamples,dimension):
        self._samples_store = torch.zeros(totalsamples,dimension)
        self._hist_likelihoods = torch.zeros(totalsamples)
    def propose(self):
        #generate from uniform grid.
        return torch.rand(2)*10.24-5.12
    def propose2(self):
        # use random walk
        pass
    def proposal_dist(self, condition):
        # since its a constant, and only used for proportionality to calc ratio
        # use 1 lazily
        return 1
    def prior(self):
        # coincidencially identical to proposal
        pass
class MetropolisHasting:
    def __init__(self,totalsamples,distributionclass):
        self.totalsamples = totalsamples
        self.target = distributionclass(self.totalsamples)
        
    def performSample(self):
        theta_last = self.target.init_theta()
        self.target.log_likelihood(theta_last)
        for i in range(1,self.totalsamples):
            # generate proposal theta* \sim q(theta|theta_last)
            theta_prop = self.target.proposal_dist(last).sample()

            log_proposal_ratio = ( 
                self.target.proposal_dist(theta_prop).log_prob(theta_last) - 
                self.target.proposal_dist(theta_last).log_prob(theta_prop)
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
                self.target.accept(theta_prop)
            else:
                #reject proposal
                self.target.reject()

if __name__ == "__main__":
    mhsampler = MetropolisHasting(5000,Rastrigin)
    mhsampler.performSample()
    print(mhsampler.target._last_sample_index)