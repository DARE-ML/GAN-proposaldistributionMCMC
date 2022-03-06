import torch, math
from abc import ABC,abstractmethod
class TargetDistribution(ABC):
    def __init__(self,totalsamples,dimension):
        # samples
        self.dimension = dimension
        self.totalsamples = totalsamples

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
    @abstractmethod
    def init_theta(self):
        pass

    def log_likelihood_iter(self,i):
        return self._hist_likelihoods[i]
    def last_sample(self):
        return self._samples_store[self._last_sample_index]
    def accept(self,newsample,newsamplelikelihood):
        self._last_sample_index += 1
        self._samples_store[self._last_sample_index] = newsample
        self._hist_likelihoods[self._last_sample_index] = newsamplelikelihood
        self._accepted_count += 1
    def reject(self):
        self._last_sample_index += 1
        self._samples_store[self._last_sample_index] = self._samples_store[self._last_sample_index-1]
        self._hist_likelihoods[self._last_sample_index] = self._hist_likelihoods[self._last_sample_index-1]
        

class Rastrigin(TargetDistribution):
    def __init__(self,totalsamples,dimension=2):
        super().__init__(totalsamples,dimension)
    def init_theta(self):
        first_sample = torch.Tensor([3,3])
        first_sample_likelihood = self.log_likelihood(first_sample)
        self.accept(first_sample,first_sample_likelihood)
    def log_likelihood(self, sample):
        A,PI = 10,torch.tensor(math.pi)
        
        #rastrigin
        #y = A*self.dimension+torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=0)

        # shifted, not normalized, inverted, rastrigin 
        y = -torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=0) + 36*self.dimension
        return torch.log(y)
class RastriginRandomWalkUnifPrior(Rastrigin):
    def proposal_dist(self, lastsample):
        # assume all points equally likely. UNIFORM
        #multunif = torch.distributions.Uniform(low = torch.Tensor([-5.14,-5.14]), high = torch.Tensor([5.14,5.14]))
        #return multunif

        # assume points from normal distriution with variance 4. RANDOM WALK
        multnorm = torch.distributions.Normal(loc = lastsample, scale = torch.Tensor([0.1,0.1]))
        return multnorm
    def prior(self):
        # assume all points between -5.14, 5.14 as equally likely
        multunif = torch.distributions.Uniform(low = torch.Tensor([-5.14*20,-5.14*20]), high = torch.Tensor([5.14*20,5.14*20]))
        return multunif

class RastriginRandomWalkNormPrior(Rastrigin):
    def __init__(self,totalsamples,dimension=2,proposal_std=0.1,prior_std=2):
        super().__init__(totalsamples,dimension)
        self.prop_std = proposal_std
        self.prior_std = prior_std
    def proposal_dist(self, lastsample):
        # assume points from normal distriution with variance 4. RANDOM WALK
        multnorm = torch.distributions.Normal(loc = lastsample, scale = torch.ones(self.dimension)*self.prop_std)
        return multnorm
    def prior(self):        
        # assume points from prior
        multnorm = torch.distributions.Normal(loc = torch.zeros(self.dimension), scale = torch.ones(self.dimension)*self.prior_std)
        return multnorm

from scipy.stats import truncnorm
class TruncNorm:
    def __init__(self, mean = 0, std = 1, low_bound = -float('inf'), up_bound= float('inf')):
        self.mylow = low_bound
        self.myup  = up_bound
        self.mean = mean
        self.std = std
        self.low, self.high = (self.mylow - mean) / std, (self.myup - mean) / std
    def sample(self):
        return torch.Tensor(truncnorm.rvs(a=self.low,b=self.high,loc=self.mean,scale = self.std))
    def log_prob(self,sample):
        return torch.Tensor(truncnorm.logpdf(sample,a=self.low,b=self.high,loc=self.mean,scale = self.std))
class RastriginTruncNorm(Rastrigin):
    def __init__(self,totalsamples,dimension=2,proposal_std=0.1,prior_std=2):
        super().__init__(totalsamples,dimension)
        self.prop_std = proposal_std
        self.prior_std = prior_std
    def proposal_dist(self, lastsample):
        # assume points from normal distriution with variance 4. RANDOM WALK
        truncmultnorm = TruncNorm(
            mean = lastsample,
            std = torch.ones(self.dimension)*self.prop_std, 
            low_bound= -5.14, up_bound = 5.14
        )
        return truncmultnorm 
    def prior(self):        
        # assume points from prior
        #multnorm = torch.distributions.Normal(loc = torch.zeros(self.dimension), scale = torch.ones(self.dimension)*2)
        truncmultnorm = TruncNorm(
            mean = torch.zeros(self.dimension),
            std = torch.ones(self.dimension)*self.prior_std, 
            low_bound= -5.14, up_bound = 5.14
        ) 
        return truncmultnorm
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
class GenericGAN:
    def __init__(self):
        pass
    def propose(self):
        pass
    def proposal_dist(self,condition):
        pass
class MetropolisHasting:
    def __init__(self,targetdistribution):
        self.target = targetdistribution #distributionclass(self.totalsamples)
    def performSample(self):
        # init first sample
        self.target.init_theta()
        
        for i in range(1,self.target.totalsamples):
            # generate proposal theta* \sim q(theta|theta_last)
            theta_last = self.target.last_sample()
            theta_prop = self.target.proposal_dist(theta_last).sample()

            log_proposal_ratio = ( 
                torch.sum(self.target.proposal_dist(theta_prop).log_prob(theta_last)) - 
                torch.sum(self.target.proposal_dist(theta_last).log_prob(theta_prop))
            )
            log_prior_ratio = (
                torch.sum(self.target.prior().log_prob(theta_prop)) - 
                torch.sum(self.target.prior().log_prob(theta_last))
            )
            prop_likelihood = torch.sum(self.target.log_likelihood(theta_prop)) 
            log_likelihood_ratio = (
                prop_likelihood - 
                torch.sum(self.target.log_likelihood_iter(i-1)  )             
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
                self.target.accept(theta_prop,prop_likelihood)
            else:
                #reject proposal
                self.target.reject()

class MetropolisHasting2GAN:
    def __init__(self,targetdistribution):
        self.target = targetdistribution #distributionclass(self.totalsamples)
    def performSample(self):
        # init first sample
        self.target.init_theta()
        
        for i in range(1,self.target.totalsamples):
            # generate proposal theta* \sim q(theta|theta_last)
            theta_last = self.target.last_sample()
            theta_prop = self.target.proposal_dist(theta_last).sample()

            log_proposal_ratio = ( 
                torch.sum(self.target.proposal_dist(theta_prop).log_prob(theta_last)) - 
                torch.sum(self.target.proposal_dist(theta_last).log_prob(theta_prop))
            )
            log_prior_ratio = (
                torch.sum(self.target.prior().log_prob(theta_prop)) - 
                torch.sum(self.target.prior().log_prob(theta_last))
            )
            prop_likelihood = torch.sum(self.target.log_likelihood(theta_prop)) 
            log_likelihood_ratio = (
                prop_likelihood - 
                torch.sum(self.target.log_likelihood_iter(i-1)  )             
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
                self.target.accept(theta_prop,prop_likelihood)
            else:
                #reject proposal
                self.target.reject()
            if i%1000 == 0:
                yield self.target
if __name__ == "__main__":
    for i in range(2):
        total_samples = 10000
        dist = RastriginRandomWalkNormPrior(total_samples,proposal_std=0.1,prior_std=2)
        mhsampler = MetropolisHasting(dist)
        mhsampler.performSample()
        print(mhsampler.target._last_sample_index)

        X = mhsampler.target._samples_store
        import matplotlib.pyplot as plt
        #plt.scatter(X[:, 0], X[:, 1])#, s=50, c = truth)
        #plt.title(f"Rastrigin sampling")
        #plt.xlabel("x")
        #plt.ylabel("y")
        #plt.show()

        x,y = X[:,0].numpy(), X[:,1].numpy()
        h =plt.hist2d(x, y,bins = 100)
        plt.colorbar(h[3])
        plt.show()
        bp = -1