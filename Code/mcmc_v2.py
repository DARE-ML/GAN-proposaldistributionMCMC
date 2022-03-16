# reimplements metropolis hasting wrt the factory method
# goal is to combines mhgan and normal mh code 
import torch, math
import torch.distributions as D
from abc import ABC,abstractmethod
from typing import Callable

class AcceptRejectBlueprint(ABC):
    @abstractmethod
    def __call__(self,propose,last)->int:
        """_summary_

        Returns:
            int: count of accepted samples
        """
        pass
    
class ProposeFuncBlueprint(ABC):
    @abstractmethod
    def propose(self,last_sample):
        """Propose the next sample base or without the last, 
            e.g. random walk with last sample as mean, langevin gradient
        """
        pass
    @abstractmethod
    def log_prob(self,sample,cond_sample):
        pass
class TargetFuncBlueprint(ABC):
    @abstractmethod
    def eval_log(self,sample):
        pass

class MH_GAN_and_RW:
    def __init__(self, normalMH, ganloop, ganMH, cycles):
        self.normalMH = normalMH
        self.trainGAN = ganloop
        self.ganMH    = ganMH
        self.cycles   = cycles
    def startSampling(self):
        for cycle in range(cycles):
            samples = self.normalMH.startSampling()
            self.trainGAN()
            gensamples = self.ganMH.startSampling()
            # 1. rank gensamples by discriminator score
            # 2. select 3 subsets of gensamples: high, med, low rank by discriminator
            # 3. evaluate all 3 sets for real 
            # 4. select batch of good ones and use as initial samples of normalMH from next cycle
class MetropolisHastings:
    """ A Generic Factory for Metropolis Hasting with different:
            1. accept reject criteria
            2. number of chains per iteration
        Used together with:
            1. AcceptReject Callable
            2. ProposeFunc  Callable
            3. TargetFuc    Callable
        Hierarchy goes as this: 
            MH -> 
                propose, 
                accept_reject ->
                    propose_prob,
                    target_intensity

    """

    def __init__(self,
        #target_func:        Callable[torch.Tensor,torch.FloatTensor], # evaluates the intensity/(scaled probability) of a sample
        propose_func:       ProposeFuncBlueprint, # input: last sample, output: new sample [n_sample, *dim of single sample]
        accept_reject:      AcceptRejectBlueprint, # accept or reject new samples

        initial_sample:     torch.Tensor,   # n initial samples, n = n_chains
        device:             torch.device,

        iterations:         int = 1000,     # How many samples per chain
        n_chains:           int = 1,        # META FROM PROPOSE_FUNC: how many chains to run in parallel,       
    ):
        self.accept_reject = accept_reject
        self.propose_func = propose_func
        self.device = device
        self.iterations = iterations
        self.n_chains = n_chains
        self.sample_store = torch.zeros(iterations,*initial_sample.shape,device = device)
        self.sample_store[0,:] = initial_sample
    def startSampling(self):
        #print("store:",self.sample_store,self.sample_store.shape)
        last_sample = self.sample_store[0,:]
        #print("last sample:",last_sample,last_sample.shape)
        
        accept_no = 0
        total_no  = 0
        for i in range(1,self.iterations):
            new_sample = self.propose_func.propose(last_sample)
            #print("NEW SAMPLE:",new_sample)
            # NOTE: if return batch of entire last sample is 
            #       considered as accepting the past = rejecting propose.
            #       Hence, accepted 
            accept_count,accepted = self.accept_reject(new_sample,last_sample)
            #print("Accepted?:",accept)
            if accept_count:    
                last_sample = accepted
            self.sample_store[i,:] = accepted
            accept_no += accept_count
            total_no  += new_sample.shape[0]
        return self.sample_store, accept_no, total_no
#%% random walk for rastrigin
class MHAcceptReject(AcceptRejectBlueprint):
    """
        designed only to work with 1 chain
    """
    def __init__(
        self,
        propose_func: ProposeFuncBlueprint,
        target_func: TargetFuncBlueprint,
        device:     torch.device
    ):
        self.last    = None #  assumes of the dimension: [n_samples, *shape_of_a_sample]
        self.last_eval_log = -1 # 0  exp(-1)<1, i.e. first sample's probability is less than 1
        self.propose_func = propose_func
        self.target_func = target_func
        self.device = device
    def __call__(self,propose,last):
        #accept reject
        # min(1, pi(w*|x)/pi(wi|x) *  q(wi|w*)/q(w*|wi) )    
        # min(1, pi(x|w*)/pi(x|wi) * p(w*)/p(wi) *  q(wi|w*)/q(w*|wi) )    
        u = torch.rand(propose.shape[0],device = self.device)

        p_xp = self.target_func.eval_log(propose)  
        #print("proposal eval:",p_xp)
        p_xi = self.last_eval_log 
        q_xi_xp = self.propose_func.log_prob(last,propose) 
        q_xp_xi = self.propose_func.log_prob(propose,last)
        pre_alpha = torch.exp(p_xp-p_xi+q_xi_xp-q_xp_xi)
        #print("ln p:",p_xp,p_xi,q_xi_xp,q_xp_xi)
        #print("u,a:",u,pre_alpha)
        if u<min(1,pre_alpha):
            self.last = propose
            self.last_eval_log = p_xp
            return 1, propose
        else:
            return 0, last
class RandomWalkProposal(ProposeFuncBlueprint):
    def __init__(self,std,device,n_samples = 1):
        self.device = device
        self.std = std
        self.n_samples = n_samples
    def propose(self,last_sample):
        return torch.distributions.Normal(
            loc = last_sample,
            scale = self.std
        ).sample() # not .sample(n_samples) because it takes the shape of last_sample
    def log_prob(self,sample,cond_sample):
        # since its a constant, and only used for proportionality to calc ratio
        # use 1 lazily
        return 1


from scipy.stats import truncnorm
class TruncNorm:
    def __init__(self, mean = 0, std = 1, low_bound = -float('inf'), up_bound= float('inf')):
        self.mylow = low_bound
        self.myup  = up_bound
        self.mean = mean
        self.std = std
        self.low, self.high = (self.mylow - mean) / std, (self.myup - mean) / std
    def sample(self):
        return torch.Tensor(
            truncnorm.rvs(
                a=self.low,
                b=self.high,
                loc=self.mean,
                scale = self.std,
                size = (self.mean.shape) 
            )
        )
    def log_prob(self,sample):
        return torch.Tensor(truncnorm.logpdf(sample,a=self.low,b=self.high,loc=self.mean,scale = self.std))
class TruncNormProposal(ProposeFuncBlueprint):
    def __init__(self,std,device,dimension,low_bound = -5.14, up_bound= 5.14):
        self.device = device
        self.std = std
        self.dimension = dimension
        self.low_bound, self.up_bound = low_bound, up_bound
    def propose(self,last_sample):
        return TruncNorm(
            mean = last_sample,
            std = torch.ones(self.dimension)*self.std, 
            low_bound= self.low_bound, up_bound = self.up_bound
        ).sample().to(self.device) # not .sample(n_samples) because it takes the shape of last_sample
    def log_prob(self,sample,cond_sample):
        return torch.sum(TruncNorm(
            mean = cond_sample,
            std = torch.ones(self.dimension)*self.std, 
            low_bound= self.low_bound, up_bound = self.up_bound
        ).log_prob(sample),dim=-1).to(self.device)

class RastriginTarget(TargetFuncBlueprint):
    def __init__(self,dimension=2):
        self.dimension = dimension
    def eval_log(self,sample):
        A,PI = 10,torch.tensor(math.pi)
        
        #rastrigin
        #y = A*self.dimension+torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=0)
        if torch.any(torch.abs(sample)>=5.14):
            return torch.Tensor([-float("inf")])
        # shifted, not normalized, inverted, rastrigin 
        y = -torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=-1) + 36*self.dimension
        #print("EVAL LOG:")
        #print("sample:",sample,sample.shape)
        #print("y:",y,y.shape)
        #print("calc:",torch.square(sample)-A*torch.cos(2*PI*sample),(torch.square(sample)-A*torch.cos(2*PI*sample)).shape)
        #print("END EVAL LOG")
        return 3*torch.log(y)
class GausMix2(TargetFuncBlueprint):
    def __init__(self,dimension=2):
        self.dimension = dimension
        mu = torch.Tensor(
            [[2,2],
            [-2,-2]]
        )
        sigma = torch.Tensor(
            [[[2, 0.  ],
              [0.  , 2]],
            [[2, 0.  ],
              [0.  , 2]]]
        )

        gaus = D.Independent(D.MultivariateNormal(mu,sigma),0)
        c_weight = D.Categorical(torch.tensor(torch.ones(2)))
        self.target = gausMixture = D.MixtureSameFamily(c_weight,gaus)
    def eval_log(self,sample):
        return self.target.log_prob(sample)
class Gaus2DShift(TargetFuncBlueprint):
    def __init__(self):
        ...
    def eval_log(self,sample):
        return D.Normal(torch.Tensor([2,2]),torch.Tensor([1])).log_prob(sample)

#%% mhgan
class MHGANAcceptReject(AcceptRejectBlueprint):
    def __init__(self,discriminator,n_chains,device,calibrator):
        self.D = discriminator
        self.batch_size = self.n_chains = n_chains
        self.device = device
        self.last_batch = None
        self.last_prob = torch.zeros(self.batch_size)
        self.calibrator = calibrator
    def __call__(self,propose,last):
        """
        Args:
            propose (_type_): _description_
            last (_type_): _description_

        Returns:
            int: count of accepted
        """
        #accept reject  
        u = torch.rand(propose.shape[0],device = self.device)

        propose_prob = self.D(propose) if not self.calibrator else self.calibrator(self.D(propose))
        alpha = torch.min(
            torch.ones(len(propose_prob),device=self.device),
            (1.0/self.last_prob - 1)/(1.0/propose_prob - 1)  
        )

        self.last_batch = torch.where(
            #torch.cat([u.view(-1, 1), u.view(-1, 1)], dim=1) <= torch.cat([a.view(-1, 1), a.view(-1, 1)], dim=1),
            u.view(-1, 1) <= a.view(-1, 1),
            x_prop, x_last
        )
        #all_samples[iter,:] = x_last
        #print("LAST_PROB_unchange",last_prob.shape)
        #print(u.shape,a.shape)
        #print(u.view(-1,1).shape,a.view(-1, 1).shape)
        self.last_prob = torch.where(
            u <= a,
            prop_prob,last_prob  
        )
        return torch.sum(u<=a).item(), self.last_batch

def test_rastTruncNorm():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    proposal_rw = TruncNormProposal(std=0.5,device=device,dimension=2)
    rast_target = RastriginTarget(dimension = 2)
    initial_sample = torch.Tensor([[-4,-4]],device = device)

    mhar = MHAcceptReject(proposal_rw,rast_target, device)

    mh = MetropolisHastings(proposal_rw,mhar, initial_sample, device, iterations=1000)

    samples,accepted,total = mh.startSampling()
def test_GausMix():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    proposal_rw = RandomWalkProposal(2, device)
    gaus_target = GausMix2(dimension = 2)
    initial_sample = torch.Tensor([[0,0]],device = device)

    mhar = MHAcceptReject(proposal_rw,gaus_target, device)

    mh = MetropolisHastings(proposal_rw,mhar, initial_sample, device, iterations=10000)

    samples,accepted,total = mh.startSampling()
if __name__ == "__main__":
    #test_GausMix()
    test_rastTruncNorm()
# %%
