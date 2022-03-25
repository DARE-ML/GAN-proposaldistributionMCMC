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

def RW_to_GANMH():
    # hyperparam
    # n.o. data to train gan
    datapoints = 1000
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # collect samples from RW
    proposal_func   = RandomWalkProposal(std= 1,device=device,n_samples = 1)
    target_func     = Gaus64Target()
    accept_func     = MHAcceptReject(proposal_func, target_func,device)

    # neural network
    gen = base_generator(lat_len=2,out_len=3).to(device)
    dis = NetWrapper(
        base_discriminator(sample_len=3,out_len = 16),
        vanilla_disc_act(input_dim = 16)
    ).to(device)
    batch_size = 128
    lr = 0.0002
    goptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
    doptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
    latent = lambda batch_size: torch.randn(batch_size,2,device=device)

    ganpropose = MHGANPropose(gen, latent,batch_size=batch_size)
    ganaccept  = MHGANAcceptReject(dis,n_chains=batch_size,device=device,calibrator=None)
    mh_gan = MetropolisHastings(
        propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
        accept_reject = ganaccept, # accept or reject new samples

        initial_sample = torch.zeros(batch_size, 3 ,device=device),
        device      = device,

        iterations  = 500,  
        n_chains    = batch_size,        
    )

    s = torch.zeros(1,3)
    sc = torch.zeros(1,1)
    total_accept = 0
    total = 0
    while total_accept < datapoints*2:
        if total_accept == 0:
            burnin = datapoints
            init_s = torch.zeros(1, 3 ,device=device)
        else:
            burnin = 0
            init_s = torch.unsqueeze(s[-1,:],0)
        normalmh = MetropolisHastings(
            propose_func = proposal_func, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = accept_func, # accept or reject new samples

            initial_sample = init_s,
            device      = device,

            iterations  = datapoints,  
            n_chains    = 1,
            burnin = burnin    
        )
        mhsamp, _accept, _total, _score = normalmh.startSampling()    

        total_accept += _accept
        total += _total
        s = torch.cat((s,mhsamp.squeeze()),0)
        sc = torch.cat((sc,_score),0)

    # GAN OPTIMIZE
    ## select points
    s=s[::s.shape[0]//(datapoints*2),:].squeeze()
    sc = sc[::sc.shape[0]//(datapoints*2)].squeeze()
    score_idx = torch.argsort(sc,0)
    sorted_s = s[score_idx,:]
    sorted_ranked_lo_s = sorted_s[:datapoints//2,:]
    sorted_ranked_hi_s = sorted_s[-(datapoints-datapoints//2):,:]
    selected_samples = torch.cat((sorted_ranked_hi_s,sorted_ranked_lo_s),0)
    
    sorted_score_lo = sc[score_idx[:datapoints//2]]
    sorted_score_hi = sc[score_idx[-(datapoints-datapoints//2):]]
    selected_score = torch.cat((sorted_score_lo,sorted_score_hi),0)
    ##
    dataloader =  DataLoader(selected_samples, batch_size, shuffle=True)
    for i in range(10):
        gan.train()
        new_samples, _accept, _total, critic_score = gan.mhsample()
        new_samples_subset = selectsubset(new_samples,critic_score, lambda x: target_func.eval_log(x))
        new_samples_score  = target_func.eval_log(x)
        selected_samples, selected_score = mergesample(
            selected_samples, selected_score, 
            new_samples_subset, new_samples_score
        )
        dataloader = DataLoader(selected_samples, batch_size, shuffle=True)

def selectsubset(samples,critic_score, true_eval_func):
    # select 50 from top and bottom, compare mean
    score_idx = torch.argsort(critic_score)
    hi_avg = true_eval_func(samples[score_idx[-50:]]).sum()
    lo_avg = true_eval_func(samples[score_idx[:50]]).sum()
    if hi_avg > lo_avg:
        # high subset
        return samples[:samples.shape[0]//2]
    else:
        # unif distributed subset
        return samples[::2]

def mergesample(samples1,score1,samples2,score2):
    joinsample = torch.cat((samples1,samples2),0)
    joinscore = torch.cat((score1,score2))
    rank = torch.argsort(jointscore,0)
    return joinsamples[rank[-samples1.shape[0]:]]
    
class MH_GAN_and_RW:
    """
    usage e.g. with 3D gaus mixture:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        gen = base_generator(lat_len=2,out_len=3).to(device)
        dis = NetWrapper(
            base_discriminator(sample_len=3,out_len = 16),
            mdgan_disc_act(input_dim = 16,out_dim = 4)
        ).to(device)
        batch_size = 128
        lr = 0.0002
        goptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
        doptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
        latent = lambda batch_size: torch.randn(batch_size,2,device=device)

        traingan_loader = DataLoader(samples, batch_size=batch_size, shuffle=True)

        # !!! unique part about mdgan
        mg = MixtureGaus(disc_dimension=4,sigma_scale=0.25)
        traingan = TogetherLoop(
            MixtureDensityGeneratorStep(mg), MixtureDensityCriticStep(mg),
            goptim, doptim,
            latent,
            gen, dis,
            traingan_loader, 600, device
        )

        # evenly distributed points in range(-10,10)
        unif_samples = torch.rand(10000,3)*20-10 
        trainunif_loader = DataLoader(unif_samples,batch_size = batch_size, shuffle= True)
        unifgan = TogetherLoop(
            MixtureDensityGeneratorStep(mg), MixtureDensityCriticStep(mg),
            goptim, doptim,
            latent,
            gen, dis,
            trainunif_loader, 600, device
        )

        proposal_func   = RandomWalkProposal(std= 0.1,device=device,n_samples = 1))
        target_func     =    
        normalmh = MetropolisHastings(
            propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = ganaccept, # accept or reject new samples

            initial_sample = torch.zeros(batch_size, 3 ,device=device),
            device      = device,

            iterations  = 1000,  
            n_chains    = 1,    
        )
        
        ganpropose = MHGANPropose(gen, latent,batch_size=batch_size)
        ganaccept  = MHGANAcceptReject(dis,n_chains=batch_size,device=device,calibrator=None)
        mh_gan = MetropolisHastings(
            propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = ganaccept, # accept or reject new samples

            initial_sample = torch.zeros(batch_size, 3 ,device=device),
            device      = device,

            iterations  = 500,  
            n_chains    = batch_size,        
        )

        mh_mixin = MH_GAN_and_RW(
            normalMH    = normalmh,
            unifgan     = unifgan,
            ganloop     = traingan,
            ganMH       = ganmh,
            cycles      = 10,
        )

    """
    def __init__(self, normalMH, unifgan, ganloop, ganMH, cycles):
        self.normalMH = normalMH
        self.unifgan  = unifgan
        self.trainGAN = ganloop
        self.ganMH    = ganMH
        self.cycles   = cycles
        self.gan_sample_size = 2000
    def startSampling(self):
        # warmup
        _discard_samples = self.normalMH.startSampling()
        if self.unifgan:
            self.unifgan.train()
        for cycle in range(cycles):
            # maybe do parallel tempering here
            # generate enough samples for gan training
            samples_for_gan = torch.zeros_like(_discard_samples)
            accepted = 0
            while samples_for_gan.shape[0]<self.gan_sample_size:
                samples,_accept,_total, mhscore = self.normalMH.startSampling()
                samples_for_gan = torch.cat((samples_for_gan,samples),dim=0)
            # train gan on new samples
            self.trainGAN.dataloader = torch.utils.data.DataLoader(samples)
            self.trainGAN.train()

            gensamples,_accept,_total, genscore = self.ganMH.startSampling()
            # 1. rank gensamples by discriminator score
            # 2. select 3 subsets of gensamples: high, med, low rank by discriminator
            # 3. evaluate all 3 sets for real 
            # 4. select batch of good ones and use as initial samples of normalMH from next cycle
            # 5. how to use difference in rank to improve discriminator?
            score_idx = np.argsort(sc,0)
            sorted_s = s[score_idx,:]
            sorted_ranked_s = s[:1000,:]
            
            sortedsamples = self.criticSort(gensamples)
            subset = self.selectSubset(sortedsamples)
            score = self.evalSamples(subset)            


    def criticSort(self, samples, ):
        pass
    def selectSubset(self,samples):
        pass
    def evalSamples(self,subset):
        pass
    def selectNewPoints(self):
        pass


#%% others
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
        burnin:             int = 100   
    ):
        self.accept_reject = accept_reject
        self.propose_func = propose_func
        self.device = device
        self.iterations = iterations
        self.n_chains = n_chains
        self.sample_store = torch.zeros(iterations+1+burnin,*initial_sample.shape,device = device)
        self.sample_store[0,:] = initial_sample
        self.target_eval  = torch.zeros(iterations+1+burnin,n_chains)
        self.burnin = burnin
    def startSampling(self):
        #print("store:",self.sample_store,self.sample_store.shape)
        last_sample = self.sample_store[0,:]
        #print("last sample:",last_sample,last_sample.shape)
        
        accept_no = 0
        total_no  = 0
        for i in range(1,self.iterations+1+self.burnin):
            new_sample = self.propose_func.propose(last_sample)
            #print("NEW SAMPLE:",new_sample)
            # NOTE: if return batch of entire last sample is 
            #       considered as accepting the past = rejecting propose.
            #       Hence, accepted 
            accept_count,accepted, score = self.accept_reject(new_sample,last_sample)
            #print("Accepted?:",accept)
            if accept_count:    
                last_sample = accepted
            self.sample_store[i,:] = accepted
            self.target_eval[i,:]  = score
            accept_no += accept_count
            total_no  += new_sample.shape[0]
        return self.sample_store[1+self.burnin:,], accept_no, total_no, self.target_eval[1+self.burnin:,]
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
            return 1, propose, p_xp
        else:
            return 0, last,    p_xi
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
class MHGANPropose():
    def __init__(self, gen, latent,batch_size):
        self.gen = gen
        self.latent = latent
        self.batch_size = batch_size
    def propose(self, last_sample):
        return self.gen(self.latent(self.batch_size))
    def log_prob(self,sample,cond_sample):
        # not used here
        pass
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

        propose_prob = self.D(propose).squeeze() if not self.calibrator else self.calibrator(self.D(propose)).squeeze()
        alpha = torch.min(
            torch.ones(len(propose_prob),device=self.device),
            (1.0/self.last_prob - 1)/(1.0/propose_prob - 1)  
        )

        self.last_batch = torch.where(
            #torch.cat([u.view(-1, 1), u.view(-1, 1)], dim=1) <= torch.cat([a.view(-1, 1), a.view(-1, 1)], dim=1),
            u.view(-1, 1) <= alpha.view(-1, 1),
            propose, last
        )
        #all_samples[iter,:] = x_last
        #print("LAST_PROB_unchange",last_prob.shape)
        #print(u.shape,a.shape)
        #print(u.view(-1,1).shape,a.view(-1, 1).shape)
        self.last_prob = torch.where(
            u <= alpha,
            propose_prob,self.last_prob  
        )
        return torch.sum(u<=alpha).item(), self.last_batch, self.last_prob
#%% langevin gradient

class MiniBatchMH:
    """Do MH sampling of neural network weight as it converge to optimum
        the reason that this is done separately is because i can't figure 
        out how to incorporate the gradient info into proposal along with 
        new batch of data.
    """
    
    def __init__(self):
        ...
class LGNNProposal(ProposeFuncBlueprint):
    def __init__(self,std,net,accept_reject):
        self.std = std
        self.net = net
        self.accept_reject = accept_reject
    def propose(self,last_sample):
        self.__load_weights(last_sample)
        out = self.net(self.data)
        
        #last_sample_stepped
        return torch.distributions.Normal(
            loc = last_sample_stepped,
            scale = self.std
        ).sample()
    def log_prob(self,sample,cond_sample):
        return torch.sum(torch.distributions.Normal(
            loc = cond_sample,
            scale = self.std
        ).log_prob(sample))
    def __load_weights(self,weights):
        cumidx = 0
        for p in self.net.parameters():
            nneurons = torch.numel(p)
            p.data = w[cumidx:cumidx+nneurons].reshape(p.data.shape)
            cumidx += nneurons
class LGNNAcceptReject(AcceptRejectBlueprint):
    pass

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
