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
        burnin:             int = 100,   
        save_prediction: bool =False,
        prediction_shape        = []
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
        self.save_prediction = save_prediction
        if save_prediction:
            self.preds = torch.zeros(iterations+1+burnin,*prediction_shape)
    def startSampling(self,debug = False):
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
            if debug:
                print(i)
                print("sample shape:",new_sample.shape,last_sample.shape)
            #accept_count,accepted, score = self.accept_reject(new_sample,last_sample)
            out = self.accept_reject(new_sample,last_sample)
            if len(out) == 3:
                accept_count,accepted, score = out
            else:
                accept_count,accepted, score, pred = out

            #print("Accepted?:",accept)
            if accept_count:    
                last_sample = accepted

            if debug:
                print("accept:",accepted.shape)
                print("score:",score.shape)
            self.sample_store[i,:] = accepted
            self.target_eval[i,:]  = score
            if self.save_prediction:
                #print("preds",self.preds[i,:].shape,pred.shape)
                self.preds[i,:] = pred
            accept_no += accept_count
            total_no  += new_sample.shape[0]
        if self.save_prediction:
            return self.sample_store[1+self.burnin:,], accept_no, total_no, self.target_eval[1+self.burnin:,],self.preds[1+self.burnin:,]
        else:
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
        device:     torch.device,
        ini_pred: torch.tensor = None,
        debug = False
    ):
        self.last    = None #  assumes of the dimension: [n_samples, *shape_of_a_sample]
        self.last_eval_log = torch.tensor([-1],device = device) # 0  exp(-1)<1, i.e. first sample's probability is less than 1
        self.last_pred  = ini_pred
        self.propose_func = propose_func
        self.target_func = target_func
        self.device = device
        self.debug = debug
    def __call__(self,propose,last):
        #accept reject
        # min(1, pi(w*|x)/pi(wi|x) *  q(wi|w*)/q(w*|wi) )    
        # min(1, pi(x|w*)/pi(x|wi) * p(w*)/p(wi) *  q(wi|w*)/q(w*|wi) )  

        if self.debug:  
            print("shapes:",propose.shape,last.shape)
        u = torch.rand(propose.shape[0],device = self.device)

        target_eval = self.target_func.eval_log(propose)  
        if type(target_eval) is tuple:
            if self.debug:
                print("YES!!!!!!!!!!!!")
            p_xp, pred = target_eval
        else: 
            p_xp = target_eval
        #print("proposal eval:",p_xp)
        p_xi = self.last_eval_log 
        q_xi_xp = self.propose_func.log_prob(last,propose) 
        q_xp_xi = self.propose_func.log_prob(propose,last)
        #print(p_xp,p_xi,q_xi_xp,q_xp_xi)
        #print(p_xp.device,p_xi.device,q_xi_xp.device,q_xp_xi.device)
        pre_alpha = torch.exp(p_xp-p_xi+q_xi_xp-q_xp_xi)
        if self.debug:  
            print("--------------------------------")
            print("u,a:",u,pre_alpha)
            print("ln p:",p_xp,p_xi,q_xi_xp,q_xp_xi)
            print("score: p_xp, p_xi:",p_xp,p_xi)
            print("pred:shape:",self.last_pred.shape)
        #print(u.device,pre_alpha.device)
        if u<min(1,pre_alpha):
            self.last = propose
            self.last_eval_log = p_xp.clone().detach() 
            if type(target_eval) is tuple:
                self.last_pred = pred
                return 1, propose, p_xp, self.last_pred
            else:
                return 1, propose, p_xp
        else:
            if type(target_eval) is tuple:
                return 0, last,    p_xi, self.last_pred
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
        return torch.tensor([1],device = self.device)


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
    def __init__(self, device, dimension=2, tempering = 1, scale_x = 1):
        self.device = device
        self.dimension = dimension
        self.tempering = tempering
        self.scale_x = scale_x
    def eval_log(self,sample):
        A,PI = 10,torch.tensor(math.pi)
        
        #rastrigin
        #y = A*self.dimension+torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=0)
        if torch.any(torch.abs(sample)>=self.scale_x*5.12):
            return torch.tensor([-float("inf")],device = self.device)
        # shifted, not normalized, inverted, rastrigin 
        y = -torch.sum(torch.square(sample/self.scale_x)-A*torch.cos(2*PI*sample/self.scale_x),dim=-1) + 36*self.dimension
        #print("EVAL LOG:")
        #print("sample:",sample,sample.shape)
        #print("y:",y,y.shape)
        #print("calc:",torch.square(sample)-A*torch.cos(2*PI*sample),(torch.square(sample)-A*torch.cos(2*PI*sample)).shape)
        #print("END EVAL LOG")
        return self.tempering*torch.log(y)#.to(self.device)
class RastriginTargetIsolate(TargetFuncBlueprint):
    def __init__(self, device, dimension=2, tempering = 1, scale_x = 1):
        self.device = device
        self.dimension = dimension
        self.tempering = tempering
        self.scale_x = scale_x
    def eval_log(self,sample):
        A,PI = 10,torch.tensor(math.pi)
        
        #rastrigin
        #y = A*self.dimension+torch.sum(torch.square(sample)-A*torch.cos(2*PI*sample),dim=0)
        if torch.any(torch.abs(sample)>=self.scale_x*5.12):
            return torch.tensor([-float("inf")],device = self.device)
        # shifted, not normalized, inverted, rastrigin 
        y = torch.max(-torch.sum(torch.square(sample/self.scale_x)-A*torch.cos(2*PI*sample/self.scale_x),dim=-1) + 4*self.dimension,0)
        #print("EVAL LOG:")
        #print("sample:",sample,sample.shape)
        #print("y:",y,y.shape)
        #print("calc:",torch.square(sample)-A*torch.cos(2*PI*sample),(torch.square(sample)-A*torch.cos(2*PI*sample)).shape)
        #print("END EVAL LOG")
        return self.tempering*torch.log(y)#.to(self.device)
class Rosenbrock(TargetFuncBlueprint):
    def __init__(self,device,dimension = 2, tempering = 1):
        self.device = device
        self.dimension = dimension
        self.tempering = tempering
    def eval_log(self,sample):
        t1 = torch.square(sample[:,1:]-torch.square(sample[:,:-1]))
        t2 = torch.square(1-sample[:,:-1])
        #print(sample.shape,t1.shape,t2.shape)
        return self.tempering*torch.log(
            torch.max(
                -torch.sum(t1+t2,-1)+10*self.dimension,torch.zeros(sample.shape[0],1)
            )
        )
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
class MHGANPropose(ProposeFuncBlueprint):
    def __init__(self, gen, latent,batch_size):
        self.gen = gen
        self.latent = latent
        self.batch_size = batch_size
    def propose(self, last_sample):
        return self.gen(self.latent(self.batch_size)).detach()
    def log_prob(self,sample,cond_sample):
        # not used here
        pass
class MHGANAcceptReject(AcceptRejectBlueprint):
    def __init__(self,discriminator,n_chains,device,calibrator):
        self.D = discriminator
        self.batch_size = self.n_chains = n_chains
        self.device = device
        self.last_batch = None
        self.last_prob = torch.zeros(self.batch_size,device = self.device)
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
        #print(self.device,self.last_prob.device,propose_prob.device)
        alpha = torch.min(
            torch.ones(len(propose_prob),device=self.device),
            (1.0/self.last_prob - 1)/(1.0/propose_prob - 1)  
        )
        #print(alpha)
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
