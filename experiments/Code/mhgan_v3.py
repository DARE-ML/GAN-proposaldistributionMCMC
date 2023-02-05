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