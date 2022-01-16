import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import time

class FC(nn.Module):
    # replica of Dr Chandra's code
    def __init__(self,layer_shapes,activation = nn.Sigmoid, out_act = nn.Sigmoid):
        super().__init__()
        self.topology = layer_shapes
        self.layerslist = []
        for i,x in enumerate(self.topology):
            if i != (len(self.topology)-1):
                self.layerslist.append(nn.Linear(*x))
                self.layerslist.append(activation())
            else:
                self.layerslist.append(nn.Linear(*x))
                self.layerslist.append(out_act())
        self.nn = nn.Sequential(*self.layerslist)

    def forward(self,x):
        return self.nn(x)
    def encode(self):
        return torch.cat([x.flatten() for x in self.parameters()])
    def decode(self,w):
        cumidx = 0
        for p in self.parameters():
            nneurons = torch.numel(p)
            p.data = w[cumidx:cumidx+nneurons].reshape(p.data.shape)
            cumidx += nneurons
class MCMC:
    def __init__(self,trainx,trainy,testx,testy, use_langevin,langevin_prob,learning_rate,n_full_batches,networktype = 'fc', hidden_size = [5] ):
        """ 
            Uses full data set for every sample.
            assumes that y is normally distributed with mean represented by nn model:
            unknowns:
                tau - sd of y
                w - weights and bias 
            hyperparameters - a,b, Sigma

            # in Dr Chandra's paper, theta = (weight,bias,tau (y sample variance) )
            # here it's different

            p(y_S|w,tau) = 1/(2pi tau^2)^(S/2) * exp(-(sum (y_t - f(x_t))^2)/(2tau^2))
            p(w1,...wn) ~ N(0_, Sigma_) , Sigma = covar matrix
            p(w1,...wn) ~ 1/(2pi sigma^2)^n exp(()/2)
            tau^2 ~ IGamma(a,b) => 1/tau ~ Gamma(a,b)
            now we want to simulate p(w|x,y,tau)
            mcmc steps:
                # the code by Dr Chandra uses log normal proposal for tau.

                for neww|tau,y,x 
                propose w*|wi ~ N(wi+sgd,sigma2)
                propose tau*|tau ~ LogNorm(tau,eta_step)
                calc diff in proposal probability: - -N(wi+sgd,sigma2)(w*)
                accept-reject
                a = min {1, [p(theta*|x)q(theta|theta*)] / [p(theta|x)q(theta*|theta)] }
                a = min {1, [p(theta*|x)/p(theta|x)]*[q(theta|theta*)/q(theta*|theta)] }
                a = min {1, [p(x|theta*)/p(x|theta)] * [p(theta*)/p(theta)] * [q(theta|theta*)/q(theta*|theta)] }
            
            p(x|theta) is reused from last iter without mini-batch, p(x|theta)
            
        Args:
            trainx ([ndarray]): [description]
            trainy ([ndarray]): [description]
            testx ([ndarray]): [description]
            testy ([ndarray]): [description]
            use_langevin ([bool]): [description]
            langevin_prob ([float32:[0,1])]): [description]
            learning_rate ([float32]): [description]
            n_full_batches ([int]): [description]
            networktype (str, optional): [description]. Defaults to 'fc'.
            hidden_size (list, optional): [not including input layer(data) size and output layer size]. Defaults to [5].
        """

        self.trainx = trainx if isinstance(trainx,torch.Tensor) else torch.from_numpy(trainx)
        self.trainy = trainy if isinstance(trainy,torch.Tensor) else torch.from_numpy(trainy)
        self.testx  = testx  if isinstance(testx,torch.Tensor)  else torch.from_numpy(testx)
        self.testy  = testy  if isinstance(testy,torch.Tensor)  else torch.from_numpy(testy)
        self.use_langevin = use_langevin
        self.langevin_prob = langevin_prob
        self.lr = learning_rate
        self.n_iter = n_full_batches

        self.networktype = networktype
        if networktype == 'fc':            
            self.network_shape = []
            self.n_weights = 0
            for i,x in enumerate(hidden_size):
                if i == 0:
                    self.network_shape.append((trainx.shape[-1],x))
                    self.n_weights += trainx.shape[-1]*x+x
                else:
                    self.network_shape.append((hidden_size[i-1],x))
                    self.n_weights += hidden_size[i-1]*x+x
                if i == (len(hidden_size)-1):
                    shape = (x,1 if testy.ndim == 1 else testy.shape[-1] )
                    self.network_shape.append(shape)
                    self.n_weights += shape[0]*shape[1]+shape[1]
            self.network = FC(self.network_shape)
            
            self.loss = nn.MSELoss()
            self.optimiser = torch.optim.SGD(self.network.parameters(),lr = learning_rate)
    def log_p_theta(self,params,sigma,tau2,v1,v2):
        # sigma not used because when calculating the ratio p()q(proposal|old), sigma is constant and cancels out
        return -torch.sum(params**2)/(2*sigma**2)-v2/tau2-(v1+1)*torch.log(tau2)
    def log_y_likelihood(self,tau2,meanloss,samplesize):
        # p(y|w,tau2) = [1/sqrt(2pi tau2) ]^batchsize * exp[-1/2 * sum[(y-f_w(x))^2] ]
        return -samplesize*torch.log(tau2)/2-samplesize*meanloss/(2*tau2)
    def sample(self, a,b,sigma,tau_prop_std,w_prop_std):
        """[summary]

        Args:
            a ([float64]): [param for tau prior]
            b ([float64]): [param for tau prior]
            sigma ([float64]): [param for w prior]
            tau_prop_std ([float64]): [sigma of log-normal proposal]
            w_prop_std ([float64]): [std of langevin normal proposal]
        """
        # uses default torch init for weights and sample variance as starter (i.e. null model)

        fx_test  = torch.zeros((self.n_iter,len(self.testy))) #
        fx_train = torch.zeros((self.n_iter,len(self.trainy))) # here assumes that y is one dimension
        theta_all        = torch.zeros((self.n_iter,self.n_weights+1))


        theta_all[0,:-1] = self.network.encode()

        fx_train[0,:] = self.network(self.trainx).flatten()

        theta_all[0,-1]  = torch.var(fx_train[0,:]-self.trainy)
        
        eta_last = torch.log(torch.var(fx_train[0,:] - self.trainy))
        tau2_last = theta_all[0,-1]
        w_last = theta_all[0,:-1]

        # first proposal mean
        loss = self.loss(self.trainy,fx_train[0,:])
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
        w_last_bar = self.network.encode()
        ln_y_w_last = self.log_y_likelihood(tau2_last,loss,self.trainy.shape[0])
        
        n_accept = 0
        for i in range(1,self.n_iter):
            
            # now we propose new w
            if self.use_langevin:
                # w_proposal = w_star = N(w_last_LD,sigma)
                w_star = torch.distributions.Normal(w_last_bar,w_prop_std).sample()
                
                self.network.decode(w_star.clone())
                propy = self.network(self.trainx)
                loss = self.loss(self.trainy,propy)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                w_star_bar = self.network.encode()
                
                # q(wi|w*)/q(w*|wi)
                log_proposal_ratio = (
                    torch.distributions.MultivariateNormal(w_star_bar,torch.eye(len(w_star_bar))*w_prop_std).log_prob(w_last) - 
                    torch.distributions.MultivariateNormal(w_last_bar,torch.eye(len(w_star_bar))*w_prop_std).log_prob(w_star)
                )
            else:
                w_star = torch.distribution.Normal(w[i-1,:],w_prop_std)
                self.network.decode(w_star.clone())
                propy = self.network(self.trainx)
                loss = self.loss(self.trainy,propy)
                
                # q(wi|w*)/q(w*|wi)
                log_proposal_ratio = 0
                
            # pi(w*|y)/pi(wi|y) NOTE: R.C's code sampled both tau and w via MH, i'm sampling only w, hence different prior
            # pi(w*|y) \sim  pi(w*)p(y|w*)
            eta_proposal = torch.normal(eta_last,tau_prop_std)
            tau2_proposal = torch.exp(eta_proposal)

            #log_proposal_ratio += (torch.distributions.LogNormal()) - (torch.distributions.LogNormal())
            log_prior_ratio = (
                self.log_p_theta(w_star,sigma=sigma,tau2=tau2_proposal, v1=a,v2=b) - 
                self.log_p_theta(w_last,sigma=sigma,tau2=tau2_last,     v1=a,v2=b)
            )
            
            ln_y_w_star = self.log_y_likelihood(tau2_proposal,loss,self.trainy.shape[0])
            log_likelihood_ratio = ln_y_w_star - ln_y_w_last
        
            # to calculate the accept reject we need 
            # min(1, pi(w*|x)/pi(wi|x) *  q(wi|w*)/q(w*|wi) )
            
            try:
                mh_prob = min(1, torch.exp(log_proposal_ratio + log_prior_ratio + log_likelihood_ratio))
            except OverflowError as e:
                mh_prob = 1

            u = torch.rand(1)

            if u < mh_prob:
                # Update position 
                n_accept += 1
                theta_all[i,:-1] = w_star
                theta_all[i,-1]  = tau2_proposal
                w_last =  w_star
                ln_y_w_last = ln_y_w_star
                eta_last = eta_proposal
                tau2_last = tau2_proposal

                #fx_test[i,]  =  fx_w_star_test.flatten()
                fx_train[i,] = propy.flatten()
                
            else:
                theta_all[i,] = theta_all[i-1,]

                #fx_test[i,] = fx_test[i-1,] 
                fx_train[i,]= fx_train[i-1,]
            if i%100 == 0:
                print(i,"Accepted: ",n_accept)
        accept_ratio = n_accept/self.n_iter
                # temporarily ignoring rmse statistic, just want to get thinks working
        return [theta_all,fx_train,fx_test,accept_ratio]

class FreqTrain:
    def __init__(self,trainx,trainy,testx,testy, use_langevin,langevin_prob,learning_rate,n_full_batches,networktype = 'fc', hidden_size = [5] ):
        self.trainx = trainx if isinstance(trainx,torch.Tensor) else torch.from_numpy(trainx)
        self.trainy = trainy if isinstance(trainy,torch.Tensor) else torch.from_numpy(trainy)
        self.testx  = testx  if isinstance(testx,torch.Tensor)  else torch.from_numpy(testx)
        self.testy  = testy  if isinstance(testy,torch.Tensor)  else torch.from_numpy(testy)
        self.use_langevin = use_langevin
        self.langevin_prob = langevin_prob
        self.lr = learning_rate
        self.n_iter = n_full_batches

        self.networktype = networktype
        if networktype == 'fc':            
            self.network_shape = []
            self.n_weights = 0
            for i,x in enumerate(hidden_size):
                if i == 0:
                    self.network_shape.append((trainx.shape[-1],x))
                    self.n_weights += trainx.shape[-1]*x+x
                else:
                    self.network_shape.append((hidden_size[i-1],x))
                    self.n_weights += hidden_size[i-1]*x+x
                if i == (len(hidden_size)-1):
                    shape = (x,1 if testy.ndim == 1 else testy.shape[-1] )
                    self.network_shape.append(shape)
                    self.n_weights += shape[0]*shape[1]+shape[1]
            self.network = FC(self.network_shape)
            
            self.loss = nn.MSELoss()
            self.optimiser = torch.optim.SGD(self.network.parameters(),lr = learning_rate)
    def train(self):
        for i in range(self.n_iter):
            propy = self.network(self.trainx)
            loss = self.loss(self.trainy,propy)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        self.network.eval()
        with torch.no_grad():
            plt.plot(self.trainy, label='actual')
            plt.plot(self.network(self.trainx), label='pred. (mean)')
            plt.savefig('freq_train_ldjoint.png') 
            plt.clf()
from torchinfo import summary
if __name__ == '__main__':
    traindata = np.loadtxt("./data/Sunspot/train.txt")
    testdata = np.loadtxt("./data/Sunspot/test.txt")  #
    name	= "Sunspot"
    trainx = torch.from_numpy(traindata[:,:-1]).type(torch.FloatTensor)
    trainy = torch.from_numpy(traindata[:,-1]).type(torch.FloatTensor).reshape((len(traindata),1))
    testx  = torch.from_numpy(testdata[:,:-1]).type(torch.FloatTensor)
    testy  = torch.from_numpy(testdata[:,-1]).type(torch.FloatTensor).reshape((len(testdata),1))
    
    if 1:
        num_samples = 100000
        lr = 0.5#0.01
        mcmc = MCMC(trainx,trainy,testx,testy,True,1,lr,num_samples,networktype='fc',hidden_size=[5])
        #print(mcmc.network)
        #print(trainx.shape)
        
        #batch_size = 10
        #summary(mcmc.network,input_size=(batch_size,4))

        #a = 0.1
        #b = 0.1
        #sigma = 0.025
        a = 0
        b = 0
        sigma = 5
        tau_prop_std = 0.01 #0.2
        w_prop_std   = 0.02
        [theta_all,fx_train,fx_test,accept_ratio] = mcmc.sample(a,b,sigma,tau_prop_std,w_prop_std)
        print(accept_ratio)
        bp = 1

        #plotting
        fx_mu_tr = fx_train[int(fx_train.shape[0]/2):].mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        plt.plot(trainy, label='actual')
        plt.plot(fx_mu_tr.detach().numpy(), label='pred. (mean)')
        plt.savefig('mcmcrestrain_ldjoint.png') 
        plt.clf()

        bp = 1
    if 0:
        lr = 0.5#0.01
        num_samples = 1000
        freq = FreqTrain(trainx,trainy,testx,testy,True,1,lr,num_samples,networktype='fc',hidden_size=[5])
        freq.train()
