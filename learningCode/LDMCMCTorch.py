import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
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
        """ assumes that y is normally distributed with mean represented by nn model:
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
                # i want to use gibb sample for tau_i|w_i-1,y,x
                # the code by Dr Chandra uses normal proposal for tau instead that might be easier to evaluate.
                # both proposal distribution by R.C uses a step for std. 

                for neww|tau,y,x 
                propose w*|wi ~ N(wi+sgd,sigma2)
                calc diff in proposal probability: - -N(wi+sgd,sigma2)(w*)
                accept-reject
            
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
    def log_y_likelihood(self,w,tau2,x,y):
        self.network.decode(w.clone())
        predy = self.network(x)
        return [torch.log(torch.sqrt(tau2))*len(y)/2 - torch.sum( (y-predy)**2)/(2*tau2), predy ,self.loss(y,predy)]
    def sample(self, a,b,sigma):
        """[summary]

        Args:
            a ([type]): [param for tau prior]
            b ([type]): [param for tau prior]
            sigma ([type]): [param for w prior]
        """
        # uses default torch init for weights and sample variance as starter (i.e. null model)
        predY = torch.zeros((self.n_iter,len(self.trainy)))
        predY[0,:] = self.network(self.trainx).flatten()

        tau2     = torch.zeros(self.n_iter)
        tau2[0]  = torch.var(self.trainy) #np.random.rand(np.var(self.trainy))+0.00001
        w_all    = torch.zeros((self.n_iter,self.n_weights))
        w_all[0,:] = self.network.encode()
        fx_test  = torch.zeros((self.n_iter,len(self.testy))) #
        fx_train = torch.zeros((self.n_iter,len(self.trainy))) # here assumes that y is one dimension

        
        w_prior = torch.distributions.Normal(torch.zeros(w_all[0,:].shape),sigma)
        
        tau_posterior_a = self.n_weights + a + 1
        

        n_accept = 0
        for i in range(1,self.n_iter):
            tau_posterior_b = b + sum((self.trainy.flatten()-predY[i-1,:])**2)/2 # flatten to change from [n,1] to [n,]
            tau2[i] = 1/torch.distributions.Gamma(tau_posterior_a,tau_posterior_b).sample() # 1/ for inverse gamma
            
            w_last = w_all[i-1,:]
            # now we propose new w
            if self.use_langevin:
                # w + delta w : w+SGD
                self.network.decode(w_last.clone())
                loss = self.loss(self.trainy,self.network(self.trainx))
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                w_last_bar = self.network.encode()
                w_star = torch.distributions.Normal(w_last_bar,sigma).sample()
                
                self.network.decode(w_star.clone())
                loss = self.loss(trainy,self.network(self.trainx))
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                w_star_bar = self.network.encode()
                
                # q(wi|w*)/q(w*|wi)
                log_proposal_ratio = (
                    torch.distributions.MultivariateNormal(w_star_bar,torch.eye(len(w_star_bar))*sigma).log_prob(w_last) - 
                    torch.distributions.MultivariateNormal(w_last_bar,torch.eye(len(w_star_bar))*sigma).log_prob(w_star)
                )
            else:
                w_star = torch.distribution.Normal(w[i-1,:],sigma)
                # q(wi|w*)/q(w*|wi)
                log_proposal_ratio = 0
            # pi(w*|y)/pi(wi|y) NOTE: R.C's code sampled both tau and w via MH, i'm sampling only w, hence different prior
            # pi(w*|y) \sim  pi(w*)p(y|w*)
            log_prior_ratio      = torch.sum(w_prior.log_prob(w_star)-w_prior.log_prob(w_last)) # sum of log of independent weight, wont work with covar matrix
            
            [ln_y_w_star, fx_w_star, rmse_w_star] = self.log_y_likelihood(w_star.clone(), tau2[i], self.trainx,self.trainy)
            [ln_y_w_last, fx_w_last, rmse_w_last] = self.log_y_likelihood(w_last.clone(), tau2[i], self.trainx,self.trainy)
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
                [ln_y_w_star_test, fx_w_star_test, rmse_w_star] = self.log_y_likelihood(w_star.clone(), tau2[i], self.testx,self.testy)

                n_accept += 1
                w_all[i,] = w_star
                fx_test[i,]  =  fx_w_star_test.flatten()
                fx_train[i,] = fx_w_star.flatten()
                
            else:
                w_all[i,] = w_all[i-1,]

                fx_test[i,] = fx_test[i-1,] 
                fx_train[i,]= fx_train[i-1,]
        accept_ratio = n_accept/self.n_iter
                # temporarily ignoring rmse statistic, just want to get thinks working
        return [tau2, w_all,fx_train,fx_test,accept_ratio]
from torchinfo import summary
if __name__ == '__main__':
    traindata = np.loadtxt("./Code/Bayesian_neuralnetwork-LangevinMCMC/data/Sunspot/train.txt")
    testdata = np.loadtxt("./Code/Bayesian_neuralnetwork-LangevinMCMC/data/Sunspot/test.txt")  #
    name	= "Sunspot"
    trainx = torch.from_numpy(traindata[:,:-1]).type(torch.FloatTensor)
    trainy = torch.from_numpy(traindata[:,-1]).type(torch.FloatTensor).reshape((len(traindata),1))
    testx  = torch.from_numpy(testdata[:,:-1]).type(torch.FloatTensor)
    testy  = torch.from_numpy(testdata[:,-1]).type(torch.FloatTensor).reshape((len(testdata),1))

    num_samples = 1000
    mcmc = MCMC(trainx,trainy,testx,testy,True,1,0.01,num_samples,networktype='fc',hidden_size=[4])
    print(mcmc.network)
    #batch_size = 10
    #summary(mcmc.network,input_size=(batch_size,5))

    #a = 0.1
    #b = 0.1
    #sigma = 0.025
    a = 1
    b = 1
    sigma = 2

    [tau2, w_all,fx_train,fx_test,accept_ratio] = mcmc.sample(a,b,sigma)

    bp = 1
