import torch
from torch import nn
from abc import ABC, abstractmethod

###############################
#
# NEED TO REFACTOR WITH FACTORY PATTERN
#
###############################


class Generator(nn.Module):
    pass
class Discriminator(nn.Module):
    pass


class VanillaTrain:
    """_summary_
        Train Discriminator on full set
        then generator once
    """
    def __init__(self, 
        epochs:         int, 
        goptim:         torch.optim.Optimizer,
        doptim:         torch.optim.Optimizer,
        generator:      torch.nn.Module, 
        discriminator:  torch.nn.Module, 
        dataloader:     torch.utils.data.Dataset,
        latentdim:      tuple
    ):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.goptim = goptim
        self.doptim = doptim
        self.latent = torch.distributions.Normal(0,1)

        self.epochs = epochs
        self.genBatchSize = 128
        # either this or all minibatch
        #self.discStep = discstep
        
        self.dataloader = dataloader
        # self.datadim, inferred from dataloader 
        self.latentdim = latentdim

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
    # \frac{1}{m} \sum_{i=1}^m [log D(x^{(i)}) + log(1-D(G(z^(i))))  ]
    # can be expressed as a binary cross entropy loss as we have the y label of the latent as 0 and real as 1, with each item being
    # y_n log(x_n) + (1-y_n) log(1-x_n)
    # x_n = D(x)
    def train(self):
        for e in range(self.epochs):
            # train discriminator
            self.trainDisc()
            # train generator
            self.trainGen()
    def trainDisc(self):
        for b,minibatch in self.dataloader:
            # sample from latent
            lat = self.latent.sample(torch.Size([minibatch.shape[0],*self.latentdim])).to(self.device)
            fake_batch = self.generator(lat)
            # sample from data
            ...
            # update disc
            y = torch.cat( (torch.ones([minibatch.shape[0],1],device = self.device),torch.zeros([minibatch.shape[0],1],device = self.device)) )
            x = self.discriminator(torch.cat((minibatch.to(self.device),fake_batch)))
            self.doptim.zero_grad()
            dloss = torch.nn.BCELoss()(x,y)
            dloss.backward()
            self.doptim.step()
    
    def trainGen(self):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        y = torch.zeros([self.genBatchSize,1],device = self.device)
        x = self.discriminator(fake_batch)
        self.goptim.zero_grad()
        gloss = -torch.nn.BCELoss()(x,y)
        gloss.backward()
        self.goptim.step()
class VanillaTrain_label(VanillaTrain):
    """_summary_
        Train Discriminator on full set
        then generator once

        dataloader have unused label
    """
    def trainDisc(self):
        for b,(minibatch,_unused_class) in enumerate(self.dataloader):
            # sample from latent
            lat = self.latent.sample(torch.Size([minibatch.shape[0],*self.latentdim])).to(self.device)
            fake_batch = self.generator(lat)
            # sample from data
            ...
            # update disc
            y = torch.cat( (torch.ones([minibatch.shape[0],1],device = self.device),torch.zeros([minibatch.shape[0],1],device = self.device)) )
            y.to(self.device)
            x = self.discriminator(torch.cat((minibatch.to(self.device),fake_batch)))
            self.doptim.zero_grad()
            dloss = torch.nn.BCELoss()(x,y)
            dloss.backward()
            self.doptim.step()

class VanillaTrainTogether(VanillaTrain):
    """_summary_
        for each minibatch train discriminator and generator
    """

    def train(self):
        for e in range(self.epochs):
            for b,minibatch in enumerate(self.dataloader):
                minibatch = minibatch.to(self.device)
                dloss = self.trainDisc(minibatch)
                gloss = self.trainGen(minibatch)
    def trainDisc(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([batch.shape[0],*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # sample from data
        ...
        # update disc
        y = torch.cat( (torch.ones([batch.shape[0],1],device = self.device),torch.zeros([batch.shape[0],1],device = self.device)) )
        x = self.discriminator(torch.cat((batch,fake_batch)))
        self.doptim.zero_grad()
        dloss = torch.nn.BCELoss()(x,y)
        dloss.backward()
        self.doptim.step()
        return dloss
    def trainGen(self,batch):
        lat = self.latent.sample(torch.Size([batch.shape[0],*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        y = torch.zeros([batch.shape[0],1],device = self.device)
        x = self.discriminator(fake_batch)
        self.goptim.zero_grad()
        gloss = -torch.nn.BCELoss()(x,y)
        gloss.backward()
        self.goptim.step()
        return gloss
    #@abstractmethod
    #def view(self):
    #    pass

import matplotlib.pyplot as plt
import numpy as np
class VanillaTrainTogether_MNIST(VanillaTrainTogether):
    """_summary_
        for each minibatch train discriminator and generator

        have custom view during training for mnist
    """
    def __init__(self,
        epochs:         int, 
        goptim:         torch.optim.Optimizer,
        doptim:         torch.optim.Optimizer,
        generator:      torch.nn.Module, 
        discriminator:  torch.nn.Module, 
        dataloader:     torch.utils.data.Dataset,
        latentdim:      tuple
    ):
        super().__init__(epochs,goptim,doptim,generator,discriminator,dataloader,latentdim)
        self.viewFreq = 5 # epochs
    def train(self):
        for e in range(self.epochs):
            for b,(minibatch,_unused_label) in enumerate(self.dataloader):
                minibatch = minibatch.to(self.device)
                dloss = self.trainDisc(minibatch)
                gloss = self.trainGen(minibatch)
                if (b+1)%100 == 0:
                    print("[",b,"/",len(self.dataloader),"]"," Disc Loss: ",dloss.item(), " Gen Loss: ",gloss.item())
            if True:#(e+1)%self.viewFreq == 0:
                print("epoch:",e+1)
                self.view()
                
    def view(self):
        #start_time = time.time()
        plt.rcParams['image.cmap'] = 'gray'
        def show_images(images):
            sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

            for index, image in enumerate(images):
                plt.subplot(sqrtn, sqrtn, index+1)
                plt.imshow(image.reshape(28, 28))
            plt.show()
        fake_images = self.generator(self.latent.sample(torch.Size([16,*self.latentdim])).to(self.device) ).cpu().detach().numpy()
        fake_images = (fake_images+1.0)/2.0
        show_images(fake_images)

from torch import autograd
class WassensteinGPTogetherTrain_MNIST(VanillaTrainTogether_MNIST):
    def trainDisc(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([batch.shape[0],*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # sample from data
        ...
        # update disc
        self.doptim.zero_grad()
        dloss = ( 
            self.discriminator(fake_batch).mean() -
            self.discriminator(batch).mean() +
            self.compute_gp(self.discriminator,batch.to(self.device),fake_batch)
        ) 
        dloss.backward()
        self.doptim.step()
        return dloss
    def trainGen(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        self.goptim.zero_grad()
        gloss = -self.discriminator(fake_batch).mean()
        gloss.backward()
        self.goptim.step()
        return gloss
    # copied from https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
    def compute_gp(self, discriminator, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, *[1]*len(real_data.shape[1:])).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # get logits for interpolated images
        interp_logits = discriminator(interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
from .simplex import simplex_params
import torch.distributions as D 
from scipy import optimize
class MixtureGaus:
    def __init__(self,disc_dimension=9,sigma_scale = 0.25):
        # the number of modes/standard normal = number of vertices in a simplex:
        #   the same as disc_dimension + 1, i.e. 2d = triangle(3 vertices), 3d = tetrahedron(4 vertices), 
        # in the paper, the h-param are set as follow: 
        # disc_dimension = 9: -> 10 gaus mode 
        # sigma_scale = 0.25: -> individual std of 0.5
        # distance between each mode is 1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.sigma_scale = sigma_scale
        self.mu, self.sigma, self.weight = simplex_params(disc_dimension)
        # default sigma is independent and of scale 0.25, let it be smaller
        self.gaus = D.Independent(D.MultivariateNormal(
            torch.tensor(self.mu).to(self.device),
            torch.tensor(self.sigma/0.25*sigma_scale).to(self.device)
        ),0)
        self.c_weight = D.Categorical(torch.tensor(self.weight).to(self.device))
        self.gausMixture = D.MixtureSameFamily(self.c_weight,self.gaus)

        self.findMax()

    def findMax(self):
        # the max of pdf should be close to one of the modes, i.e. use that as a starter point to optimize
        #maxiter = 2000 works for up to (9 dim 10 mode)     
        gaus_cpu = D.Independent(D.MultivariateNormal(
            torch.tensor(self.mu),
            torch.tensor(self.sigma/0.25*self.sigma_scale)
        ),0)
        c_weight_cpu = D.Categorical(torch.tensor(self.weight))
        gausMixture_cpu = D.MixtureSameFamily(c_weight_cpu,gaus_cpu)


        _xopt , fopt, _iter, _funcalls, _warnflag = optimize.fmin(lambda x:(-gausMixture_cpu.log_prob(torch.FloatTensor(x)) ), 
            torch.FloatTensor(self.mu[0,:]),maxiter = 2000,full_output=True) 
        # numerical tolerance
        self.num_tol = 1e-8  
        # maxval to be used for in loss function, it is the true prob, not log_prob
        self._lg_maxval = self.gausMixture.log_prob(torch.FloatTensor(_xopt).to(self.device))
        self.maxval = torch.exp(self.gausMixture.log_prob(torch.FloatTensor(_xopt).to(self.device))) + self.num_tol

class MixtureSimplexTrainTogether_MNIST(VanillaTrainTogether_MNIST):    
    def __init__(self,
        epochs:         int, 
        goptim:         torch.optim.Optimizer,
        doptim:         torch.optim.Optimizer,
        generator:      torch.nn.Module, 
        discriminator:  torch.nn.Module, 
        dataloader:     torch.utils.data.Dataset,
        latentdim:      tuple,
        disc_dimension: int                         =9,
        sigma_scale:    float                       =0.25
    ):
        super().__init__(epochs,goptim,doptim,generator,discriminator,dataloader,latentdim)
        self.mixture = MixtureGaus(disc_dimension=disc_dimension,sigma_scale=sigma_scale)
    def trainDisc(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([batch.shape[0],*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # sample from data
        ...
        # update disc
        self.doptim.zero_grad()
        # this is a target to maximize so mult by -1
        dloss = - (
            self.mixture.gausMixture.log_prob(self.discriminator(batch)).mean() +
            torch.log(
                self.mixture.maxval - 
                #1-
                torch.exp(self.mixture.gausMixture.log_prob(self.discriminator(fake_batch)))
            ).mean()
        )
        dloss.backward()
        self.doptim.step()
        return dloss
    def trainGen(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        self.goptim.zero_grad()
        gloss = torch.log(
            self.mixture.maxval - 
            #1-
            torch.exp(self.mixture.gausMixture.log_prob(self.discriminator(fake_batch)))
        ).mean()
        gloss.backward()
        self.goptim.step()
        return gloss

class MixtureSimplexTrainTogether(VanillaTrainTogether):    
    def __init__(self,
        epochs:         int, 
        goptim:         torch.optim.Optimizer,
        doptim:         torch.optim.Optimizer,
        generator:      torch.nn.Module, 
        discriminator:  torch.nn.Module, 
        dataloader:     torch.utils.data.Dataset,
        latentdim:      tuple,
        disc_dimension: int                         =9,
        sigma_scale:    float                       =0.25
    ):
        super().__init__(epochs,goptim,doptim,generator,discriminator,dataloader,latentdim)
        self.mixture = MixtureGaus(disc_dimension=disc_dimension,sigma_scale=sigma_scale)
    def viewDensity(self):
        latent_sample = torch.randn(1000,2).to(self.device)
        fake = self.generator(latent_sample).detach().cpu().numpy()
        #plt.scatter(fake[:,0],fake[:,1])
        #plt.show()
        plt.hist2d(fake[:,0],fake[:,1],bins=50)
        plt.show()
    def train(self):
        for e in range(self.epochs):
            for b,minibatch in enumerate(self.dataloader):
                minibatch = minibatch.to(self.device)
                dloss = self.trainDisc(minibatch)
                gloss = self.trainGen(minibatch)
                #if (b+1)%100 == 0:
            if (e+1)%10 == 0:
                print("epoch:",e+1)
                print("[",b,"/",len(self.dataloader),"]"," Disc Loss: ",dloss.item(), " Gen Loss: ",gloss.item())
                self.viewDensity()
    def trainDisc(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([batch.shape[0],*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # sample from data
        ...
        # update disc
        self.doptim.zero_grad()
        # this is a target to maximize so mult by -1
        dloss = - (
            self.mixture.gausMixture.log_prob(self.discriminator(batch)).mean() +
            torch.log(
                self.mixture.maxval - 
                #1-
                torch.exp(self.mixture.gausMixture.log_prob(self.discriminator(fake_batch)))
            ).mean()
        )
        dloss.backward()
        self.doptim.step()
        return dloss
    def trainGen(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        self.goptim.zero_grad()
        gloss = torch.log(
            self.mixture.maxval - 
            #1-
            torch.exp(self.mixture.gausMixture.log_prob(self.discriminator(fake_batch)))
        ).mean()
        gloss.backward()
        self.goptim.step()
        return gloss