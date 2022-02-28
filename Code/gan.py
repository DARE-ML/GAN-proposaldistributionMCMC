import torch
from torch import nn
from abc import ABC, abstractmethod

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
            for b,minibatch in enumerate(self.dataloader):
                minibatch = minibatch.to(self.device)
                self.trainDisc(minibatch)
                self.trainGen(minibatch)
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
    def train(self):
        for e in range(self.epochs):
            for b,(minibatch,_unused_label) in enumerate(self.dataloader):
                minibatch = minibatch.to(self.device)
                self.trainDisc(minibatch)
                self.trainGen(minibatch)
                if (b+1)%100 == 0:
                    print("[",b,"/",len(self.dataloader),"]")
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
    def trainGen(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        self.goptim.zero_grad()
        gloss = -self.discriminator(fake_batch).mean()
        gloss.backward()
        self.goptim.step()
    # copied from https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
    def compute_gp(self, discriminator, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, *real_data.shape[1:]).to(real_data.device)
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
class MixtureSimplexTrain(VanillaTrainTogether):
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
    def trainGen(self,batch):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.latentdim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        self.goptim.zero_grad()
        gloss = -self.discriminator(fake_batch).mean()
        gloss.backward()
        self.goptim.step()