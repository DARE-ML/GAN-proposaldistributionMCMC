import torch
from abc import ABC

class Generator(nn.Module):
    pass
class Discriminator(nn.Module):
    pass


class VanillaTrain:
    def __init__(self, 
        epochs:         int, 
        goptim:         torch.optim.Optimizer,
        doptim:         torch.optim.Optimizer,
        generator:      torch.nn.Module, 
        discriminator:  torch.nn.Module, 
        dataloader:     torch.utils.data.Dataset,
        datadim:        tuple
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
        # should be inferred from data 
        self.datadim = datadim #(2,)

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
            lat = self.latent.sample(torch.Size([minibatch.shape[0],*self.datadim])).to(self.device)
            fake_batch = self.generator(lat)
            # sample from data
            ...
            # update disc
            y = torch.cat( (torch.ones([minibatch.shape[0],1]),torch.zeros([minibatch.shape[0],1])) )
            x = self.discriminator(torch.cat(minibatch.to(self.device),fake_batch))
            self.doptim.zero_grad()
            dloss = torch.nn.BCELoss(y,x)
            dloss.backward()
            self.doptim.step()
    def trainGen(self):
        # sample from latent
        lat = self.latent.sample(torch.Size([self.genBatchSize,*self.datadim])).to(self.device)
        fake_batch = self.generator(lat)
        # update gen
        y = torch.zeros([self.genBatchSize,1])
        x = self.discriminator(fake_batch)
        self.goptim.zero_grad()
        gloss = torch.nn.BCELoss(y,x)
        gloss.backward()
        self.goptim.step()
        
class WassensteinGPTrain(VanillaTrain):
    def trainDisc(self):
        pass
    def trainGen(self):
        pass
class MixtureSimplexTrain(VanillaTrain):
    pass