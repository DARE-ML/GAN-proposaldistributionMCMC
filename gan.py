import torch

class Generator:
    pass
class Discriminator:
    pass


class VanillaTrain:
    def __init__(self,epochs, discstep, generator, discriminator, dataloader):
        
        # \frac{1}{m} \sum_{i=1}^m [log D(x^{(i)}) + log(1-D(G(z^(i))))  ]
        # can be expressed as a binary cross entropy loss as we have the y label of the latent as 0 and real as 1, with each item being
        # y_n log(x_n) + (1-y_n) log(1-x_n)
        # x_n = D(x)
        self.dLoss 
        self.gLoss

        self.optimizer
        self.latent = torch.distributions.Normal(0,1)

        self.epochs = epochs

        # either this or all minibatch
        #self.discStep = discstep
        
        self.dataloader = dataloader
        # should be inferred from data 
        self.datadim = (2,)
        self.batchsize = 64

        self.generator = generator
        self.discriminator = discriminator
    def train(self):
        for e in range(self.epochs):
            # train discriminator
            self.trainDisc()
            # train generator
            self.trainGen()
    def trainDisc(self):
        for b,minibatch in self.dataloader:
            # sample from latent
            lat = self.latent.sample(torch.Size([self.batchsize,*self.datadim]))
            x_fake = self.generator(lat)
            # sample from data
            ...
            # update disc

    def trainGen(self):
        # sample from latent
        # update gen
        ...
class WassensteinGPTrain:
    def train(self):
        pass
class MixtureSimplexTrain:
    pass