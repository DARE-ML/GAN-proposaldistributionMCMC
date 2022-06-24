from typing import Callable
from abc import ABC, abstractmethod
from torch import nn, optim, distributions, autograd
import torch
import matplotlib.pyplot as plt
class NetWrapper(nn.Module):
    def __init__(self,base,out_act):
        super().__init__()
        self.base = base
        self.out_act = out_act
    def forward(self,x):
        return self.out_act(self.base(x))

class vanilla_disc_act(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.act = nn.Sequential(
            nn.Linear(input_dim,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.act(x)
class wgan_disc_act(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.act = nn.Linear(input_dim, 1)
    def forward(self,x):
        return self.act(x)
class mdgan_disc_act(nn.Module):
    def __init__(self,input_dim,out_dim):
        super().__init__()
        self.act = nn.Linear(input_dim,out_dim)
    def forward(self,x):
        return self.act(x)

class VisLoss:
    def __init__(self,counts):
        self.loss_store = torch.zeros(counts)
        self.idx = 0
    def store(self,loss):
        self.loss_store[self.idx] = loss
        self.idx += 1
    def __call__(self):
        fig = plt.figure(figsize=(14,2))
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.loss_store.cpu().detach().numpy())
        plt.show()

class GANLoop(ABC):
    def __init__(
        self,
        genStep: Callable[[nn.Module,optim.Optimizer],float], # input: net,optim, output loss, includes loss declaration, backwards, steps and zero grad
        disStep: Callable[[nn.Module,optim.Optimizer],float],
        goptim: optim.Optimizer,
        doptim: optim.Optimizer, 
        latent: Callable,
        gen: nn.Module,
        dis: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,
        gschedule = None,
        dschedule = None
    ):
        self.genStep = genStep 
        self.disStep = disStep 
        self.goptim = goptim 
        self.doptim = doptim 
        self.latent = latent 
        self.gen = gen 
        self.dis = dis 
        self.dataloader = dataloader 
        self.epochs = epochs 
        self.device = device
        self.gschedule = gschedule
        self.dschedule = dschedule
    @abstractmethod
    def train(self, vis = None, interval = None, visloss = None):
        ...
class AlternateEpochLoop(GANLoop):
    def train(self, vis = None, interval = None, visloss = None):
        if visloss:
            self.dloss = visloss(self.epochs*len(self.dataloader))
            self.gloss = visloss(self.epochs*len(self.dataloader))
        else:
            self.dloss = None
            self.gloss = None
        for e in range(self.epochs*2):
            if e%2:
                for i,batch in enumerate(self.dataloader):
                    # train discriminator
                    dloss = self.disStep(
                        self.gen,self.dis,self.doptim,self.latent,
                        batch.to(self.device),self.device
                    )
                    if self.dloss:
                        self.dloss.store(dloss.detach())
                if self.dschedule:
                    self.dschedule.step()
            else:
                for i,batch in enumerate(self.dataloader):
                    # train generator
                    gloss = self.genStep(
                        self.gen,self.dis,self.goptim,self.latent,
                        batch.to(self.device),self.device
                    )    
                    if self.gloss:
                        self.gloss.store(gloss.detach())
                if self.gschedule:    
                    self.gschedule.step()
            
            if vis and (e+1)%(interval*2) == 0:
                vis(self.gen,self.latent)
        if self.dloss:
            print("discriminator loss")
            self.dloss()
            print("generator loss")
            self.gloss()

class TogetherLoop(GANLoop):
    def train(self, vis = None, interval = None, visloss = None):
        if visloss:
            self.dloss = visloss(self.epochs*len(self.dataloader))
            self.gloss = visloss(self.epochs*len(self.dataloader))
        else:
            self.dloss = None
            self.gloss = None

        for e in range(self.epochs):
            for i,batch in enumerate(self.dataloader):
                # train discriminator
                dloss = self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                # train generator
                gloss = self.genStep(
                    self.gen,self.dis,self.goptim,self.latent,
                    batch.to(self.device),self.device
                )    
                if self.dloss:
                    self.dloss.store(dloss.detach())
                    self.gloss.store(gloss.detach())
            if self.gschedule:    
                self.gschedule.step()
            if self.dschedule:
                self.dschedule.step()
            
            if vis and (e+1)%interval == 0:
                vis(self.gen,self.latent)
        if self.dloss:
            print("discriminator loss")
            self.dloss()
            print("generator loss")
            self.gloss()

class UnwantedLabelLoop(GANLoop):
    def train(self, vis = None, interval = None, visloss = None):
        if visloss:
            self.dloss = visloss(self.epochs*len(self.dataloader))
            self.gloss = visloss(self.epochs*len(self.dataloader))
        else:
            self.dloss = None
            self.gloss = None

        for e in range(self.epochs):
            for i,(batch,_useless_label) in enumerate(self.dataloader):
                # train discriminator
                dloss = self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                # train generator
                gloss = self.genStep(
                    self.gen,self.dis,self.goptim,self.latent,
                    batch.to(self.device),self.device
                )
                if self.dloss:
                    self.dloss.store(dloss.detach())
                    self.gloss.store(gloss.detach())
            if vis and (e+1)%interval == 0:
                vis(self.gen,self.latent)
        if self.dloss:
            print("discriminator loss")
            self.dloss()
            print("generator loss")
            self.gloss()
def OneGenPerEpochLoop():
    ...
class KGenPerEpochLoop(GANLoop):
    def train(self, vis = None, interval = None, visloss = None):
        k = 10
        if visloss:
            self.dloss = visloss(self.epochs*len(self.dataloader))
            self.gloss = visloss(self.epochs*len(self.dataloader))
        else:
            self.dloss = None
            self.gloss = None

        for e in range(self.epochs):
            for i,batch in enumerate(self.dataloader):
                # train discriminator
                dloss = self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                if dloss:
                    self.dloss.store(dloss)
                
                if i%k == 0:
                    # train generator
                    gloss = self.genStep(
                        self.gen,self.dis,self.goptim,self.latent,
                        batch.to(self.device),self.device
                    )
                    if gloss:
                        self.gloss.store(gloss)

            if vis and (e+1)%interval == 0:
                vis(self.gen,self.latent)
        if self.dloss:
            print("discriminator loss")
            self.dloss()
            print("generator loss")
            self.gloss()

# currently assumes all latent of generator are normal

#%% vanilla step
def VanillaDiscriminatorStep(generator,discriminator,optimizer,latent,batch,device):
    fake_batch = generator(latent(batch.shape[0]).to(device))
    # update disc
    y = torch.cat( (torch.ones([batch.shape[0],1],device = device),torch.zeros([batch.shape[0],1],device = device)) )
    y.to(device)
    x = discriminator(torch.cat((batch.to(device),fake_batch)))
    optimizer.zero_grad()
    dloss = torch.nn.BCELoss()(x,y)
    dloss.backward()
    optimizer.step()
    return dloss
def VanillaGeneratorStep(generator,discriminator,optimizer,latent,batch,device):
    # sample from latent
    fake_batch = generator(latent(batch.shape[0]).to(device))    
    # update gen
    y = torch.zeros([batch.shape[0],1],device = device)
    x = discriminator(fake_batch)
    optimizer.zero_grad()
    gloss = -torch.nn.BCELoss()(x,y)
    gloss.backward()
    optimizer.step()
    return gloss
#%% wgan-gp
def WassersteinCriticStep(generator,discriminator,optimizer,latent,batch,device):
    # sample from latent
    fake_batch = generator(latent(batch.shape[0]).to(device))    
    # sample from data
    ...
    # update disc
    optimizer.zero_grad()

    dloss = ( 
        discriminator(fake_batch).mean() -
        discriminator(batch).mean() +
        __compute_gp(discriminator,batch,fake_batch)
    ) 
    dloss.backward()
    optimizer.step()
    return dloss
def WassersteinGeneratorStep(generator,discriminator,optimizer,latent,batch,device):
    # sample from latent
    fake_batch = generator(latent(batch.shape[0]).to(device)) 
    # update gen
    optimizer.zero_grad()
    gloss = -discriminator(fake_batch).mean()
    gloss.backward()
    optimizer.step()
    return gloss
def __compute_gp(discriminator, real_data, fake_data):
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
#%% mdgan
from .simplex import simplex_params
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
        self.gaus = distributions.Independent(distributions.MultivariateNormal(
            torch.tensor(self.mu).to(self.device),
            torch.tensor(self.sigma/0.25*sigma_scale).to(self.device)
        ),0)
        self.c_weight = distributions.Categorical(torch.tensor(self.weight).to(self.device))
        self.gausMixture = distributions.MixtureSameFamily(self.c_weight,self.gaus)

        self.findMax()

    def findMax(self):
        # the max of pdf should be close to one of the modes, i.e. use that as a starter point to optimize
        #maxiter = 2000 works for up to (9 dim 10 mode)     
        gaus_cpu = distributions.Independent(distributions.MultivariateNormal(
            torch.tensor(self.mu),
            torch.tensor(self.sigma/0.25*self.sigma_scale)
        ),0)
        c_weight_cpu = distributions.Categorical(torch.tensor(self.weight))
        gausMixture_cpu = distributions.MixtureSameFamily(c_weight_cpu,gaus_cpu)


        _xopt , fopt, _iter, _funcalls, _warnflag = optimize.fmin(lambda x:(-gausMixture_cpu.log_prob(torch.FloatTensor(x)) ), 
            torch.FloatTensor(self.mu[0,:]),maxiter = 2000,full_output=True) 
        # numerical tolerance
        self.num_tol = 1e-8  
        # maxval to be used for in loss function, it is the true prob, not log_prob
        self._lg_maxval = self.gausMixture.log_prob(torch.FloatTensor(_xopt).to(self.device))
        self.maxval = torch.exp(self.gausMixture.log_prob(torch.FloatTensor(_xopt).to(self.device))) + self.num_tol

class MixtureDensityCriticStep:
    def __init__(
        self,
        mixture #= MixtureGaus(disc_dimension=4,sigma_scale=0.25)
    ):
        self.mixture = mixture
    def __call__(self,generator,discriminator,optimizer,latent,batch,device):
        # sample from latent
        fake_batch = generator(latent(batch.shape[0]).to(device))    
        # sample from data
        ...
        # update disc
        optimizer.zero_grad()
        # this is a target to maximize so mult by -1
        dloss = - (
            self.mixture.gausMixture.log_prob(discriminator(batch)).mean() +
            torch.log(
                self.mixture.maxval - 
                #1-
                torch.exp(self.mixture.gausMixture.log_prob(discriminator(fake_batch)))
            ).mean()
        )
        dloss.backward()
        optimizer.step()
        return dloss
class MixtureDensityGeneratorStep:
    def __init__(
        self,
        mixture #= MixtureGaus(disc_dimension=4,sigma_scale=0.25)
    ):
        self.mixture = mixture
    def __call__(self,generator,discriminator,optimizer,latent,batch,device):
        # sample from latent
        fake_batch = generator(latent(batch.shape[0]).to(device))    
        # update gen
        optimizer.zero_grad()
        gloss = torch.log(
            self.mixture.maxval - 
            #1-
            torch.exp(self.mixture.gausMixture.log_prob(discriminator(fake_batch)))
        ).mean()
        gloss.backward()
        optimizer.step()
        return gloss