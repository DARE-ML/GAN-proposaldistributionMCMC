from typing import Callable
from abc import ABC, abstractmethod
from torch import nn, optim, distributions

class NetWrapper(nn.module):
    def __init__(self,base,out_act):
        self.base = base
        self.out_act = out_act
    def forward(self,x):
        return self.out_act(self.base(x))

class vanilla_disc_act(nn.Module):
    def __init__(self,input_dim):
        self.act = nn.Sequential(
            nn.Linear(input_dim,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.act(x)
class wgan_disc_act(nn.Module):
    def __init__(self,input_dim):
        self.act = nn.Linear(input_dim, 1)
    def forward(self,x):
        return self.act(x)
class mdgan_disc_act(nn.Module):
    def __init__(self,input_dim,out_dim):
        self.act = nn.Linear(input_dim,out_dim)
    def forward(self,x):
        return self.act(x)

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
        dataloader: torch.utils.data.Dataloader,
        epochs: int,
        device: torch.device
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
    @abstractmethod
    def train(self):
        ...

class TogetherLoop(GANLoop):
    def train(self):
        for e in range(self.epochs):
            for i,batch in enumerate(self.dataloader):
                # train discriminator
                self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                # train generator
                self.genStep(
                    self.gen,self.dis,self.goptim,self.latent,
                    batch.to(self.device),self.device
                )
class UnwantedLabelLoop(GANLoop):
    def train(self):
        for e in range(self.epochs):
            for i,(batch,_useless_label) in enumerate(self.dataloader):
                # train discriminator
                self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                # train generator
                self.genStep(
                    self.gen,self.dis,self.goptim,self.latent,
                    batch.to(self.device),self.device
                )
def OneGenPerEpochLoop():
    ...
def KGenPerEpochLoop(k):
    ...

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
    gloss = -self.discriminator(fake_batch).mean()
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
class MixtureDensityCriticStep:
    def __init__(
        self,
        mixture = MixtureGaus(disc_dimension=4,sigma_scale=0.25)
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
        mixture = MixtureGaus(disc_dimension=4,sigma_scale=0.25)
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