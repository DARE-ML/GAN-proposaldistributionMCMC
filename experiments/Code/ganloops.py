import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from abc import ABC, abstractmethod
from typing import Callable
from torch import nn, optim
from torch.utils.data import DataLoader 
from Code.gan_v3 import NetWrapper
import torch
import pickle
import matplotlib.pyplot as plt
def trainForLosses(dataset_name, traindata,evaldata, genStruct, baseDisStruct, 
                  lossnsample, latent, epochs, lr, generateVisualFunc,
                   device, folder = './',ignore_label = False, gendisratio = 1,dim_reduct = None,repeats = 10):
    batch_size     = 64
    lr             = lr #0.0002
    epochs         = epochs
    #trainloader    = DataLoader(data.to(device), batch_size = batch_size,shuffle = True)
    trainloader    = DataLoader(traindata, batch_size = batch_size,shuffle = True)
    evalInterval = epochs//10
    for lossName in lossnsample:
        for repeat in range(repeats):
            # train
            gen = genStruct().to(device)
            print(lossName,lossnsample[lossName])
            discAct = lossnsample[lossName]['DiscFCAct']
            dis     = NetWrapper(baseDisStruct(64).to(device),discAct(64,1).to(device))
            goptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
            doptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.9)) 
            disstep,genstep = lossnsample[lossName]['backward'] 
            if ignore_label:
                trainingLoop = KDisLoopCollectFixMetaIgnoreLabel(
                    genstep, disstep, goptim, doptim, latent,
                    gen,dis,trainloader, epochs, device, eval_dataset = evaldata
                )
            else:
                trainingLoop = KDisLoopCollectFixMeta(
                    genstep, disstep, goptim, doptim, latent,
                    gen,dis,trainloader, epochs, device, eval_dataset = evaldata
                )
            # the savepath here is wrong (i.e. can't append gen dis before folder)
            # but since save is not called anyway so it runs fine
            trainingLoop.train(
                generateVisualFunc,evalInterval, 
                k = gendisratio, experiment_path = folder+lossName+"_repeat"+str(repeat)+"_",
                dim_reduct = dim_reduct
                # Wasserstein can go higher but for uniformity we choose 1 for comparison
            )
            # visualize the metastatistics

def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def VisHist2D(gen,latent,save_path):
        fake = gen(latent(1000)).cpu().detach().numpy()
        fig,ax = plt.subplots(1,1)
        ax.hist2d(fake[:,0],fake[:,1],bins=50)
        fig.savefig(save_path)
        plt.close('all')
def VisMNIST(gen,latent,save_path):
        fake = gen(latent(9)).cpu().detach().numpy()
        fig, ax = plt.subplots(3,3)
        plt.subplots_adjust(wspace=0, hspace=0)
        ax = ax.flatten()
        for i in range(9):
            ax[i].imshow(fake[i,0,:])
            ax.axis('off')
        fig.savefig(save_path)
        plt.close('all')
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
        dschedule = None,
        eval_dataset = None
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
        self.eval_dataset = eval_dataset
    @abstractmethod
    def train(self, vis = None, interval = None, visloss = None):
        ...

class KDisLoopCollectFixMeta(GANLoop):
    # Fix Meta includes:
    # loss (gen,dis)
    # avg grad (gen,dis)
    # first grad(gen,dis)
    # last grad(gen,dis)
    # evaluate score using and frechet
    def train(self, vis = None, interval = None,  
        k = 10, experiment_path = None, dim_reduct = None):
        self.meta = dict()
        for name in ['dloss','gloss','davggrad','gavggrad','dfgrad','gfgrad','dlgrad','glgrad']:
            self.meta[name] =  np.zeros(self.epochs*len(self.dataloader))
            self.meta[name][:] = np.nan
        self.meta['frechetIntervalEval'] = np.zeros(self.epochs//k)
        self.meta['frechetIntervalEval'][:] = np.nan

        for e in range(self.epochs):
            for i,batch in enumerate(self.dataloader):
                # train discriminator
                dloss = self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                self.meta['dloss'][e*len(self.dataloader)+i] = dloss.detach()
                total_grad = 0
                total_neur = 0
                for param in self.dis.parameters():
                    total_grad += param.grad.sum()
                    total_neur += np.prod(param.shape)
                self.meta['davggrad'][e*len(self.dataloader)+i] = (total_grad/total_neur)
                params = list(self.dis.parameters())
                self.meta['dfgrad'][e*len(self.dataloader)+i] = params[0].grad.mean()
                self.meta['dlgrad'][e*len(self.dataloader)+i] = params[-2].grad.mean()

                if i%k == 0:
                    # train generator
                    gloss = self.genStep(
                        self.gen,self.dis,self.goptim,self.latent,
                        batch.to(self.device),self.device
                    )
                    self.meta['gloss'][e*len(self.dataloader)+i] = gloss.detach()
                    total_grad = 0
                    total_neur = 0
                    for param in self.gen.parameters():
                        total_grad += param.grad.sum()
                        total_neur += np.prod(param.shape)
                    self.meta['gavggrad'][e*len(self.dataloader)+i] = (total_grad/total_neur)
                    params = list(self.dis.parameters())
                    self.meta['gfgrad'][e*len(self.dataloader)+i] = params[0].grad.mean()
                    self.meta['glgrad'][e*len(self.dataloader)+i] = params[-2].grad.mean()          

            if (e+1)%interval == 0:
                # save visual
                vis(self.gen,self.latent,experiment_path+"_generated_"+str((e+1)/interval)+".png")
                # evaluate using pca and frechet distance.
                sample_no = self.eval_dataset.shape[0]
                batch_no = int(np.ceil(sample_no/64))
                batchs = []
                for i in range(batch_no):
                    batchs.append( self.gen(self.latent(sample_no)).detach().to('cpu') )
                fake = torch.cat(batchs,dim=0)
                real_flat = torch.flatten(self.eval_dataset,start_dim=1).to('cpu').detach().numpy()
                fake_flat = torch.flatten(fake,start_dim=1).to('cpu').detach().numpy()
                if dim_reduct:
                    real_flat = dim_reduct(real_flat)
                    fake_flat = dim_reduct(fake_flat)
                # torch.cov # row are variable,col are obs
                fid = calculate_fid(real_flat,fake_flat)
                self.meta['frechetIntervalEval'][(e+1)//interval-1] = fid
                # perform mcmc sampling

                # visualize mcmc sampled 

                # evaluate using pca and frechet distance.
                
                # save_model
                if experiment_path:
                    torch.save(self.gen.state_dict(), experiment_path+"_gen_"+str((e+1)/interval)+'.pth')
                    torch.save(self.dis.state_dict(), experiment_path+"_dis_"+str((e+1)/interval)+'.pth')    
        with open(experiment_path+"_meta.pkl", 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
class KDisLoopCollectFixMetaIgnoreLabel(GANLoop):
    # Fix Meta includes:
    # loss (gen,dis)
    # avg grad (gen,dis)
    # first grad(gen,dis)
    # last grad(gen,dis)
    # evaluate score using and frechet
    def train(self, vis = None, interval = None,  
        k = 10, experiment_path = None, dim_reduct = None):
        self.meta = dict()
        for name in ['dloss','gloss','davggrad','gavggrad','dfgrad','gfgrad','dlgrad','glgrad']:
            self.meta[name] =  np.zeros(self.epochs*len(self.dataloader))
            self.meta[name][:] = np.nan
        self.meta['frechetIntervalEval'] = np.zeros(self.epochs//k)
        self.meta['frechetIntervalEval'][:] = np.nan

        for e in range(self.epochs):
            for i,(batch,_labels) in enumerate(self.dataloader):
                # train discriminator
                dloss = self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                self.meta['dloss'][e*len(self.dataloader)+i] = dloss.detach()
                total_grad = 0
                total_neur = 0
                for param in self.dis.parameters():
                    total_grad += param.grad.sum()
                    total_neur += np.prod(param.shape)
                self.meta['davggrad'][e*len(self.dataloader)+i] = (total_grad/total_neur)
                params = list(self.dis.parameters())
                self.meta['dfgrad'][e*len(self.dataloader)+i] = params[0].grad.mean()
                self.meta['dlgrad'][e*len(self.dataloader)+i] = params[-2].grad.mean()

                if i%k == 0:
                    # train generator
                    gloss = self.genStep(
                        self.gen,self.dis,self.goptim,self.latent,
                        batch.to(self.device),self.device
                    )
                    self.meta['gloss'][e*len(self.dataloader)+i] = gloss.detach()
                    total_grad = 0
                    total_neur = 0
                    for param in self.gen.parameters():
                        total_grad += param.grad.sum()
                        total_neur += np.prod(param.shape)
                    self.meta['gavggrad'][e*len(self.dataloader)+i] = (total_grad/total_neur)
                    params = list(self.dis.parameters())
                    self.meta['gfgrad'][e*len(self.dataloader)+i] = params[0].grad.mean()
                    self.meta['glgrad'][e*len(self.dataloader)+i] = params[-2].grad.mean()          

            if (e+1)%interval == 0:
                # save visual
                vis(self.gen,self.latent,experiment_path+"_generated_"+str((e+1)/interval)+".png")
                # evaluate using pca and frechet distance.
                sample_no = self.eval_dataset.shape[0]
                batch_no = int(np.ceil(sample_no/64))
                batchs = []
                for i in range(batch_no):
                    batchs.append( self.gen(self.latent(sample_no)).detach().to('cpu') )
                fake = torch.cat(batchs,dim=0)
                real_flat = torch.flatten(self.eval_dataset,start_dim=1).to('cpu').detach().numpy()
                fake_flat = torch.flatten(fake,start_dim=1).to('cpu').detach().numpy()
                # torch.cov # row are variable,col are obs
                fid = calculate_fid(real_flat,fake_flat)
                self.meta['frechetIntervalEval'][(e+1)//interval-1] = fid

                if dim_reduct:
                    real_flat_reduce = dim_reduct(real_flat)
                    fake_flat_reduce = dim_reduct(fake_flat)
                    reduce_fid = calculate_fid(real_flat_reduce,fake_flat_reduce)
                    self.meta['frechetIntervalEval_Reduce'][(e+1)//interval-1] = reduce_fid

                # save_model
                if experiment_path:
                    torch.save(self.gen.state_dict(), experiment_path+"_gen_"+str((e+1)/interval)+'.pth')
                    torch.save(self.dis.state_dict(), experiment_path+"_dis_"+str((e+1)/interval)+'.pth')    
        with open(experiment_path+"_meta.pkl", 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
class KDisLoopCollectFixMetaIgnoreLabel_OLD(GANLoop):
    def train(self, vis = None, interval = None, 
                visloss_gen = None, visloss_dis = None, 
                k = 10,save_model_path = None,
                avggrad_gen = None, avggrad_dis = None):
        self.dloss = visloss_dis
        self.gloss = visloss_gen
        self.avggrad_gen = avggrad_gen
        self.avggrad_dis = avggrad_dis
        for e in range(self.epochs):
            for i,(batch,_labels) in enumerate(self.dataloader):
                # train discriminator
                dloss = self.disStep(
                    self.gen,self.dis,self.doptim,self.latent,
                    batch.to(self.device),self.device
                )
                if dloss:
                    self.dloss.store(dloss.detach())
                if avggrad_dis:
                    total_grad = 0
                    total_neur = 0
                    for param in self.dis.parameters():
                        total_grad += param.grad.sum()
                        total_neur += np.prod(param.shape)
                    self.avggrad_dis.store(total_grad/total_neur)
                if i%k == 0:
                    # train generator
                    gloss = self.genStep(
                        self.gen,self.dis,self.goptim,self.latent,
                        batch.to(self.device),self.device
                    )
                    if gloss:
                        self.gloss.store(gloss.detach())
                    if avggrad_gen:
                        total_grad = 0
                        total_neur = 0
                        for param in self.gen.parameters():
                            total_grad += param.grad.sum()
                            total_neur += np.prod(param.shape)
                        self.avggrad_gen.store(total_grad/total_neur)
            if vis and (e+1)%interval == 0:
                vis(self.gen,self.latent)
                if save_model_path:
                    torch.save(self.gen.state_dict(), save_model_path+"_gen_"+str(e))
                    torch.save(self.dis.state_dict(), save_model_path+"_dis_"+str(e))                    
        if self.dloss:
            print("discriminator loss")
            self.dloss()
            print("generator loss")
            self.gloss()
            print("discriminator gradient")
            self.avggrad_dis()
            print("generator gradient")
            self.avggrad_gen()
