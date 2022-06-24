from mcmc_v2 import *
from gan_v2 import *
def RW_to_GANMH():
    # hyperparam
    # n.o. data to train gan
    datapoints = 1000
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # collect samples from RW
    proposal_func   = RandomWalkProposal(std= 1,device=device,n_samples = 1)
    target_func     = Gaus64Target()
    accept_func     = MHAcceptReject(proposal_func, target_func,device)
    # random sample for
    noise = lambda n: torch.rand(n,3)

    # neural network
    gen = base_generator(lat_len=2,out_len=3).to(device)
    dis = NetWrapper(
        base_discriminator(sample_len=3,out_len = 16),
        vanilla_disc_act(input_dim = 16)
    ).to(device)
    batch_size = 128
    lr = 0.001
    goptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
    doptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
    latent = lambda batch_size: torch.randn(batch_size,2,device=device)

    ganpropose = MHGANPropose(gen, latent,batch_size=batch_size)
    ganaccept  = MHGANAcceptReject(dis,n_chains=batch_size,device=device,calibrator=None)
    mhgan = MetropolisHastings(
        propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
        accept_reject = ganaccept, # accept or reject new samples

        initial_sample = torch.zeros(batch_size, 3 ,device=device),
        device      = device,

        iterations  = 500,  
        n_chains    = batch_size,        
    )

    s = torch.zeros(1,3)
    sc = torch.zeros(1,1)
    total_accept = 0
    total = 0
    while total_accept < datapoints*2:
        if total_accept == 0:
            burnin = datapoints
            init_s = torch.zeros(1, 3 ,device=device)
        else:
            burnin = 0
            init_s = torch.unsqueeze(s[-1,:],0)
        normalmh = MetropolisHastings(
            propose_func = proposal_func, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = accept_func, # accept or reject new samples

            initial_sample = init_s,
            device      = device,

            iterations  = datapoints,  
            n_chains    = 1,
            burnin = burnin    
        )
        mhsamp, _accept, _total, _score = normalmh.startSampling()    

        total_accept += _accept
        total += _total
        s = torch.cat((s,mhsamp.squeeze()),0)
        sc = torch.cat((sc,_score),0)

    # select points
    s=s[::s.shape[0]//(datapoints*2),:].squeeze()
    sc = sc[::sc.shape[0]//(datapoints*2)].squeeze()
    score_idx = torch.argsort(sc,0)
    sorted_s = s[score_idx,:]
    sorted_ranked_lo_s = sorted_s[:datapoints//2,:]
    sorted_ranked_hi_s = sorted_s[-(datapoints-datapoints//2):,:]
    selected_samples = torch.cat((sorted_ranked_hi_s,sorted_ranked_lo_s),0)
    
    sorted_score_lo = sc[score_idx[:datapoints//2]]
    sorted_score_hi = sc[score_idx[-(datapoints-datapoints//2):]]
    selected_score = torch.cat((sorted_score_lo,sorted_score_hi),0)
    
    # GAN OPTIMIZE
    dataloader =  DataLoader(selected_samples, batch_size, shuffle=True)
    for i in range(10):
        gan.train()
        new_samples, _accept, _total, critic_score = mhgan.startSampling()
        new_samples_subset = selectsubset(new_samples,critic_score, lambda x: target_func.eval_log(x))
        new_samples_score  = target_func.eval_log(x)
        selected_samples, selected_score = mergesample(
            selected_samples, selected_score, 
            new_samples_subset, new_samples_score
        )
        random_samples = noise(selected_samples.shape[0]//2)
        dataloader = DataLoader(selected_samples, batch_size, shuffle=True)

def selectsubset(samples,critic_score, true_eval_func):
    # select 50 from top and bottom, compare mean
    score_idx = torch.argsort(critic_score)
    hi_avg = true_eval_func(samples[score_idx[-50:]]).sum()
    lo_avg = true_eval_func(samples[score_idx[:50]]).sum()
    if hi_avg > lo_avg:
        # high subset
        return samples[:samples.shape[0]//2]
    else:
        # unif distributed subset
        return samples[::2]

def mergesample(samples1,score1,samples2,score2):
    joinsample = torch.cat((samples1,samples2),0)
    joinscore = torch.cat((score1,score2))
    rank = torch.argsort(jointscore,0)
    return joinsamples[rank[-samples1.shape[0]:]]


# prepared for 64 gan mode
def mhrwganroulette():
    # hyperparam
    # n.o. data to train gan
    datapoints = 3000
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # collect samples from RW
    proposal_func   = RandomWalkProposal(std= 1,device=device,n_samples = 1)
    target_func     = Gaus64Target()
    accept_func     = MHAcceptReject(proposal_func, target_func,device)
    # random sample for
    noise = lambda n: torch.rand(n,3)

    # neural network
    gen = base_generator(lat_len=2,out_len=3).to(device)
    dis = NetWrapper(
        base_discriminator(sample_len=3,out_len = 16),
        vanilla_disc_act(input_dim = 16)
    ).to(device)
    batch_size = 128
    lr = 0.001
    goptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
    doptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
    latent = lambda batch_size: torch.randn(batch_size,2,device=device)
    ganloop  = TogetherLoop(
        VanillaGeneratorStep, VanillaDiscriminatorStep,
        goptim, doptim,
        latent,
        gen, dis,
        train_loader, 100, device
    )

    ganpropose = MHGANPropose(gen, latent,batch_size=batch_size)
    ganaccept  = MHGANAcceptReject(dis,n_chains=batch_size,device=device,calibrator=None)
    

    s = torch.zeros(1,3)
    sc = torch.zeros(1,1)
    total_accept = 0
    total = 0
    while total_accept < datapoints*2:
        if total_accept == 0:
            burnin = datapoints
            init_s = torch.zeros(1, 3 ,device=device)
        else:
            burnin = 0
            init_s = torch.unsqueeze(s[-1,:],0)
        normalmh = MetropolisHastings(
            propose_func = proposal_func, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = accept_func, # accept or reject new samples

            initial_sample = init_s,
            device      = device,

            iterations  = datapoints,  
            n_chains    = 1,
            burnin = burnin    
        )
        mhsamp, _accept, _total, _score = normalmh.startSampling()    

        total_accept += _accept
        total += _total
        s = torch.cat((s,mhsamp.squeeze()),0)
        sc = torch.cat((sc,_score),0)

    # select points
    s=s[::s.shape[0]//(datapoints*2),:].squeeze()
    sc = sc[::sc.shape[0]//(datapoints*2)].squeeze()
    score_idx = torch.argsort(sc,0)
    sorted_s = s[score_idx,:]
    sorted_ranked_lo_s = sorted_s[:datapoints//2,:]
    sorted_ranked_hi_s = sorted_s[-(datapoints-datapoints//2):,:]
    selected_samples = torch.cat((sorted_ranked_hi_s,sorted_ranked_lo_s),0)
    
    sorted_score_lo = sc[score_idx[:datapoints//2]]
    sorted_score_hi = sc[score_idx[-(datapoints-datapoints//2):]]
    selected_score = torch.cat((sorted_score_lo,sorted_score_hi),0)
    
    # roulette
    dataloader =  DataLoader(selected_samples, batch_size, shuffle=True)
    for i in range(10):
        print("Cycle:",i)
        interval = 10
        ganloop.train(Vis3d(),interval,VisLoss)

        mhgan = MetropolisHastings(
            propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = ganaccept, # accept or reject new samples

            initial_sample = torch.zeros(batch_size, 3 ,device=device),
            device      = device,

            iterations  = 100,  
            n_chains    = batch_size,        
        )
        new_samples, _accept, _total, critic_score = mhgan.startSampling(i)
        # merge chains
        new_samples = new_samples.view(-1,3) # 3 is problem dependent
        critic_score = critic_score.view(-1,1)
        #select samples 
        score_idx = torch.argsort(critic_score,0)
        hiloprob_sample = new_samples[
            torch.cat((score_idx[:10],score_idx[-10:]),0)
        ]
        
        new_rw_sample = None
        local_total = 0
        local_accept = 0
        for i in range(20):
            normalmh = MetropolisHastings(
                propose_func = proposal_func, # input: last sample, output: new sample [n_sample, *dim of single sample]
                accept_reject = accept_func, # accept or reject new samples

                initial_sample = hiloprob_sample[i],
                device      = device,

                iterations  = 500,  
                n_chains    = 1,
                burnin = burnin    
            )
            mhsamp, _accept, _total, _score = normalmh.startSampling()    
            local_total += _total
            local_accept += _accept
            if new_rw_sample is None:
                new_rw_sample = mhsample
            else:
                new_rw_sample = torch.cat((new_rw_sample,mhsamp),0)
        #downsample
        if local_accept/local_total < 0.8:
            new_rw_downsample = new_rw_sample[::new_rw_sample.shape[0]//(local_total/local_accept)]
            selected_samples = torch.cat((selected_samples,new_rw_downsample),0)[-datapoints:]
        else:
            selected_samples = torch.cat((selected_samples,new_rw_sample),0)[-datapoints:]
        dataloader = DataLoader(selected_samples, batch_size, shuffle=True)
    return mhgan
class MH_GAN_and_RW:
    """
    usage e.g. with 3D gaus mixture:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        gen = base_generator(lat_len=2,out_len=3).to(device)
        dis = NetWrapper(
            base_discriminator(sample_len=3,out_len = 16),
            mdgan_disc_act(input_dim = 16,out_dim = 4)
        ).to(device)
        batch_size = 128
        lr = 0.0002
        goptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
        doptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999)) # betas=(0,0.9))#
        latent = lambda batch_size: torch.randn(batch_size,2,device=device)

        traingan_loader = DataLoader(samples, batch_size=batch_size, shuffle=True)

        # !!! unique part about mdgan
        mg = MixtureGaus(disc_dimension=4,sigma_scale=0.25)
        traingan = TogetherLoop(
            MixtureDensityGeneratorStep(mg), MixtureDensityCriticStep(mg),
            goptim, doptim,
            latent,
            gen, dis,
            traingan_loader, 600, device
        )

        # evenly distributed points in range(-10,10)
        unif_samples = torch.rand(10000,3)*20-10 
        trainunif_loader = DataLoader(unif_samples,batch_size = batch_size, shuffle= True)
        unifgan = TogetherLoop(
            MixtureDensityGeneratorStep(mg), MixtureDensityCriticStep(mg),
            goptim, doptim,
            latent,
            gen, dis,
            trainunif_loader, 600, device
        )

        proposal_func   = RandomWalkProposal(std= 0.1,device=device,n_samples = 1))
        target_func     =    
        normalmh = MetropolisHastings(
            propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = ganaccept, # accept or reject new samples

            initial_sample = torch.zeros(batch_size, 3 ,device=device),
            device      = device,

            iterations  = 1000,  
            n_chains    = 1,    
        )
        
        ganpropose = MHGANPropose(gen, latent,batch_size=batch_size)
        ganaccept  = MHGANAcceptReject(dis,n_chains=batch_size,device=device,calibrator=None)
        mh_gan = MetropolisHastings(
            propose_func = ganpropose, # input: last sample, output: new sample [n_sample, *dim of single sample]
            accept_reject = ganaccept, # accept or reject new samples

            initial_sample = torch.zeros(batch_size, 3 ,device=device),
            device      = device,

            iterations  = 500,  
            n_chains    = batch_size,        
        )

        mh_mixin = MH_GAN_and_RW(
            normalMH    = normalmh,
            unifgan     = unifgan,
            ganloop     = traingan,
            ganMH       = ganmh,
            cycles      = 10,
        )

    """
    def __init__(self, normalMH, unifgan, ganloop, ganMH, cycles):
        self.normalMH = normalMH
        self.unifgan  = unifgan
        self.trainGAN = ganloop
        self.ganMH    = ganMH
        self.cycles   = cycles
        self.gan_sample_size = 2000
    def startSampling(self):
        # warmup
        _discard_samples = self.normalMH.startSampling()
        if self.unifgan:
            self.unifgan.train()
        for cycle in range(cycles):
            # maybe do parallel tempering here
            # generate enough samples for gan training
            samples_for_gan = torch.zeros_like(_discard_samples)
            accepted = 0
            while samples_for_gan.shape[0]<self.gan_sample_size:
                samples,_accept,_total, mhscore = self.normalMH.startSampling()
                samples_for_gan = torch.cat((samples_for_gan,samples),dim=0)
            # train gan on new samples
            self.trainGAN.dataloader = torch.utils.data.DataLoader(samples)
            self.trainGAN.train()

            gensamples,_accept,_total, genscore = self.ganMH.startSampling()
            # 1. rank gensamples by discriminator score
            # 2. select 3 subsets of gensamples: high, med, low rank by discriminator
            # 3. evaluate all 3 sets for real 
            # 4. select batch of good ones and use as initial samples of normalMH from next cycle
            # 5. how to use difference in rank to improve discriminator?
            score_idx = np.argsort(sc,0)
            sorted_s = s[score_idx,:]
            sorted_ranked_s = s[:1000,:]
            
            sortedsamples = self.criticSort(gensamples)
            subset = self.selectSubset(sortedsamples)
            score = self.evalSamples(subset)            


    def criticSort(self, samples, ):
        pass
    def selectSubset(self,samples):
        pass
    def evalSamples(self,subset):
        pass
    def selectNewPoints(self):
        pass