# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# small network
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self,input_size):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, int(input_size/4)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(input_size/4), 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
class generator(nn.Module):
    def __init__(self,output_size):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(4, int(output_size/4)),
            nn.ReLU(),
            nn.Linear(int(output_size/4), output_size),
        )

    def forward(self, input):
        return self.main(input)
    


# %%
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
#from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt


# %%
# Discriminator Loss => BCELoss
# abides with comp9417 loss, generator emphasis on samples that fails to fool discriminator
def d_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)

def g_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)


# %%
from time import time
from LDJoint import MCMC
traindata = np.loadtxt("./data/Sunspot/train.txt")
testdata = np.loadtxt("./data/Sunspot/test.txt")  #
name	= "Sunspot"
trainx = torch.from_numpy(traindata[:,:-1]).type(torch.FloatTensor)
trainy = torch.from_numpy(traindata[:,-1]).type(torch.FloatTensor).reshape((len(traindata),1))
testx  = torch.from_numpy(testdata[:,:-1]).type(torch.FloatTensor)
testy  = torch.from_numpy(testdata[:,-1]).type(torch.FloatTensor).reshape((len(testdata),1))


num_samples = 3000
lr = 0.01
# mcmc
st = time()
mcmc = MCMC(trainx,trainy,testx,testy,True,1,lr,num_samples,networktype='fc',hidden_size=[5])#[4,3])
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
[theta_all,theta_accepted,fx_train,fx_test,accept_ratio] = mcmc.sample(a,b,sigma,tau_prop_std,w_prop_std)
print(accept_ratio,len(theta_accepted))
print("Time Taken:",time()-st)
theta_accept_burn = theta_accepted[int(theta_accepted.shape[0]/2):,:]
print(theta_accept_burn.shape)


# %%
from torch.utils.data import DataLoader
traindata = DataLoader(theta_accept_burn,batch_size=16)


# %%
# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Model
G = generator(output_size = theta_accept_burn.shape[1]).to(device)
D = discriminator(input_size=theta_accept_burn.shape[1]).to(device)
print(G)
print(D)

# Settings
epochs = 200
lr = 0.0002
batch_size = 16
#g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
#d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

g_optimizer = optim.SGD(G.parameters(), lr=lr)
d_optimizer = optim.SGD(D.parameters(), lr=lr)

wholepassd = theta_accept_burn.to(device)
realout = D(wholepassd)
wholeloss = -torch.mean(torch.log(realout))
d_optimizer.zero_grad()
wholeloss.backward()
d_optimizer.step()

# %%
# Train
for epoch in range(epochs):
    epoch += 1

    for times, data in enumerate(traindata):
        #print(times,data[0].shape,data[1].shape,print())
        times += 1

        real_inputs = data.to(device)
        #test = 255 * (0.5 * real_inputs[0] + 0.5)
        print(real_inputs.shape)
        real_inputs = real_inputs.view(-1, 32)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        latent = torch.randn(real_inputs.shape[0],4)
        #print(noise.shape)
        latent = latent.to(device)
        fake_inputs = G(latent)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        # Zero the parameter gradients
        d_optimizer.zero_grad()

        # Backward propagation
        d_loss = d_loss_function(outputs, targets)
        print(d_loss)
        d_loss.backward()
        d_optimizer.step()

        # Generator
        latent = torch.randn(real_inputs.shape[0],4)
        latent = latent.to(device)

        fake_inputs = G(latent)
        fake_outputs = D(fake_inputs)

        g_loss = g_loss_function(fake_outputs)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if times % 100 == 0 or times == len(train_loader):
            print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader), d_loss.item(), g_loss.item()))
        #break
    #imgs_numpy = (fake_inputs.data.cpu().numpy()+1.0)/2.0
    #show_images(imgs_numpy[:16])
    #plt.show()

    if epoch % 50 == 0:
        torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')


print('Training Finished.')
#print('Cost Time: {}s'.format(time.time()-start_time))
