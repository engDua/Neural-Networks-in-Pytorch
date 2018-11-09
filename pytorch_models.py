import torch
import os, sys
import numpy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import numpy as np

class Train_model(nn.Module):

  def __init__(self):

        super(Train_model, self).__init__()
        self.fc1 = nn.Linear(60, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,60)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

  def forward(self,x):

        x = F.relu(self.fc1(x))
        #x = self.bn1(x)

        x = F.relu(self.fc2(x))
        #x = self.bn2(x)

        x = F.relu(self.fc3(x))
        #x = self.bn3(x)

        x = F.relu(self.fc4(x))
        #x = self.bn4(x)

        return x




class enc(nn.Module):

    def __init__(self, src_dim, hidden_dim, latent_dim):

        super(enc, self).__init__()
        self.lin_s_h = nn.Linear(src_dim, hidden_dim)
        self.lin_h_t = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.lin_s_h(x))

        return F.relu(self.lin_h_t(x))


class dec(nn.Module):

    def __init__(self, latent_dim, hidden_dim, src_dim):

        super(dec, self).__init__()
        self.lin_s_h = nn.Linear(latent_dim, hidden_dim)
        self.lin_h_t = nn.Linear(hidden_dim, src_dim)

    def forward(self, x):
        x = F.relu(self.lin_s_h(x))

        return F.relu(self.lin_h_t(x))


class Train_vae(nn.Module):


  def __init__(self, enc, dec):

        super(Train_vae, self).__init__()

        latent_dim = 60
        hidden_dim = 512
        src_dim = 60
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.src_dim = src_dim
        self.enc = enc
        self.dec = dec
        self._enc_mu = torch.nn.Linear(hidden_dim,  latent_dim)
        self._enc_log_sigma = torch.nn.Linear(hidden_dim,latent_dim)


 
  def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False), self.z_mean, self.z_sigma  # Reparameterization trick

 
  def loss_calc(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)



  def forward(self, state):
        h_enc = self.enc(state)
        z, mean, sigma = self._sample_latent(h_enc)
        latent_loss = self.loss_calc(mean, sigma)
        return self.dec(z), latent_loss


class my_cnn(nn.Module):
  def __init__(self)
   super(my_cnn,self).__init__()
   self.cnn=nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding)
   self.batchnorm = nn.BatchNorm2d(out_channels)
   self.relu = n.ReLU()
 
 def forward(self,x):
  out = self.cnn(x)
  out = self.bacthnorm(out)
  out = self.relu(out)
  return out
