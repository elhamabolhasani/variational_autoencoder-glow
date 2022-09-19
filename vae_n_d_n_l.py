import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import os
import pickle
import zipfile
import datetime
from shutil import copy
import torch
from torch.distributions import Normal, Independent
import numpy as np


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        # self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")**
        self.m4 = DecoderModule(32, color_channels, stride=1)

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.m4(out)  # **


class VAE(nn.Module):
    def __init__(self, dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert dataset in ["mnist", "fashion-mnist", "cifar", "stl"]

        super().__init__()
        # # latent features
        self.n_latent_features = 256

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
            self.decoder_output_size = 28
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
            self.decoder_output_size = 32
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            self.color_channels = 1
        else:
            self.color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size
        self.n_neurons_last_decoder_layer = self.color_channels * self.decoder_output_size * self.decoder_output_size
        print(self.n_neurons_last_decoder_layer)
        # Encoder
        self.encoder = Encoder(self.color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc4 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(self.color_channels, pooling_kernel, encoder_output_size)
        self.fc5 = nn.Linear(self.n_neurons_last_decoder_layer,
                             self.decoder_output_size * self.decoder_output_size * self.color_channels)  # check if its true ?
        self.fc6 = nn.Linear(self.n_neurons_last_decoder_layer,
                             self.decoder_output_size * self.decoder_output_size * self.color_channels)

        # data
        # self.train_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss": [], "val_loss": []}

        # model name
        self.model_name = dataset + 'n_decoder_n_latent_dim=256'
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

        self.log_norm_constant = -0.5 * np.log(2 * np.pi)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def decoder_bottleneck(self, d):
        mu, logvar = self.fc5(d), self.fc6(d)
        return mu, logvar

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc4(z)
        d = self.decoder(z)
        d_ = d.view(-1, self.n_neurons_last_decoder_layer)
        mu_d, logvar_d = self.decoder_bottleneck(d_)
        return mu_d, logvar_d, mu, logvar

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)