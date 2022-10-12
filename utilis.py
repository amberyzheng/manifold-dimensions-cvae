import math
import os
import sys
import time
import datetime

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchvision import datasets as datasets
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Select device
def select_device():
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    return cuda if torch.cuda.is_available() else cpu


def load_data(bs=512, device=torch.device("cuda"), ds="fmnist"):
    # Load MNIST data
    if ds == "mnist":
        mnist = datasets.MNIST(
            root="/home/ubuntu/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        testset = datasets.MNIST(
            root="/home/ubuntu/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    else:
        mnist = datasets.FashionMNIST(
            root="/home/ubuntu/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        testset = datasets.FashionMNIST(
            root="/home/ubuntu/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    dat = 0.001 + 0.998 * mnist.data / 255.0
    test_dat = (0.001 + 0.998 * testset.data / 255.0).to(device)
    dat = dat.to(device=device)
    loader = DataLoader(mnist, batch_size=bs, shuffle=True)

    return dat, test_dat, loader, testset.train_labels.to(device)


# Check the weights of generator
def check_weights(save_path, latent_dim=32):
    generator = torch.load(f"{save_path}/generator.pth")
    zw = torch.mean(torch.square(list(generator.upsample.parameters())[0]), dim=0)[: latent_dim]
    cw = torch.mean(torch.square(list(generator.upsample.parameters())[0]), dim=0)[latent_dim:]

    return {"z weights": zw, "c weights": cw}


# Save and load models
def save_models(path, models):
    os.makedirs(path, exist_ok=True)
    def namestr(obj, namespace=globals()):
        return [name for name in namespace if namespace[name] is obj]
    for model in models:
        torch.save(model, f"{path}/{namestr(model)[0]}.pth")

def load_models(path):
    encoder = torch.load(f"{path}/encoder.pth")
    generator = torch.load(f"{path}/generator.pth")
    loggamma = torch.load(f"{path}/loggamma.pth")
    if os.path.exists(f"{path}/prior.pth"):
        prior = torch.load(f"{path}/prior.pth")
        return encoder, generator, loggamma, prior
    return encoder, generator, loggamma