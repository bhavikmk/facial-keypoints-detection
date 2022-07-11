import matplotlib.pyplot as plt
import numpy as np

# PyTorch functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Importing model
from models import Net

# I tried to use my laptop's gpu. It wasn't working properly.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

net = Net()
net.to(device)
print(net)

