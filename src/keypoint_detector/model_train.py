import matplotlib.pyplot as plt
import numpy as np

# PyTorch functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

# Importing model, dataset
from models import Net
from dataset import *

# I tried to use my laptop's gpu. It wasn't working properly.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

net = Net()
net.to(device)
print(net)

train_csv_path = 'data/training_frames_keypoints.csv'
train_root = 'data/training'
test_csv_path = 'data/test_frames_keypoints.csv'
test_root = 'data/test'

## Loading and training data

data_transform = transforms.Compose( [Rescale(100) , RandomCrop(96) , Normalize() , ToTensor()] )

batch_size = 20

transformed_dataset = KeypointDataset(train_csv_path,train_root, transform=data_transform)
test_dataset = KeypointDataset(test_csv_path, test_root, transform=data_transform)

print('Number of images: ', len(transformed_dataset))

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

