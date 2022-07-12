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
from module_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

net = Net()
net.to(device)

train_csv_path = 'data/training_frames_keypoints.csv'
train_root = 'data/training'
test_csv_path = 'data/test_frames_keypoints.csv'
test_root = 'data/test'

## Transforming data

data_transform = transforms.Compose( [Rescale(100) , RandomCrop(96) , Normalize() , ToTensor()] )
batch_size = 20

transformed_dataset = KeypointDataset(train_csv_path,train_root, transform=data_transform)
test_dataset = KeypointDataset(test_csv_path, test_root, transform=data_transform)

# Loading transformed dataset

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

def train_net(n_epochs):

    net.train()

    for epoch in range(n_epochs): 
        running_loss = 0.0

        for batch_i, data in enumerate(train_loader):
            images = data['image']
            key_pts = data['keypoints']

            key_pts = key_pts.view(key_pts.size(0), -1)

            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            key_pts, images = key_pts.to(device), images.to(device)

            output_pts = net(images)
            loss = criterion(output_pts, key_pts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_i % 10 == 9:   
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

n_epochs = 5 
train_net(n_epochs)

test_images, test_outputs, gt_pts = net_sample_output()

torch.save(net.state_dict(), 'trained_models/model_epoch'+ str(n_epochs)+ '.pt')