# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# importing functions 
from module_utils import display_stats, show_keypoints
from dataset import *

# Define path to CSV file
path = 'data/training_frames_keypoints.csv'
root = 'data/training'

display_stats(path)
show_keypoints(path,root , 3)

face_dataset = KeypointDataset(path, root)

print('length of dataset : ', len(face_dataset))

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

transformed_dataset = KeypointDataset(path, root, transform=data_transform)

print('length of transformed dataset : ', len(transformed_dataset))

# Size of image & keypoints : torch.Size([1, 224, 224]) torch.Size([68, 2])

## Data has been prepared