# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Importing torch modules & fuctions
from torch.utils.data import Dataset, DataLoader

class KeypointDataset(Dataset):
    
    def __init__(self, csv, root, transform=None):

        self.key_pts_frame = pd.read_csv(csv)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, index):
        
        image_name = os.path.join(self.root, self.key_pts_frame.iloc[index, 0])
        image = mpimg.imread(image_name)
        