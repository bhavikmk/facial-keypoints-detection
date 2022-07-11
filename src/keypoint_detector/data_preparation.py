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

# importing functions 
from visualizer import display_stats, show_keypoints

# Define path to CSV file
path = 'data/training_frames_keypoints.csv'

display_stats(path)
show_keypoints(path, 3)



