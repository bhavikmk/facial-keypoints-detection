from logging import root
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os

def show_keypoints(csv_path,root, n):
    
    """Show image with keypoints"""

    key_pts_frame = pd.read_csv(csv_path)
    image_name = key_pts_frame.iloc[n, 0]
    key_pts = key_pts_frame.iloc[n, 1:].as_matrix()
    key_pts = key_pts.astype('float').reshape(-1, 2)
    plt.figure(figsize=(5, 5))
    plt.imshow(mpimg.imread(os.path.join(root, image_name)))
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    plt.show()


def display_stats(path):

    key_pts_frame = pd.read_csv(path)
    n = 0
    image_name = key_pts_frame.iloc[n, 0]
    key_pts = key_pts_frame.iloc[n, 1:].as_matrix()
    key_pts = key_pts.astype('float').reshape(-1, 2)

    print('Image name: ', image_name)
    print('Landmarks shape: ', key_pts.shape)
    print('First 4 key pts: {}'.format(key_pts[:4]))
    print('Number of images: ', key_pts_frame.shape[0])