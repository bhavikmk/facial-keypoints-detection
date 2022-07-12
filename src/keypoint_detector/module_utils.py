import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import torch

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

def net_sample_output():
    for i, sample in enumerate(test_loader):
        images = sample['image']
        key_pts = sample['keypoints']
        images = images.type(torch.FloatTensor)        
        images = images.to(device)
        key_pts = key_pts.to(device)
        output_pts = net(images)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        if i == 0:
            return images, output_pts, key_pts

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=5):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot( batch_size, 1, i+1)
        
        image = test_images[i].data   # get the image from it's Variable wrapper
        
        if torch.cuda.is_available():
            image = image.cpu()
            
        image = image.data
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        predicted_key_pts = test_outputs[i].data
        
        if torch.cuda.is_available():
            predicted_key_pts = predicted_key_pts.cpu()
            
        predicted_key_pts = predicted_key_pts.data
        predicted_key_pts = predicted_key_pts.numpy()
        predicted_key_pts = predicted_key_pts*50.0+100
        
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i] 
            
            if torch.cuda.is_available():
                ground_truth_pts = ground_truth_pts.cpu()  
                
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()