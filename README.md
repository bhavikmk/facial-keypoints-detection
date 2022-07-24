# Keypoint detection for facial applications

## Overview

In this project, I have combined knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. 

### Applications 
* facial tracking
* facial pose recognition
* facial filters 
* emotion recognition

## Module : keypoint_detector

  - **dataset.py**     
    - for preparing dataset
  - **models.py** 
    - for defining architecture of CNN model
  - **model_train.py** 
    - for training and saving model  
  - **model_test.py** 
    - for testing trained models
  - **model_utils.py** 
    - utility functions for above scripts

## Dependencies

- torch
- torchvision
- opencv-python (3.4)
- matplotlib
- pandas
- numpy
- scipy