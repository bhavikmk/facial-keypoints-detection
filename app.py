import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from models import Net

# load in color image for face detection

st.title("Facial Keypoint Detection System")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(image, channels="BGR")

# image = cv2.imread('images/obamas.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# st.image(image, channels="BGR")

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image, 1.2, 2)

image_with_detections = image.copy()

for (x,y,w,h) in faces:
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

st.header('Image with Detections')

st.image(image_with_detections)

net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_10.pt'))
net.eval()

def show_all_keypoints(image, predicted_key_pts):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=5, marker='.', c='m')
    ax.axis('off')
    st.pyplot(fig)

st.header('Image with Keypoints')

image_copy = np.copy(image)

# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y-50:y+h+50, x-40:x+w+40] # the numeric values are needed for scaling the output keypoints correctly.
    width_roi = roi.shape[1] # needed later for scaling keypoints
    height_roi = roi.shape[0] # needed later for scaling keypoints
    roi_copy = np.copy(roi) # will be used as background to display final keypints.
    
    ## TODO: Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi/255.0
    
    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (96, 96))  # resize the image to get a square 96*96 image.
    roi = np.reshape(roi,(96, 96, 1)) # reshape after rescaling to add the third color dimension.
    
    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi = roi.transpose(2, 0, 1)
    
    roi = torch.from_numpy(roi).type(torch.FloatTensor) # convert images to FloatTensors (common source of error)
    
    roi = roi.unsqueeze(0)
    
    keypoints = net(roi)
    keypoints = keypoints.view(68, 2)    
    keypoints = keypoints.data.numpy()
    keypoints = keypoints*50.0 + 100
    keypoints = keypoints * (width_roi / 96, height_roi / 90) # scale the keypoints to match the size of the output display.
    plt.figure() 
     
    # Using helper function for display as defined previously.  
    show_all_keypoints(roi_copy, keypoints)