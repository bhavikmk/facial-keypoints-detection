# Facial Keypoints Detection

https://github.com/bhavikmk/keypoints-detection/assets/72643864/c389cefd-8dd9-4ef5-850a-6398d56e5bf4



## Project Description

Facial Keypoints Detection is a computer vision project that aims to detect and locate key facial features such as eyes, nose, and mouth in an image. It involves building a model that can predict the coordinates of these keypoints on a face.

The project utilizes a simple architecture designed for ease of deployment on CPU. It leverages a deep learning model trained on a large dataset of facial images annotated with keypoints. The model is capable of analyzing new images and predicting the positions of facial keypoints.

To make the project more accessible and user-friendly, an interactive web application has been developed. This allows users to upload their own images, visualize the detected keypoints, and explore the predictions made by the model. The web application provides a seamless and intuitive experience for facial keypoints detection.

## Pipeline

The pipeline for the Facial Keypoints Detection project involves the following steps:

1. **Data Collection**: This is a diverse dataset of facial images with annotated keypoints. This dataset serves as the training data for the model. Data were taken from [here](https://github.com/udacity/P1_Facial_Keypoints/tree/master/data)

2. **Data Preprocessing**: The collected dataset is preprocessed to normalize the images, perform data augmentation, and prepare the annotations for training.

3. **Model Training**: The preprocessed dataset is used to train the facial keypoints detection model. The model is trained using a simple deep learning architecture, optimized for keypoint regression on CPU device deployment.

4. **Model Evaluation**: The trained model is evaluated on a test set to assess its performance and fine-tune its hyperparameters if necessary.

5. **Web Application Development**: An interactive web application is developed using Streamlit. The application allows users to upload images, visualize the detected keypoints, and explore the model's predictions.

6. **Deployment** (future): The web application, along with the trained model, will be deployed on a server or cloud platform, making it accessible to everyone.

## Application

The Facial Keypoints Detection project finds applications in various domains, including:

- **Facial Analysis**: The accurate detection of facial keypoints enables advanced facial analysis tasks such as emotion recognition, age estimation, and gender classification.

- **Virtual Try-On**: Facial keypoints can be used to precisely align virtual objects like glasses, hats, or makeup on a person's face in virtual try-on applications.

- **Augmented Reality**: By accurately detecting and tracking facial keypoints, it becomes possible to overlay virtual objects on a person's face in real-time augmented reality experiences.

- **Biometrics**: Facial keypoints can serve as landmarks for facial recognition systems, improving accuracy and robustness in biometric authentication scenarios.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- Streamlit
- Numpy
- Matplotlib

## Getting Started

To get started with the Facial Keypoints Detection project, follow these steps:

1. Clone the repository: `git clone https://github.com/bhavikmk/keypoints-detection.git`

2. Install the dependencies

3. Download the pre-trained model weights: [keypoints_model.pt](https://drive.google.com/file/d/1YwcuOcXrWwDjbttLwho6o2O8nphoBdgI/view?usp=sharing)

4. Run the web application: `streamlit run app.py`

5. Open your browser and navigate to `http://localhost:8501` to access
