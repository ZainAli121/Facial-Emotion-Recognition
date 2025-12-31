# Facial Emotion Recognition Using Deep Learning

This project is about recognizing human emotions from facial images using deep learning.

The system looks at a face image and predicts the emotion like happy, sad, angry, fear, surprise, or neutral.

## Dataset Used

I used the FER2013Plus dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/subhaditya/fer2013plus

This dataset contains many face images with different emotions.

## Models Used

I trained and tested two deep learning models.

### 1. CNN (Convolutional Neural Network)

This is a simple deep learning model used for image classification.

Accuracy results:
- Train Accuracy: 62.74%
- Test Accuracy: 69%

### 2. VGG Model

This is a deeper and more powerful model compared to CNN.

Accuracy results:
- Train Accuracy: 80%
- Test Accuracy: 79%

## Tools and Libraries Used

- Python
- TensorFlow and Keras
- PyTorch
- NumPy
- Matplotlib

## How the Project Works

1. Load the facial emotion dataset
2. Preprocess the images
3. Train the CNN model
4. Train the VGG model
5. Test both models on new images
6. Predict the emotion from face images

## Results and Observations

- Both models can detect emotions from facial images
- The VGG model gives better accuracy than the CNN model
- Deep learning works well for facial emotion recognition

## Files in This Repository

- emotion-recognition-cnn.ipynb  
  CNN model training and testing

- vgg-for-emotion-recognition.ipynb  
  VGG model training and testing

- README.md  
  Project explanation

## Purpose of This Project

This project is made for learning and practice.
It helps to understand:
- Deep learning
- Image classification
- Facial emotion recognition

## Conclusion

This project shows how deep learning can be used to recognize human emotions from facial images.
Using a deeper model like VGG improves the accuracy.

Thank you for reading.
