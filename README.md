# Recognizing-Traffic-Signs-using-CNN

Recognizing Traffic Sings is one of the Learning objectives for a Self-Driving Car. This project comprises the implementation of it using the Keras application. We’ll implement TrafficSignNet, a Convolutional Neural Network which we’ll train on our dataset.

## What is Traffic Sign Classification ?

Traffic sign classification is the process of automatically recognizing traffic signs along the road, including speed limit signs, yield signs, merge signs, etc. Being able to automatically recognize traffic signs enables us to build “smarter cars”.

Self-driving cars need traffic sign recognition in order to properly parse and understand the roadway. Similarly, “driver alert” systems inside cars need to understand the roadway around them to help aid and protect drivers.

Traffic sign recognition is just one of the problems that computer vision and deep learning can solve.

## Two steps :

In the real-world, traffic sign recognition is a two-stage process:

**Localization:** Detect and localize where in an input image/frame a traffic sign is.
**Recognition:** Take the localized ROI and actually recognize and classify the traffic sign.

The dataset we are using is [GTSRB dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/). It has all the traffic signs we required i.e. the first step is completed already. So we'll focus on the next step.

## Project Structure

- **output** : The output model is stored here.
- **examples** : It contains some example of the predicted Images.
- **model** : It contains the our model TrafficSignNet.py

## Dependencies
- OpenCV
- NumPy
- scikit-learn
- scikit-image
- imutils
- matplotlib
- TensorFlow 2.0

