# Deep Learning and Neural Networks Challenges
This repository contains the solutions to three challenges of the **Deep Learning and Neural Networks** exam at **Politecnico di Milano**. The challenges involve image classification, image segmentation, and visual question answering, each focusing on different aspects of deep learning and computer vision.

## Challenge 1: Image Classifier (Object Classification)
In this challenge, the goal was to build an image classifier to identify various objects from a given dataset. The dataset includes a wide range of objects, from planes to wine bottles. The task was to train a deep learning model capable of classifying images into predefined categories.

### Key Steps:
1. **Dataset Preparation**: The dataset was preprocessed by resizing images, normalizing pixel values, and augmenting the data to improve model generalization.
2. **Model Architecture**: A convolutional neural network (CNN) was used for feature extraction, followed by fully connected layers for classification.
3. **Training and Evaluation**: The model was trained using cross-entropy loss and evaluated using accuracy. Performance was assessed on a validation set.

### Tools and Libraries:
- TensorFlow / Keras
- OpenCV
- NumPy

---

## Challenge 2: Image Segmentation
The second challenge focused on image segmentation, where the task was to classify each pixel of an image into one of several categories. This type of challenge is common in medical imaging, autonomous driving, and robotics.

### Key Steps:
1. **Dataset Preparation**: Images were annotated with pixel-level masks for training.
2. **Model Architecture**: A U-Net architecture was employed, which is particularly well-suited for image segmentation tasks. The model includes an encoder-decoder structure with skip connections.
3. **Training and Evaluation**: The model was trained using the Dice coefficient and Intersection over Union (IoU) as evaluation metrics to measure the overlap between predicted and ground truth masks.

### Tools and Libraries:
- TensorFlow
-  Keras
- NumPy
- Matplotlib

### Key Steps:
1. **Dataset Preparation**: The VQA dataset contains images paired with questions and answers. Each question is related to a specific image, and the answers are categorized.
2. **Model Architecture**: The model consisted of a two-stream approach where the image was processed using a CNN (such as ResNet) to extract features, and the question was encoded using a Recurrent Neural Network (RNN) or Transformer model. The two streams were fused and used to generate an answer.
3. **Training and Evaluation**: The model was trained using cross-entropy loss and evaluated based on the accuracy of the generated answers.

### Tools and Libraries:
- TensorFlow / Keras
- NumPy

---

## Requirements
To run the challenges, you will need the following libraries:
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
