
# Image Classification with SVM using HOG Features

## Objective
The objective of this project is to perform binary classification using Support Vector Machines (SVM) with an image dataset. The dataset consists of images of ice cream and pizza, and the goal is to train an SVM model to accurately classify new images into these two categories.

## Tools and Language
- Google Colab
- Python

## Description
Image classification is a fundamental problem in the field of artificial intelligence and machine learning. It involves categorizing images into different predefined classes or categories. In this project, we utilize the SVM algorithm and Histogram of Oriented Gradients (HOG) features to train a model for image classification.

### Support Vector Machines (SVM)
Support Vector Machines are a set of supervised learning methods used for classification, regression, and outlier detection. SVMs choose the decision boundary that maximizes the distance from the nearest data points of all the classes, resulting in a maximum margin classifier or hyperplane.

### Histogram of Oriented Gradients (HOG)
HOG is a feature descriptor that captures the local shape and structure of an image by computing histograms of gradient orientations within small cells. By extracting HOG features, we can represent images in a way that captures important information about edges and texture.

### Implementation Steps
1. Importing necessary libraries and setting up the environment.
2. Extracting the dataset from a zipped file.
3. Reading image paths and labels from the dataset.
4. Computing HOG features for each image.
5. Preparing the data by converting features and labels to NumPy arrays.
6. Splitting the data into training and testing sets.
7. Training the SVM classifier using the training data.
8. Making predictions on the test set and evaluating the model's performance.
9. Testing the model on individual images.

## Experiment Analysis
In this project, we performed Principal Component Analysis (PCA) to reduce the dimensionality of the HOG features. We experimented with different values of k (number of principal components) and evaluated the model's accuracy.

Value of K   | Accuracy
------------ | --------
100          | 0.76
500          | 0.75
1000         | 0.77
3000         | 0.77

Based on the results, we chose k=3000 for PCA.

## Conclusion
The implemented SVM model utilizing HOG features and PCA demonstrates its effectiveness in classifying images of ice cream and pizza. The model shows promising accuracy, but there is room for improvement, particularly in the pizza class. Further optimizations and fine-tuning can be performed to enhance the model's performance.
