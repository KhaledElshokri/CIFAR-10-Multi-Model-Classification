# CIFAR-10-Multi-Model-Classification

This is a project that implements multiple machine learning models, including Naïve Bayes, Decision Trees, Multi-Layer Perceptrons (MLPs), and Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset.

## Table of Contents
1. [Installation](#installation)
2. [Running the Code](#running-the-code)
3. [Model Details](#model-details)
4. [Results](#results)

## Installation

### Prerequisites
- Python 3.8 or higher https://www.python.org/downloads/
- `pip` for package management

## Running The Code
### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/KhaledElshokri/CIFAR-10-Multi-Model-Classification.git
   ```
   ```bash
   cd CIFAR-10-Multi-Model-Classification
2. Install Pytorch  
   * you can follow this link https://pytorch.org/get-started/locally/ to install the correct config depending on your device. The example shown is for windows directly on cpu.
   ```bash
   python -m pip install torch torchvision torchaudio 
3. Create an sklearn environment and activate it
   ```bash
   python -m venv sklearn-env
   ```
   ```bash
   sklearn-env\Scripts\activate
   ```
   * make sure that your command line now starts with (sklearn-env)

4. In the sklearn environment run the data_preparation.py to create the training labels and test labels and feature extraction using PCA
   ```bash
   python data_preparation.py
   ```
   * After running this script you should have 4 new .npy files in the root directory
5. Now you can train and test the Decision Tree models and the Naive Bayes models (always in the sklearn environment)
   ```bash
   python decision_tree_custom.py
   ```
   ```bash
   python decision_tree_sklearn.py
   ```
   ```bash
   python naive_bayes_custom.py
   ```
   ```bash
   python naive_bayes_sklearn.py
   ```
   * After running each script you should see a new csv file generated in the route directory that contains the confusion matrix generated by testing the model.
6. To run the MLP model you have to first run the data_preperation_MLP.py file to generate the training features without PCA.
   ```bash
   python data_preparation_MLP.py
   ```
   * You should now see the train_features_full.npy files in the route directory.
7. Now you can train and test the MLP model by executing the following file.
   ```bash
   python MLP_pytorch.py
   ```
   * You should now see a new csv file called confusion_matrix_mlp.csv that shows the confusion matrix generated by the results of the testing.

8. To train and test the VG11 CNN model you only need to run this script without any data preparation
   ```bash
   python train_vgg11.py
   ```
   * This step might take alot of time (~ 2 hours) if you are running it directly on cpu locally, try using google collab platform to have access to better resources. use this link to learn more on how to use it https://colab.research.google.com/drive/16pBJQePbqkz3QFV54L4NIkOn1kwpuRrj
   * If you want to use google collab go to this link https://colab.research.google.com/ create a new notebook and copy paste the vg11_collab.py file in the code section. Make sure to change the runtime type from the task bar to GPU instead of CPU.
   * Using google collab, it takes 10 minutes to download, train and test the model.

## Model Details
### Naive Bayes
* naive_bayes_custom.py is my custom implementation of an algorithm that classifies CIFAR-10 images using a probabilistic approach.
* naive_bayes_sklearn.py is an implementation that relies on sklearn Gaussian Naive Bayes library to make probablistic classifications.
### Decision Tree
* decision_tree_custom.py is my custom implementation of an N-dimensional tree that classifies CIFAR-10 images using a deterministic approach.
* decision_tree_sklearn.py is an implementation that uses the DecisionTreeClassifier object from sklearn to classify the images using the gini index.
### Multi-Layer Perception (MLP)
* MLP_pytorch.py is an implementation of a multi-layer perceptron that uses 2 hidden layers of 512 neurons each with ReLU activation and batch normalisation in the second layer with mini-batch gradient descent.
### Convolutional Neural Network (CNN)
* vg11_model.py is the implementation of the VGG11 convolutional neural network, a variant of the VGG architecture where the features block consists of a series of convolutional layers with ReLU activations, batch normalization, and max-pooling to extract hierarchical features from the input image. The classifier block is a fully connected network with dropout layers, which processes the extracted features and outputs class scores for 10 classes in the CIFAR-10 dataset.
* train_vg11.py is the implementation to load the CIFAR-10 dataset and test the CNN specified in the vg11_model.py.

## Results

In root directory of the repository there is a pdf called ProjectReport.pdf that includes thorow explantions for all the models and some example results acheived using these models.
