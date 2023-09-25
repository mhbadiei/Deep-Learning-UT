# Deep-Learning-UT

## Exercise 1: McCulloch-Pitts Neurons and Neural Network Design

In this exercise, you will delve into the fundamentals of neural networks by focusing on McCulloch-Pitts neurons and designing a neural network architecture to address a specific problem.

### Question 1: McCulloch-Pitts Neurons

This question introduces you to McCulloch-Pitts neurons, which are simple binary threshold units. The task here is to design a neural network architecture with two input neurons that can separate points inside a convex quadrilateral from those outside it. The four corner points of the convex quadrilateral are given as (3, 3), (1, 3), (5, -2), and (-1, -2), and the objective is to classify points as either inside (1) or outside (0) this region.

### Question 2: Adaline

This part of the exercise shifts focus to Adaline, which is a type of neural network used for binary classification. You are provided with two sets of data points, each with specific distributions for their x and y coordinates. Your task is to train an Adaline network to separate these two datasets. Additionally, you should plot the error changes (mean squared error) over iterations and analyze whether Adaline is a suitable approach for this classification task. If it's not suitable, you are expected to propose an alternative solution.

### Question 3: Perceptron

Question 3 explores the Perceptron, another type of neural network used for binary classification. You will need to provide a concise explanation of how Perceptrons work and outline the weight update process in brief. Then, you are given a Perceptron with specific weight values and a bias. Given an input vector and learning rate, you need to update the weights for two iterations and calculate the expected output.

### Question 4: Madaline

The final part of the exercise introduces Madaline, a single-layer neural network used for classification tasks. You are presented with three categories of data points with specific distributions. Your task is to train a single-layer neural network (Madaline) to classify these data points correctly. You should explain the architecture of the network, the choice of the number and type of neurons, and provide visualizations of the separating lines. Furthermore, you are expected to experiment with different learning rates and analyze their effects on training and classification performance.

## Exercise 2: Multilayer Perceptron for Classification and Regression

### Question 1: Multilayer Perceptron for Classification

The goal of this exercise is to create a multilayer perceptron (MLP) neural network for classifying the Ionosphere dataset into two categories. The dataset contains radar information that is ultimately classified into two classes. You should answer the following questions:

* Part A: Explain how you divide the data into training, testing, and evaluation sets. Describe different methods and provide a clear rationale for choosing the most suitable one. Consider appropriate data preprocessing steps. Design an MLP model with at least two layers, and specify the architecture of the created network.

* Part B: Utilize the stochastic mini-batch-based method and choose batch size arbitrarily. Then, modify the number of neurons in each hidden layer and analyze the impact of varying the number of neurons on both accuracy and training time. Perform this experiment three times with different hidden layer neuron configurations and report the results.

* Part C: Using the best model from the previous part, employ the stochastic mini-batch-based method with batch sizes of 40 and 254. Examine the effect of batch size differences (32, 40, 254) on accuracy and training time.

* Part D: Alter the activation functions of each layer in the network and analyze the impact of these changes on training accuracy. Modify activation functions in the hidden layers before the output layer a total of three times and report the results. Evaluate the advantages and disadvantages of these activation functions compared to others.

* Part E: Change the loss function of the network and analyze how different loss functions affect training accuracy. Modify the loss function a total of two times and provide mathematical reasoning for these changes.

* Part F: Adjust the network's optimizer, and examine the influence of different optimizers on training accuracy. Modify the optimizer twice and report the results.

* Part G: Add additional hidden layers to the network and investigate how it affects the output. Perform a minimum of three experiments by adding various numbers of hidden layers. Report not only loss and accuracy but also additional evaluation metrics such as Precision, Recall, and F-Score.

* Part H: Based on the evaluations conducted, determine which parameters yield the best results. Provide a comprehensive explanation for your choice based on the previous parts.

* Part I: Discuss the challenges and solutions when dealing with imbalanced datasets, where the number of samples in different classes is not equal.

### Question 2: Multilayer Perceptron for Regression

In this question, you will explore the application of multilayer perceptron neural networks in regression. You will use the Data-Reg dataset, which includes music feature data and geographical coordinates. Perform the following tasks:

* Part 1: Implement a linear regression model using the sklearn.linear_model library and compare its performance to a neural network with no hidden layers, using both Mean Squared Error (MSE) and Mean Absolute Error (MAE) as loss functions. Report results for 11 and 51 epochs.

* Part 2: Use the best optimization and loss functions from the previous part and replace the linear activation function with a non-linear one. Analyze the differences in results and provide a rationale for your choice of activation function. Report results for 11 and 51 epochs.

* Part 3: Investigate the impact of batch size on the results and provide evidence to support your conclusions. Experiment with different batch sizes and discuss the outcomes.

* Part 4: Introduce dropout layers to the neural network in the cases with 2 layers, 1 layer, and no hidden layers. Analyze the effects of dropout layers on the results and provide insights.

* Part 5: Add batch normalization layers to the previously mentioned network architectures. Investigate how batch size affects the results and provide an analysis.

* Part 6: Design a neural network with more than 2 hidden layers. Optimize the network using your knowledge from previous parts, including optimization methods, loss functions, and batch size. Analyze the changes in results and provide insights.

### Question 3: Dimensionality Reduction

In this question, you'll explore dimensionality reduction techniques. Perform the following steps and compare the results to those from Question 1:

* Part 1: Use an autoencoder neural network to reduce the data's dimensions. Determine an acceptable reduction level, train the winning model using the reduced data, and report results in terms of time, accuracy, and error.

* Part 2: Explain mathematically how Principal Component Analysis (PCA) can be used to reduce data dimensions. Implement PCA without using a library and apply dimensionality reduction.


## Exercise 3: Character Recognition, Auto-associative Networks, and Hopfield Network

### Question 1: Character Recognition using Hebbian Learning Rule

* a. Design a network to recognize characters with * dimensions using input patterns consisting of 1s (black squares) and -1s (white squares). Can this network recognize all input patterns correctly?

* b. Determine the minimum dimension that the network can handle when you introduce 11% and 01% random noise to the input patterns instead of 1s and -1s. Apply this to both dimensions of the output from part (a). What is the network's output, and what percentage of cases does it correctly recognize the output?

* c. Replace 11% and 01% of the information with zeros instead of 1s and -1s in the input patterns. What is the network's output for both dimensions of the output in part (a)? What percentage of the time does it correctly recognize the output?

* d. Analyze whether the network is more resilient to noise or loss of information. Discuss the effect of output dimensions on the network's resilience.

### Question 2: Auto-associative Network

In this question, you will store two matrices using the Hebbian learning rule and the modified Hebbian learning rule. Then, you will introduce errors into the input and analyze the results.

### Question 3: Discrete Hopfield Network

Use a Discrete Hopfield Network to recover an image on the left by storing it and then providing the image on the right to restore it. Observe the convergence of the right-side image to the left-side image.

### Question 4: Bidirectional Associative Memory (BAM)

* a. Determine the weight matrix for the first three patterns (C, E, R).

* b. Assess the network's ability to retrieve information from both directions and report the results.

* c. Introduce 01% noise to the inputs and run the network 100 times. Report the percentage of correct network outputs.

* d. Determine the maximum number of patterns that can be stored in this network.

* e. Consider all patterns and implement the network. Identify which patterns are more likely to have errors and provide reasons.


## Exercise 4: MaxNet and HammingNet

### Question 1: Mexican Hat Network & Similarity-Based Classification Network

* a. Explain the operation of the Mexican Hat network briefly. Then, using this network, find the maximum value of the given array for 𝑅1 = 1 and 𝑅2 = 0. In each iteration, plot the indices of array elements and the output signal. Use the activation function as defined below:

𝑓(𝑥) = {
1 if 𝑥 < 2
𝑥 if 1 ≤ 𝑥 < 2
2 if 𝑥 ≥ 2

* b. Under what conditions does this network behave similarly to a MaxNet? Consider suitable values for 𝑅, 𝑅1, 𝑅2, 𝐶1, 𝐶2, and 𝑡_𝑚𝑎𝑥, and repeat part (a) with these values.


Design a network that categorizes input vectors based on their similarity to the following base vectors:

𝑒1 = [1, -1, 1, -1, -1, -1]
𝑒2 = [-1, 1, -1, 1, 1, -1]
𝑒3 = [-1, -1, 1, 1, -1, 1]
𝑒0 = [1, 1, -1, 1, 1, -1]

* a. Explain the architecture of the network in detail.

* b. Apply the following input vectors to the network, categorize them based on the base vectors, and report the results:

𝑣1 = [-1, -1, 1, -1, 1, -1]
𝑣2 = [1, 1, 1, 1, -1, -1]
𝑣3 = [-1, -1, -1, 1, 1, -1]
𝑣0 = [1, -1, 1, 1, -1, 1]
𝑣5 = [1, 1, 1, -1, -1, -1]
𝑣6 = [1, -1, -1, 1, 1, 1]
𝑣7 = [-1, 1, -1, -1, -1, 1]

### Question 3: Self-Organizing Map (SOM)

In this question, we want to classify data from the Fashion-MNIST dataset using a SOM network.

* a. Create a SOM network with 225 neurons, using the first 1111 data points from the dataset for training. Then, report the requested results on the remaining 311 data points.

* b. Consider neurons arranged in a square grid with a radius 𝑅 = 1.

i. Plot a graph showing the number of data points in each cluster. The x-axis represents cluster numbers, and the y-axis represents the number of test data points mapped to each cluster.

* c. Visualize the process of changing the cluster assignment. Do this by transforming the network weights into images of 28x28 pixels for different epochs. This will create a 225x28x28 image where each cell represents the weights of a cluster. Perform this for at least 20 epochs to observe the weight update process effectively.

* d. Display images of the weight clusters (up to 31 clusters).

## Mini Project 1: Convolutional Neural Networks (CNN) for Fashion-MNIST Classification

### Question 1: CNN for Fashion-MNIST

In this exercise, we aim to classify the Fashion-MNIST dataset using Convolutional Neural Networks (CNNs).

* a. Load the dataset using the provided code:

import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

Perform the necessary preprocessing for training an MLP network with two hidden layers. Use stochastic mini-batch training and obtain error and accuracy plots, as well as a confusion matrix for three different batch sizes: 32, 60, and 256.

* b. Change the activation function and loss function and investigate their effects on accuracy and error. Try using activation functions like TanH, ReLU, and Sigmoid, and loss functions like Cross Entropy and Mean Squared Error (MSE).

* c. Add convolutional layers to the best-performing network obtained in part (a) and compare the results in terms of accuracy and error with the previous setup.

* d. Explain the functions of Batch Normalization and Pooling layers and add them to your network. Then, evaluate the results in terms of accuracy and error.

* e. Discuss the purpose of the Dropout layer. Add a Dropout layer to your network and investigate its effects on accuracy and error for different dropout percentages.

### Question 2: Transfer Learning

In this question, we will explore different neural network architectures based on CNNs that have been developed for various tasks.

Choose a model based on your student ID's last digit (or any other method you prefer) from the following list:

** AlexNet
** VGG-16
** VGG-11
** Inception
** GoogLeNet
** ResNet
** SqueezeNet
** DenseNet
** ShuffleNet
** ENet

* a. Explain the architecture of the selected model, including its advantages and disadvantages. Discuss any necessary preprocessing for input data.

* b. Implement the chosen model and, after preprocessing, select three arbitrary images to make predictions. Explain the results.

* c. Investigate which types of images your model can recognize. If it fails to recognize certain images, discuss potential solutions.

* d. Choose a category of different images that your model can recognize, preprocess them accordingly, and retrain the model. You can use datasets like ImageNet or other suitable ones.

### Question 3: Semantic Segmentation

Semantic segmentation is the task of labeling each pixel in an image with a corresponding class label. Select one of the following models: UNet or DeepLab, and answer the following questions:

* a. Study the paper related to the selected model and explain its architecture and application comprehensively.

* b. Implement the chosen model, and apply it to an arbitrary image after necessary preprocessing. Describe the results.

### Question 4: Object Detection

Object detection involves identifying and locating objects within an image. You will explore YOLO (You Only Look Once) models in this question.

* a. Explain the differences between one-stage and two-stage object detection approaches. Provide an example for each.

* b. Discuss the challenges of YOLOv1 in handling overlapping objects. Explain the solution adopted in YOLOv2.

* c. Describe the advancements in YOLOv3 and YOLOv5 compared to previous versions.

* d. Specify the hyperparameters that need adjustment when training YOLOv3 on a custom dataset. Explain their roles and how to determine suitable values.

* e. Train YOLOv5s and YOLOv5x on the provided dataset and compare their results, including mAP, 1.5 mAP, 1.501.15 Precision, Recall, and training/inference time. Discuss the advantages and disadvantages of using a larger model versus a smaller one.

## Mini Project 2: Music Generation and Music Note Recognition

### Question 1: Music Generation

* Part 1: Loading the Data
In this task, we will generate music using Recurrent Neural Networks (RNNs). We'll start by loading the Classical Music MIDI dataset, specifically focusing on compositions by the famous Polish composer Frédéric Chopin. You should use the music21 library to read the music files, extract chords and notes, and store them in a list.

* Part 2: Data Preprocessing
After obtaining the list of musical elements (chords and notes), you should count the occurrences of each note and remove rare notes from the corpus. You can specify a minimum threshold for the occurrence of notes to remove. To create training and testing data, you can read sequences of 40 notes from the corpus and consider the 41st note as the label. Then, encode the sequences and labels, converting labels into one-hot encoding.

* Part 3: Implementing the Recurrent Neural Network (RNN)
Implement an RNN-based neural network for music generation. You can design your own architecture or use a predefined one. Use the categorical cross-entropy loss function and a softmax activation function for the fully connected layer. Explain the chosen network structure in your report.

* Part 4: Network Evaluation
Print the loss during the training process. Generate a piece of music using the provided function and upload it along with your report. You can convert the generated MIDI music into WAV format using available online tools.

Repeat the above steps for compositions composed by Mozart.

Experiment with the dropout layer's impact on network performance. Add dropout layers after each LSTM layer and fully connected layers.

### Question 2: Music Note Recognition

* Part 1: Data Preparation
Convert the MIDI files into WAV format and visualize the waveforms of files "0_0.wav" and "0_12.wav". Explain the horizontal and vertical axes in the waveform plots.

* Part 2: Creating Training, Validation, and Test Sets
Divide the dataset into three parts and provide a rationale for this division, considering the number of samples in each category, the instruments in each category, etc.

* Part 3: Building the Recurrent Neural Network (RNN)
Design RNN models to identify the musical notes in a short audio sample of 250 milliseconds. Implement RNNs with different types of layers (RNN, LSTM, GRU) and evaluate them based on speed and accuracy.

** a. Implement the RNNs with LSTM, GRU, and SimpleRNN layers, and compare their performance in terms of speed and accuracy.

** b. Examine the impact of dropout layers on all three RNN models.

** c. Test the best-performing RNN model with frame sizes of 35, 70, 150, and 300 and analyze the results.

** d. Select and explain a few feature extraction algorithms (e.g., Fourier transform, wavelet transform) suitable for this task. Use these algorithms to improve the performance of the best RNN model from the previous section.

** e. (Optional) Discuss data augmentation techniques suitable for this task. Try to increase the accuracy of the best RNN model by adding at least one of these techniques.

** f. (Optional) Utilize spectrogram tools to convert audio frames into images. Combine CNN and RNN architectures to make predictions and compare the results with the best-performing models from previous sections.

