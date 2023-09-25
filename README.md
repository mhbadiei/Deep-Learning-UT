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
