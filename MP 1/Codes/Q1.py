"""
Implementation of multilayer perceptron network (MLP) with convolutional neural network (CNN) for classification applications
Authors : 
Abbas Badiei --------------------> mh.badiei@ut.ac.ir
saeed mohammadi dashtaki --------> saeedmohammadi.d@ut.ac.ir
Students at Tehran University, School of Electrical and Computer Engineering
"""
import numpy as np
import copy
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from numpy.core.arrayprint import DatetimeFormat
from numpy.core.fromnumeric import shape
from sklearn.model_selection import train_test_split
from keras.layers import Input,Conv2D, Flatten, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_recall_fscore_support
import time
from google.colab import files
from keras.utils import np_utils
from tensorflow.keras.layers import BatchNormalization, MaxPool2D

class Classifier:
    def __init__(self):
        self.epochs = None
        self.test_size = None
        self.NoOfNeuronsPerLayer = []
        self.activationFunction = []
        self.loss = ''
        self.optimizer = ''
        self.metrics = ''
        self.batch_size = None
        self.validation_split = None
        self.data = []
        self.label = []
        self.xTrain = []
        self.xTest = []
        self.yTrain = []
        self.YTest = []
        self.yTest = []
        self.model = None
        self.result = None
        self.cnnFlag = False
        self.pool_batchNormalizationFlag = False
        self.dropout = None

    def datasetRetrieval(self):
        (self.xTrain, self.yTrain), (self.xTest, self.yTest) = fashion_mnist.load_data()
        self.YTest = copy.copy(self.yTest)
        self.yTrain = np_utils.to_categorical(self.yTrain)
        self.yTest = np_utils.to_categorical(self.yTest)

    def normalization(self):
        self.xTrain = np.reshape(self.xTrain, (60000, 784)).astype('float32') 
        self.xTest = np.reshape(self.xTest, (10000, 784)).astype('float32') 
        normalizedData = MinMaxScaler()
        self.xTrain = normalizedData.fit_transform(self.xTrain)
        self.xTest = normalizedData.fit_transform(self.xTest)

    def standardization(self):
        srandardizedData = StandardScaler()
        self.xTrain = srandardizedData.fit_transform(self.xTrain)
        self.xTest = srandardizedData.fit_transform(self.xTest)
        self.xTrain = np.reshape(self.xTrain, (60000, 28, 28,1)) 
        self.xTest = np.reshape(self.xTest, (10000, 28, 28,1)) 

    def train(self,epochs = 15, NoOfNeuronsPerLayer = [400, 200], activationFunction = ['relu', 'relu', 'softmax'], loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'], batch_size = 256, validation_split = 0.2, cnnFlag = False, pool_batchNormalizationFlag = False, dropout=None):
        self.epochs = epochs
        self.NoOfNeuronsPerLayer = NoOfNeuronsPerLayer
        self.activationFunction = activationFunction
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.cnnFlag = cnnFlag
        self.pool_batchNormalizationFlag = pool_batchNormalizationFlag
        self.dropout = dropout
        self.train_()

    def train_(self):
        model = Sequential()
        model.add(Input(shape=self.xTrain[0].shape))
        if (self.dropout): model.add(Dropout(self.dropout))
        if(self.cnnFlag):
          model.add(Conv2D(64, (3, 3),strides = 2 , activation='relu'))
          model.add(Conv2D(64, (3, 3),strides = 2 , activation='relu'))
          model.add(Conv2D(64, (3, 3),strides = 2 , activation='relu'))
          if self.pool_batchNormalizationFlag:
            model.add(MaxPool2D(pool_size=(2,2)))
            model.add(BatchNormalization())

        model.add(Flatten())
        if (self.dropout): model.add(Dropout(self.dropout))
        for count, No in enumerate(self.NoOfNeuronsPerLayer):
            layerName = "Hidden-Layer-No-" + str(count+1)
            model.add(Dense(No, activation=self.activationFunction[count], name=layerName))
        model.add(Dense(10, activation=self.activationFunction[-1], name="Output-Layer")) 
        model.summary()
        model.compile(optimizer=self.optimizer , loss = self.loss, metrics=self.metrics)
        res = model.fit(self.xTrain, self.yTrain, epochs=self.epochs, validation_split=self.validation_split)
        self.result = res
        self.model = model

    def plotAccuracy(self):
        trainAcc = self.result.history['accuracy']
        evalAcc = self.result.history['val_accuracy']
        plt.figure()
        plt.plot(trainAcc,'red')
        plt.plot(evalAcc)
        plt.title('Accuracy of Model')
        plt.ylabel('ACCURACY')
        plt.xlabel('EPOCH')
        plt.legend(['Training Set', 'Validation Set'], loc='lower right')
        plt.grid()        

    def plotLoss(self):
        trainLoss = self.result.history['loss']
        evalLoss = self.result.history['val_loss']
        plt.figure()
        plt.plot(trainLoss,'red')
        plt.plot(evalLoss)
        plt.title('Loss of Model')
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH')
        plt.legend(['Training Set', 'Validation Set'], loc='upper right')
        plt.grid()
        
    def testModel(self):
        results = self.model.evaluate(self.xTest, self.yTest)
        print("Test Loss, Test Accuracy:", results)

    def showPlot(self):
        plt.show()

    def plotConfusionMatrix(self):
        cmap=plt.cm.Reds
        yPred = self.model.predict(self.xTest)
        predicted_categories = tf.argmax(yPred, axis=1)
        classes=[0,1,2,3,4,5,6,7,8,9]
        cm = confusion_matrix(self.YTest, predicted_categories)
        print("Confusion Matrix of Test Data:\n",cm)
        
        plt.figure(figsize = (4,4))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

def plotTable(clust_data, row_labels):
    plt.figure()
    plt.axis('tight')
    plt.axis('off')
    plt.title('Train Time')
    the_table = plt.table(cellText=clust_data,rowLabels=row_labels,loc='center')


if __name__ == "__main__":
    model = Classifier()
    model.datasetRetrieval()
    model.normalization()
    model.standardization()

    # this function just set the initial parameters of model in order to balance time complexity of algorithm for distinct inputs  
    model.train(epochs=1)

    """ Part A """
    # batchSizeList = [32, 64, 256]
    # elapsed = np.zeros((len(batchSizeList), 1))
    # rowLabel = [] 
    # for count, batchSize in enumerate(batchSizeList):
    #   t = time.time()
    #   model.train(batch_size = batchSize)
    #   elapsed[count] = time.time() - t        
    #   model.testModel()
    #   model.plotAccuracy()
    #   model.plotLoss()
    #   model.plotConfusionMatrix()
    #   rowLabel.append('batch: ' + str(batchSize))
    # plotTable(elapsed, rowLabel)

    """ Part B """
    # activationFunctionList = ['tanh', 'relu', 'sigmoid']
    # lossFuncList = ['categorical_crossentropy','mean_squared_error']
    # elapsed = np.zeros((len(activationFunctionList)*len(lossFuncList), 1))
    # rowLabel = [] 
    # count= 0
    # for loss in lossFuncList:
    #   for activationFunction in activationFunctionList:
    #     t = time.time()
    #     model.train(batch_size = 256, activationFunction = ['relu'] + [activationFunction]+['softmax'], loss=loss)
    #     elapsed[count] = time.time() - t        
    #     model.testModel()
    #     model.plotAccuracy()
    #     model.plotLoss()
    #     model.plotConfusionMatrix()
    #     count+=1
    #     rowLabel.append('Loss Function: ' + str(loss) + 'Activation Function: ' + str(activationFunction))
    # plotTable(elapsed, rowLabel)

    """ Part C """
    # elapsed = np.zeros((1,1))
    # t = time.time()
    # model.train(batch_size = 256, activationFunction = ['relu'] + ['sigmoid'] + ['softmax'], loss = 'categorical_crossentropy', cnnFlag = True)
    # elapsed[0] = time.time() - t        
    # model.testModel()
    # model.plotAccuracy()
    # model.plotLoss()
    # model.plotConfusionMatrix()
    # rowLabel = ['CNN Time Complexity']
    # plotTable(elapsed, rowLabel)

    """ Part D """
    # elapsed = np.zeros((1,1))
    # t = time.time()
    # model.train( batch_size = 256, activationFunction = ['relu'] + ['sigmoid'] + ['softmax'], loss = 'categorical_crossentropy', cnnFlag = True, pool_batchNormalizationFlag = True)
    # elapsed[0] = time.time() - t        
    # model.testModel()
    # model.plotAccuracy()
    # model.plotLoss()
    # model.plotConfusionMatrix()
    # rowLabel = ['CNN Time Complexity']
    # plotTable(elapsed, rowLabel)

    """ Part E """
    dropoutList = [0.05, 0.1, 0.3]
    elapsed = np.zeros((len(dropoutList), 1))
    rowLabel = [] 
    for count, dropout in enumerate(dropoutList):
      t = time.time()
      model.train( batch_size = 256, activationFunction = ['relu'] + ['sigmoid'] + ['softmax'], loss = 'categorical_crossentropy', cnnFlag = True, pool_batchNormalizationFlag = True, dropout=dropout)
      elapsed[count] = time.time() - t        
      model.testModel()
      model.plotAccuracy()
      model.plotLoss()
      model.plotConfusionMatrix()
      rowLabel.append('dropout :'+str(dropout))
      count+=1
    plotTable(elapsed, rowLabel)

    model.showPlot()

