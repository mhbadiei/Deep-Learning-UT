"""
Implementation of multilayer perceptron network (MLP) for Regression applications
Author : Abbas Badiei
mh.badiei@ut.ac.ir
Student at Tehran University, School of Electrical and Computer Engineering
"""
from keras import activations
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import itertools
import time
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

class MLP_Regression:
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
        self.yTest = []
        self.model = None
        self.result = None
        self.dropout = None

    def datasetRetrieval(self):
        with open("./Reg-Data.txt", 'r') as f:
            data = np.array([[element for element in np.array(line.split(','))] for line in f])
            data = self.normalization(data)
            data = self.standardization(data)
            self.label = np.array(data[:, 68:70]).astype(np.float)
            self.data = np.array(data[:, 0:68]).astype(np.float)

    def normalization(self, x):
        normalizedData = MinMaxScaler()
        return normalizedData.fit_transform(x)

    def standardization(self, x):
        srandardizedData = StandardScaler()
        return srandardizedData.fit_transform(x)

    def trainTestSplit(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.data, self.label, test_size=0.15,random_state=46)

    def train(self,epochs = 10, NoOfNeuronsPerLayer = [], activationFunction = ['linear'], loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'], batch_size = 64, validation_split = 0.2, dropout =0, batchNormalizationFlag=False):
        self.epochs = epochs
        self.NoOfNeuronsPerLayer = NoOfNeuronsPerLayer
        self.activationFunction = activationFunction
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dropout = dropout
        self.batchNormalizationFlag = batchNormalizationFlag
        self.train_()

    def train_(self):
        model = Sequential()
        model.add(Input(shape=(np.shape(self.data)[1],)))
        if (self.dropout): model.add(Dropout(self.dropout))
        if (self.batchNormalizationFlag): model.add(BatchNormalization())
        for count, No in enumerate(self.NoOfNeuronsPerLayer):
            layerName = "Hidden-Layer-No-" + str(count+1)
            print(No, self.activationFunction[count], layerName)
            model.add(Dense(No, activation=self.activationFunction[count], name=layerName))
        model.add(Dense(2, activation=self.activationFunction[-1], name="Output-Layer")) 
        model.summary()
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        res = model.fit(self.xTrain, self.yTrain, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        self.result = res
        self.model = model

    def plotLoss(self):
        trainLoss = self.result.history['loss']
        evalLoss = self.result.history['val_loss']
        plt.figure()
        plt.plot(trainLoss,'red')
        plt.plot(evalLoss)
        plt.title('Loss of MLP Model')
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH')
        plt.legend(['Training Set', 'Validation Set'], loc='upper right')
        #plt.grid()
        
    def testModel(self):
        results = self.model.evaluate(self.xTest, self.yTest)
        print("Test Loss, Test Accuracy:", results)

    def showErrorTable(self):
        y_pred = self.model.predict(self.xTest)
        clust_data = [[metrics.mean_absolute_error(self.yTest, y_pred), metrics.mean_squared_error(self.yTest, y_pred), np.sqrt(metrics.mean_squared_error(self.yTest, y_pred))]]
        plt.figure()
        collabel=["MAE", "MSE", "RMSE"]
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=clust_data,colLabels=collabel,loc='center')

    def showPlot(self):
        plt.show()

def plotTable(clust_data, row_labels):
    plt.figure()
    plt.axis('tight')
    plt.axis('off')
    the_table = plt.table(cellText=clust_data,rowLabels=row_labels,loc='center')

if __name__ == "__main__":
#    NoOfNeuronsPerLayerList = [[6, 2], [15, 6], [55, 40]]

    model = MLP_Regression()
    model.datasetRetrieval()
    model.trainTestSplit()
    model.train()
 
    """ Part 2 """
    # epochsList = [10, 50]
    # lossFuncList = ['mean_squared_error', 'mean_absolute_error']
    # optimizerList = ["adam", "sgd"]
    # elapsed = np.zeros((8, 1))
    # i = 0
    # collabel = []
    # for epochs in epochsList:
    #     for lossFunc in lossFuncList:
    #         for optimizer in optimizerList:
    #             t = time.time()
    #             model.train(epochs = epochs, loss=lossFunc, optimizer=optimizer)
    #             elapsed[i,0] = time.time() - t        
    #             #model.testModel()
    #             model.plotLoss()
    #             model.showErrorTable()
    #             collabel.append("Time")
    #             i += 1
    # plotTable(elapsed, collabel)

    """ Part 3 """
    # epochsList = [10, 50]
    # elapsed = np.zeros((2, 1))
    # i = 0
    # collabel = []
    # for epochs in epochsList:
    #     t = time.time()
    #     model.train(epochs = epochs, loss='mean_squared_error', optimizer="sgd", activationFunction=['softsign'])
    #     elapsed[i,0] = time.time() - t        
    #     #model.testModel()
    #     model.plotLoss()
    #     model.showErrorTable()
    #     collabel.append("Time")
    #     i += 1
    # plotTable(elapsed, collabel)

    """ Part 4 """
    # batchSizeList = [16, 64, 256]
    # elapsed = np.zeros((3, 1))
    # i = 0
    # collabel = []
    # for batchSize in batchSizeList:
    #     t = time.time()
    #     model.train(epochs = 50, loss='mean_squared_error', optimizer="sgd", activationFunction=['softsign'], batch_size = batchSize)
    #     elapsed[i,0] = time.time() - t        
    #     #model.testModel()
    #     model.plotLoss()
    #     model.showErrorTable()
    #     collabel.append("Time")
    #     i += 1
    # plotTable(elapsed, collabel)

    """ Part 5 """
    # addLayersList = [[25],[25,12]]
    # lossFuncList = ['mean_squared_error', 'mean_absolute_error']
    # optimizerList = ["adam", "sgd"]
    # elapsed = np.zeros((8, 1))
    # i = 0
    # collabel = []
    # for No in addLayersList:
    #     for lossFunc in lossFuncList:
    #         for optimizer in optimizerList:
    #             t = time.time()
    #             lengthOfHiddenLayer = len(No)
    #             NoOfReluActivationFunc =  ['relu']*(lengthOfHiddenLayer)
    #             model.train(epochs = 50, NoOfNeuronsPerLayer = No, activationFunction = NoOfReluActivationFunc + ['softsign'] , loss = lossFunc, optimizer="sgd", batch_size = 64)
    #             elapsed[i,0] = time.time() - t        
    #             #model.testModel()
    #             model.plotLoss()
    #             model.showErrorTable()
    #             collabel.append("Time")
    #             i += 1
    # plotTable(elapsed, collabel)

    """ Part 6 """
    # addLayersList = [[], [25], [25,12]]
    # winnerModel = [('mean_squared_error', 'sgd', 'linear'), ('mean_squared_error', 'adam', 'softsign'), ('mean_absolute_error', 'adam', 'softsign')]
    # elapsed = np.zeros((3, 1))
    # i = 0
    # collabel = []
    # for lossFunc, optimizer, activationFunc in winnerModel:
    #     t = time.time()
    #     lengthOfHiddenLayer = len(addLayersList[i])
    #     NoOfReluActivationFunc =  ['relu']*(lengthOfHiddenLayer)
    #     model.train(epochs = 128, NoOfNeuronsPerLayer = addLayersList[i], activationFunction = NoOfReluActivationFunc + [activationFunc] , loss = lossFunc, optimizer=optimizer, batch_size = 64, dropout=0.25)
    #     elapsed[i,0] = time.time() - t        
    #     #model.testModel()
    #     model.plotLoss()
    #     model.showErrorTable()
    #     collabel.append("Time")
    #     i += 1
    # plotTable(elapsed, collabel)

    # """ Part 7 """
    # addLayersList = [[], [25], [25,12]]
    # winnerModel = [('mean_squared_error', 'sgd', 'linear'), ('mean_squared_error', 'adam', 'softsign'), ('mean_absolute_error', 'adam', 'softsign')]
    # elapsed = np.zeros((3, 1))
    # i = 0
    # collabel = []
    # for lossFunc, optimizer, activationFunc in winnerModel:
    #     t = time.time()
    #     lengthOfHiddenLayer = len(addLayersList[i])
    #     NoOfReluActivationFunc =  ['relu']*(lengthOfHiddenLayer)
    #     model.train(epochs = 256, NoOfNeuronsPerLayer = addLayersList[i], activationFunction = NoOfReluActivationFunc + [activationFunc] , loss = lossFunc, optimizer=optimizer, batch_size = 64, dropout=0.25, batchNormalizationFlag=True)
    #     elapsed[i,0] = time.time() - t        
    #     #model.testModel()
    #     model.plotLoss()
    #     model.showErrorTable()
    #     collabel.append("Time")
    #     i += 1
    # plotTable(elapsed, collabel)

    """ Part 7 """
    addLayersList = [[32,22,12]]
    winnerModel = [('mean_absolute_error', 'adam', 'softsign')]
    elapsed = np.zeros((1, 1))
    i = 0
    collabel = []
    for lossFunc, optimizer, activationFunc in winnerModel:
        t = time.time()
        lengthOfHiddenLayer = len(addLayersList[i])
        NoOfReluActivationFunc =  ['softsign']*(lengthOfHiddenLayer)
        model.train(epochs = 256, NoOfNeuronsPerLayer = addLayersList[i], activationFunction = NoOfReluActivationFunc + [activationFunc] , loss = lossFunc, optimizer=optimizer, batch_size = 64, dropout=0.25, batchNormalizationFlag=True)
        elapsed[i,0] = time.time() - t        
        #model.testModel()
        model.plotLoss()
        model.showErrorTable()
        collabel.append("Time")
        i += 1
    plotTable(elapsed, collabel)


    model.showPlot()

