"""
Implementation of Autoencoder with multilayer perceptron network (MLP)
Author : Abbas Badiei
mh.badiei@ut.ac.ir
Student at Tehran University, School of Electrical and Computer Engineering
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from numpy.core.arrayprint import DatetimeFormat
from numpy.core.fromnumeric import shape
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
import time

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

    def train(self,epochs = 220, NoOfNeuronsPerLayer = [], activationFunction = ['linear'], loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'], batch_size = 64, validation_split = 0.2, dropout =0, batchNormalizationFlag=False):
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

    def addAutoencoder(self, encodingDimension = 20):
        self.encodingDimension = encodingDimension
        xTrain, xTest, _, _= train_test_split(self.data, self.label, train_size = 0.1 )
        input_dim = Input(shape = (np.shape(self.data)[1], ))
        encoded1 = Dense(encodingDimension, activation = 'relu')(input_dim)
        decoded1 = Dense(np.shape(self.data)[1], activation = 'sigmoid')(encoded1)
        autoencoder = Model(inputs = input_dim, outputs = decoded1)
        autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        autoencoder.summary()
        autoencoder.fit(xTrain, xTrain, epochs = 20, batch_size = 64, validation_data = (xTest, xTest))
        encoder = Model(inputs = input_dim, outputs = encoded1)
        self.encodedData = np.array(encoder.predict(self.data))
        self.label = self.data

def plotTable(clust_data, row_labels):
    plt.figure()
    plt.axis('tight')
    plt.axis('off')
    the_table = plt.table(cellText=clust_data,rowLabels=row_labels,loc='center')


if __name__ == "__main__":
    model = MLP_Regression()
    model.datasetRetrieval()
    model.trainTestSplit()
    model.addAutoencoder(encodingDimension = 10)
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

