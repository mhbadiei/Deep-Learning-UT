"""
Implementation of multilayer perceptron network (MLP) for classification applications
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
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_recall_fscore_support
import time

class MLP_Classification:
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

    def datasetRetrieval(self):
        with open('./ionosphere.data', 'r') as f:
            data = np.array([[element for element in np.array(line.split(','))] for line in f])
            self.label = np.array([[0 if element=='g\n' else 1] for element in data[:,34]])   
            self.data = np.array(data[:,0:34].astype(np.float))

    def normalization(self):
        normalizedData = MinMaxScaler()
        self.data = normalizedData.fit_transform(self.data)

    def standardization(self):
        srandardizedData = StandardScaler()
        self.data = srandardizedData.fit_transform(self.data)

    def trainTestSplit(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.data, self.label, test_size=0.15,random_state=46)

    def train(self,epochs = 220, NoOfNeuronsPerLayer = [15, 6], activationFunction = ['relu', 'softplus', 'sigmoid'], loss = 'hinge', optimizer = 'adam', metrics = ['accuracy'], batch_size = 32, validation_split = 0.2):
        self.epochs = epochs
        self.NoOfNeuronsPerLayer = NoOfNeuronsPerLayer
        self.activationFunction = activationFunction
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.train_()

    def train_(self):
        model = Sequential()
        model.add(Input(shape=(np.shape(self.data)[1],)))
        for count, No in enumerate(self.NoOfNeuronsPerLayer):
            layerName = "Hidden-Layer-No-" + str(count+1)
            print(No, self.activationFunction[count], layerName)
            model.add(Dense(No, activation=self.activationFunction[count], name=layerName))
        model.add(Dense(1, activation=self.activationFunction[-1], name="Output-Layer")) 
        model.summary()
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        res = model.fit(self.xTrain, self.yTrain, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        self.result = res
        self.model = model

    def plotAccuracy(self):
        trainAcc = self.result.history['accuracy']
        evalAcc = self.result.history['val_accuracy']
        plt.figure()
        plt.plot(trainAcc,'red')
        plt.plot(evalAcc)
        plt.title('Accuracy of MLP Model')
        plt.ylabel('ACCURACY')
        plt.xlabel('EPOCH')
        plt.legend(['Training Set', 'Validation Set'], loc='lower right')
        #plt.grid()

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

    def showPlot(self):
        plt.show()

    def activation(self,x):
        return np.where(x >= 0.5, np.where(x < 0.5, x, 1.0), 0.0)

    def plotConfusionMatrix(self):
        cmap=plt.cm.Reds
        yPred = self.activation(self.model.predict(self.xTest))
        classes=[0,1]
        cm = confusion_matrix(self.yTest, yPred)
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

    def precisionRecallFscorSupport(self):
        yPred = self.activation(self.model.predict(self.xTest))
        print(precision_recall_fscore_support(self.yTest, yPred, average='binary'))

def plotTable(clust_data):
    plt.figure()
    collabel=("1st Time", "2nd Time", "3rd Time")
    plt.axis('tight')
    plt.axis('off')
    the_table = plt.table(cellText=clust_data,colLabels=collabel,loc='center')


if __name__ == "__main__":
    elapsed = np.zeros((1, 3))
    NoOfNeuronsPerLayerList = [[6, 2], [15, 6], [55, 40]]
    batchSizeList = [32, 64, 256]
    secondActivationFuncList = ['softplus','sigmoid', 'relu', 'tanh']
    lossFuncList = ['hinge', 'squared_hinge','binary_crossentropy']
    optimizerList = ["adam", "sgd", "rmsprop"]
    addLayersList = [[15,6,4],[15,6,4,3],[15,6,4,3,2]]

    model = MLP_Classification()
    model.datasetRetrieval()
    model.normalization()
    model.standardization()
    model.trainTestSplit()

    # this function just set the initial parameters of model in order to balance time complexity of algorithm for distinct inputs  
    model.train()

    """ Part B """
    #for count, No in enumerate(NoOfNeuronsPerLayerList):
    #    t = time.time()
    #    model.train(NoOfNeuronsPerLayer = No)
    #    elapsed[0,count] = time.time() - t        
    #    model.testModel()
    #    model.plotAccuracy()
    #    model.plotLoss()
    #    model.plotConfusionMatrix()
    #plotTable(elapsed)

    """ Part C """
    #for count, batchSize in enumerate(batchSizeList):
    #   t = time.time()
    #   model.train(batch_size = batchSize)
    #   elapsed[0,count] = time.time() - t        
    #   model.testModel()
    #   model.plotAccuracy()
    #   model.plotLoss()
    #   model.plotConfusionMatrix()
    #plotTable(elapsed)


    """ Part D """
    #for count, secondActivationFunc in enumerate(secondActivationFuncList):
    #  model.train(batch_size = 64, NoOfNeuronsPerLayer = [15, 6], activationFunction = ['relu', secondActivationFunc, 'sigmoid'])
    #  model.testModel()
    #  model.plotAccuracy()
    #  model.plotLoss()
    #  model.plotConfusionMatrix()

    """" Part E """
    # for count, lossFunc in enumerate(lossFuncList):
    #     model.train(batch_size = 64, NoOfNeuronsPerLayer = [15, 6], activationFunction = ['relu', 'tanh', 'sigmoid'], loss=lossFunc)
    #     model.testModel()
    #     model.plotAccuracy()
    #     model.plotLoss()
    #     model.plotConfusionMatrix()

    """ Part F """
    # for count, optm in enumerate(optimizerList):
    #     model.train(batch_size = 64, NoOfNeuronsPerLayer = [15, 6], activationFunction = ['relu', 'tanh', 'sigmoid'], loss='binary_crossentropy', optimizer=optm)
    #     model.testModel()
    #     model.plotAccuracy()
    #     model.plotLoss()
    #     model.plotConfusionMatrix()

    """ Part G """
    for count, No in enumerate(addLayersList):
        lengthOfHiddenLayer = len(No)
        NoOfReluActivationFunc =  ['relu']*(lengthOfHiddenLayer - 1)
        model.train(batch_size = 64, NoOfNeuronsPerLayer = No, activationFunction = NoOfReluActivationFunc + ['tanh', 'sigmoid'], loss='binary_crossentropy', optimizer='adam')
        model.testModel()
        model.plotAccuracy()
        model.plotLoss()
        model.plotConfusionMatrix()
        model.precisionRecallFscorSupport()

    model.showPlot()

