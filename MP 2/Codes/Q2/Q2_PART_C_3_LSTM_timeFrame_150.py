from scipy.io import wavfile
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time
random.seed(42)

class noteRecognizer:
    # Data length should be between 1 and length of data
    # Time frame should be in unit of mili second 
    def __init__(self, compressionLength, timeFrame, notesLength, instrumentLength):
        self.model = Sequential()
        self.compressionLength = compressionLength
        self.timeFrame = timeFrame
        self.notesLength = notesLength
        self.instrumentLength = instrumentLength
        self.musicLength = 1000 # 1000 ms
        self.dataFrameLength = int((self.timeFrame/self.musicLength)* self.compressionLength)
        self.allFiles = [str(j)+'_'+str(i)+'.wav' for j in range(instrumentLength) for i in range(self.notesLength)]
        random.shuffle(self.allFiles)


    def trainTestSeperation(self, trainSize):
        self.trainSize = trainSize
        self.dataset = {'Train':[], 'Test':[]}
        self.dataset['Train'] = ([x for x in self.allFiles[:int(self.trainSize*self.instrumentLength*self.notesLength)]])
        self.dataset['Test'] = ([x for x in self.allFiles[int(self.trainSize*self.instrumentLength*self.notesLength):]])
        
    def loadData(self, path):
        self.path = path
        self.xTrain = np.empty((0, self.dataFrameLength), int)
        self.xTest = np.empty((0, self.dataFrameLength), int)
        self.yTrain = []
        self.yTest = []
        for fileName in self.dataset['Train']:
            xTrain_ = self.processing(self.path + fileName)
            self.xTrain = np.append(self.xTrain, xTrain_ , axis=0)
            self.yTrain.extend(np.repeat(fileName.split('.wav')[0].split('_')[1], np.shape(xTrain_)[0])) 
        for fileName in self.dataset['Test']:
            xTest = self.processing(self.path + fileName)
            self.xTest = np.append(self.xTest, xTest , axis=0)
            self.yTest.extend(np.repeat(fileName.split('.wav')[0].split('_')[1], np.shape(xTest)[0])) 
        self.xTrain = np.reshape(self.xTrain, (self.xTrain.shape[0], 1, self.xTrain.shape[1]))
        self.xTest = np.reshape(self.xTest, (self.xTest.shape[0], 1, self.xTest.shape[1]))
        self.yTrain = np_utils.to_categorical(self.yTrain, self.notesLength)
        self.yTest = np_utils.to_categorical(self.yTest, self.notesLength)
        #print(np.shape(self.xTrain),np.shape(self.xTest),np.shape(self.yTrain),np.shape(self.yTest))
        
    def processing(self, filePath):
        rate, wav = wavfile.read(filePath)
        data = np.array([wav[i*int(len(wav)/self.compressionLength), 0] for i in range(self.compressionLength)])
        data_ = np.array([float(i)/max(data) if max(data) != 0 else float(i)/20 for i in data])
        return [data_[i:i+self.dataFrameLength, ] for i in range(self.compressionLength - self.dataFrameLength)]
        
    def train(self, epochs=50, batch_size=256, validation_split=0.12, learningRate = 0.001):
        self.optimiser = tf.keras.optimizers.Adam(lr=learningRate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        t = time.time()
        self.train_()
        print('Training Time: ', time.time() - t)
        
    
    
    def train_(self):
        self.model.add(tf.keras.layers.LSTM(128, input_shape=(1, self.dataFrameLength), activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.05))
        self.model.add(Dense(self.notesLength, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimiser, metrics=['accuracy'])
        self.model.summary()
        self.result = self.model.fit(self.xTrain, self.yTrain, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
    
    def plotAccuracy(self):
        trainAcc = self.result.history['accuracy']
        evalAcc = self.result.history['val_accuracy']
        plt.figure(figsize=(8, 4))
        plt.plot(trainAcc,'red')
        plt.plot(evalAcc)
        plt.title('Accuracy of Model')
        plt.ylabel("ACCURACY")  
        plt.xlabel('EPOCH')
        plt.legend(['Training Set', 'Validation Set'], loc='lower right')
        plt.grid()

    def plotLoss(self):
        trainLoss = self.result.history['loss']
        evalLoss = self.result.history['val_loss']
        plt.figure(figsize=(8, 4))
        plt.plot(trainLoss,'red')
        plt.plot(evalLoss)
        plt.title('Loss of Model')
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH')
        plt.legend(['Training Set', 'Validation Set'], loc='upper right')
        plt.grid()
    
    def testModel(self):
        res = self.model.evaluate(self.xTest, self.yTest)
        print("Test Loss, Test Accuracy:", res)
    
    def showPlot(self):
        plt.show()
    
if __name__ == "__main__" :
    # Data length should be between 1 and length of data (in this case, it can be 44100)
    # Time frame should be in unit of mili second     
    model = noteRecognizer(compressionLength = 200, timeFrame=150, notesLength = 10, instrumentLength = 10)
    model.trainTestSeperation(trainSize = 0.85) # this function just seperate the names of file, so it dose not need to load the dataset before call it
    model.loadData(path = './../notes/wav/')
    model.train(epochs=50, batch_size=256, validation_split=0.15, learningRate=0.001)
    model.plotAccuracy()
    model.plotLoss()
    model.showPlot()
    model.testModel()