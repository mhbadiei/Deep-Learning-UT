"""
Implementation of Regression
Author : Abbas Badiei
mh.badiei@ut.ac.ir
Student at Tehran University, School of Electrical and Computer Engineering
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import time

class Linear_Regression:
    def __init__(self):
        self.data = []
        self.label = []
        self.xTrain = []
        self.xTest = []
        self.yTrain = []
        self.yTest = []
        self.model = None

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

    def train(self):
        self.train_()

    def train_(self):
        t = time.time()
        self.model = LinearRegression().fit(self.xTrain, self.yTrain)
        print('training time :', time.time() - t)  
    
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

        
if __name__ == "__main__":
    model = Linear_Regression()
    model.datasetRetrieval()
    model.trainTestSplit()
    model.train()
    model.showErrorTable()
    model.showPlot()

