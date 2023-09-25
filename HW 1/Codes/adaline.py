import matplotlib.pyplot as plt
import numpy as np

class Adaline:
    def __init__(self, learningRate = 0.01, epoch = 200):
        self.input = []
        self.target = []
        self.test = []
        self.weight = []
        self.learningRate = learningRate
        self.epoch = epoch
        self.error = []

    def activationFunction(self,x):
        return np.where(x >= 0, np.where(x < 0, x, 1), -1)

    def generateTrainingSet(self,mu=[[2,0],[0,1]], sigma=[[0.5,0.2],[0.1,0.7]]):
        class1 = np.vstack((np.random.normal(mu[0][0], sigma[0][0], 100),np.random.normal(mu[0][1], sigma[0][1], 100),np.ones(100))).T
        class2 = np.vstack((np.random.normal(mu[1][0], sigma[1][0], 30),np.random.normal(mu[1][1], sigma[1][1], 30),np.ones(30))).T
        self.input = np.concatenate((class1, class2), axis=0)
        self.target = np.concatenate((np.ones(100), -np.ones(30)), axis=0).T

    def train(self, activation = 'sign'):
        self.error = []
        w = np.zeros((1,3))
        self.weight = 2*np.random.random((1,3)) - 1
        for epoch in range(self.epoch):
            self.error.append(self.train_(activation))
            if (np.absolute(w[0]-self.weight[0])<[0.0005]).all():
                print("convergence occurred at epoch ",epoch)
                print("weights are: ", self.weight)
                break
            else:
                w = np.copy(self.weight)

    def train_(self, activation):
        err = 0.
        for input, target in zip(self.input, self.target):           
            net = np.dot(self.weight, input)
            if activation == 'sign':
                output = self.activationFunction(net)
                self.weight += self.learningRate*(target - net[0])*input
                err += 1./2*(target - net)**2
            elif activation == 'tanh':
                gamma = 4
                output = np.tanh(gamma*net)
                self.weight += self.learningRate*(1-output**2)*gamma*(target - output)*input
                err += 1.*(target - net)**2          
        return err

    def generateTestSet(self, mu, sigma):
        class1 = np.vstack((np.random.normal(mu[0][0], sigma[0][0], 1000),np.random.normal(mu[0][1], sigma[0][1], 1000),np.ones(1000), np.ones(1000))).T
        class2 = np.vstack((np.random.normal(mu[1][0], sigma[1][0], 1000),np.random.normal(mu[1][1], sigma[1][1], 1000),np.ones(1000), -np.ones(1000))).T
        self.test = np.concatenate((class1, class2), axis=0)

    def predectTestSet(self):
        wrongPrediction = 0
        for test, target in zip(self.test[:,0:np.shape(self.test)[1]-1], self.test[:,np.shape(self.test)[1]-1]):
            output = self.predect(test)
            if output != target:
                wrongPrediction += 1
        return 1.-(wrongPrediction*1.)/len(self.test)
    
    def predect(self, x):
        for x_ in x:
            net = np.dot(self.weight, x)
            return self.activationFunction(net)
    
    def plotTrainingError(self):
        self.plot(self.error, '#f92c50', 'Training Err', 'ADALINE TRAINING ERR PER EPOCH', 'EPOCH', 'ERR -> 1/2*RSS (Residual Sum of Squares)')
        
    def showPlot(self):
        plt.show()

    def networkValidation(self, itr, mu=[[2,0],[0,1]], sigma=[[0.5,0.2],[0.1,0.7]]):
        accuracy = []
        for iteration in range(itr):
            np.random.seed(iteration)
            model.generateTestSet(mu, sigma)
            accuracy.append(100*model.predectTestSet())
        self.plot(accuracy, '#f92c50', 'Average Acc', 'TEST SET AVERAGE ACCURACY', 'ITERATION', 'ACCURACY (%)',ylim=(0,105))
        print('Average Accuracy', 1.*sum(accuracy)/len(accuracy))

    def plot(self, data, color, label, title, xlabel, ylabel, xlim=None, ylim=None):
        plt.figure()
        plt.plot(data, color = color, label=label)
        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)


    def plotModel(self):
        wrongPrediction = 0
        predictedClass1 = []
        predictedClass2 = []
        for test, target in zip(self.test[:,0:np.shape(self.test)[1]-1], self.test[:,np.shape(self.test)[1]-1]):
            output = self.predect(test)
            if target == 1:
                predictedClass1.append(test)
            else:
                predictedClass2.append(test)
            
            if output != target:
                wrongPrediction += 1
        
        print('Wrong Prediction', wrongPrediction)
        
        x_ = np.linspace(0,5,100)-1
        fig, ax = plt.subplots() 
        ax.plot(x_, -(1.*self.weight[0][0]/self.weight[0][1])*x_ -(1.*self.weight[0][2]/self.weight[0][1]), "r", linewidth=2) 
        ax.scatter([x[0] for x in predictedClass1], [x[1] for x in predictedClass1], s=6, c="#f92c50", label="First Class") 
        ax.scatter([x[0] for x in predictedClass2], [x[1] for x in predictedClass2], s=6, c="#5db85d", label="Second Class") 
        ax.legend() 
        ax.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('CLASSIFIED TEST POINTS BY ADALINE MODEL')
        plt.ylim((-3,5))

if __name__ == "__main__" :
    np.random.seed(1)

    model = Adaline(0.01, 500)
    model.generateTrainingSet()
    model.train()
    model.plotTrainingError()
    model.networkValidation(100)
    model.plotModel()
    model.train('tanh')
    model.networkValidation(100)
    model.plotModel()
    model.showPlot()    
