import matplotlib.pyplot as plt
import numpy as np
import time

class Madaline:
    def __init__(self, learningRate = 0.005, epoch = 600, v=[0.5,0.5,0.5,0.5], b=-2):
        self.class1 = []
        self.class2 = []
        self.class3 = []
        self.input = []
        self.target = []
        self.test = []
        self.weight = []
        self.learningRate = learningRate
        self.epoch = epoch
        self.error = []
        self.v = v
        self.b = -2

    def activationFunction(self,x):
        return np.where(x >= 0, np.where(x < 0, x, 1), -1)

    def generateTrainSet(self,mu=[[3,0],[0,0],[1.5,0]], sigma=[[0.5,0.5],[0.5,0.5],[0.5]]):
        self.class1 = np.vstack((np.random.normal(mu[0][0], sigma[0][0], 100),np.random.normal(mu[0][1], sigma[0][1], 100),np.ones(100))).T
        self.class2 = np.vstack((np.random.normal(mu[1][0], sigma[1][0], 100),np.random.normal(mu[1][1], sigma[1][1], 100),np.ones(100))).T
        self.class3 = []
        
        angle = np.random.uniform(-np.pi, np.pi,size=250)    
        while len(self.class3) < 250:
            radius = np.random.normal(5, sigma[2][0])
            x = radius*np.cos(angle[len(self.class3)]) + mu[2][0]
            y = radius*np.sin(angle[len(self.class3)]) + mu[2][1]
            if x**2 + y**2 > 16 and x**2 + y**2 < 36:
                self.class3.append([x,y]+[1])
        self.target = np.concatenate((np.ones(100), -np.ones(350)), axis=0).T

    def plotTrainingSet(self):
        fig, ax = plt.subplots() 
        ax.scatter([x[0] for x in self.class1], [x[1] for x in self.class1], s=6, c="#f92c50", label="First Class") 
        ax.scatter([x[0] for x in self.class2], [x[1] for x in self.class2], s=6, c="#3ddcb0", label="Second Class")
        ax.scatter([x[0] for x in self.class3], [x[1] for x in self.class3], s=6, c="#aA45a0", label="Third Class")
        ax.legend()
        ax.axis('equal')
        ax.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('TRAINING SET POINTS')

    def train(self):
        self.target = np.vstack(([[1]]*100,[[-1]]*350))
        self.input = np.vstack([self.class1, self.class2, self.class3])
        self.weight=2*np.random.random((4,3)) - 1
        self.weight = self.weight.reshape([4,3])  
        c=0
        err = 0
        while True:
            self.weight = np.random.uniform(-0.4,0.4,12).reshape([4,3])
            for epoch in range(self.epoch):
                c=c+1
                err = 0
                for input, target in zip(self.input,self.target):
                    net=np.dot(self.weight,input)
                    z=self.activationFunction(net)
                    outNet=z[0]*self.v[0]+z[1]*self.v[1]+z[2]*self.v[2]+z[3]*self.v[3]+self.b
                    output=self.activationFunction(outNet)
                    if output!=target:
                        err = err + 1
                        if output==+1:
                            if z[0]==+1:
                                self.weight[0] += self.learningRate*(-1-net[0])*input
                            if z[1]==+1:
                                self.weight[1] += self.learningRate*(-1-net[1])*input
                            if z[2]==+1:
                                self.weight[2] += self.learningRate*(-1-net[2])*input
                            if z[3]==+1:
                                self.weight[3] += self.learningRate*(-1-net[3])*input
                        elif output==-1:
                            arg=np.argmin(abs(net))
                            self.weight[arg,:]=self.weight[arg,:]+self.learningRate*(1-net[arg])*input
                print(c)
                if err ==0:
                    break
            if err ==0:
                break

    def plotModel(self):
        plt.figure() 
        x = np.linspace(-6,6,100)
        y0 = -(self.weight[0][0]/self.weight[0][1])*x -(self.weight[0][2]/self.weight[0][1])
        y1 = -(self.weight[1][0]/self.weight[1][1])*x -(self.weight[1][2]/self.weight[1][1])
        y2 = -(self.weight[2][0]/self.weight[2][1])*x -(self.weight[2][2]/self.weight[2][1])
        y3 = -(self.weight[3][0]/self.weight[3][1])*x -(self.weight[3][2]/self.weight[3][1])
        plt.plot(x, y0, 'r')
        plt.plot(x, y1, 'b')
        plt.plot(x, y2, 'g')
        plt.plot(x, y3, 'y')

        plt.scatter([x[0] for x in self.class1], [x[1] for x in self.class1], s=6, c="#f92c50", label="First Class") 
        plt.scatter([x[0] for x in self.class2], [x[1] for x in self.class2], s=6, c="#3ddcb0", label="Second Class")
        plt.scatter([x[0] for x in self.class3], [x[1] for x in self.class3], s=6, c="#aA45a0", label="Third Class")
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim((-7,7))
        plt.ylim((-7,7))
        plt.title('TRAINING SET POINTS')

    def testModel(self):
        err = 0
        for input, target in zip(self.input,self.target):
            net=np.dot(self.weight,input)
            z=self.activationFunction(net)
            outNet=z[0]*self.v[0]+z[1]*self.v[1]+z[2]*self.v[2]+z[3]*self.v[3]+self.b
            output=self.activationFunction(outNet)
            if output!=target:
                err = err + 1
        print(err)

class Madaline2:
    def __init__(self, learningRate = 0.005, epoch = 600, v=[0.5,0.5,0.5,0.5], b=-2):
        self.class1 = []
        self.class2 = []
        self.class3 = []
        self.input = []
        self.target = []
        self.test = []
        self.weight = []
        self.learningRate = learningRate
        self.epoch = epoch
        self.error = []
        self.v = v
        self.b = -2

    def activationFunction(self,x):
        return np.where(x >= 0, np.where(x < 0, x, 1), -1)

    def generateTrainSet(self,mu=[[3,0],[0,0],[1.5,0]], sigma=[[0.5,0.5],[0.5,0.5],[0.5]]):
        self.class1 = np.vstack((np.random.normal(mu[0][0], sigma[0][0], 100),np.random.normal(mu[0][1], sigma[0][1], 100),np.ones(100))).T
        self.class2 = np.vstack((np.random.normal(mu[1][0], sigma[1][0], 100),np.random.normal(mu[1][1], sigma[1][1], 100),np.ones(100))).T
        self.class3 = []
        
        angle = np.random.uniform(-np.pi, np.pi,size=250)    
        while len(self.class3) < 250:
            radius = np.random.normal(5, sigma[2][0])
            x = radius*np.cos(angle[len(self.class3)]) + mu[2][0]
            y = radius*np.sin(angle[len(self.class3)]) + mu[2][1]
            if x**2 + y**2 > 16 and x**2 + y**2 < 36:
                self.class3.append([x,y]+[1])
        self.target = np.concatenate((np.ones(100), -np.ones(350)), axis=0).T

    def plotTrainingSet(self):
        fig, ax = plt.subplots() 
        ax.scatter([x[0] for x in self.class1], [x[1] for x in self.class1], s=6, c="#f92c50", label="First Class") 
        ax.scatter([x[0] for x in self.class2], [x[1] for x in self.class2], s=6, c="#3ddcb0", label="Second Class")
        ax.scatter([x[0] for x in self.class3], [x[1] for x in self.class3], s=6, c="#aA45a0", label="Third Class")
        ax.legend()
        ax.axis('equal')
        ax.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('TRAINING SET POINTS')

    def train(self):
        self.target = np.vstack(([[1]]*100,[[-1]]*350))
        self.input = np.vstack([self.class2, self.class1, self.class3])
        self.weight=2*np.random.random((4,3)) - 1
        self.weight = self.weight.reshape([4,3])  
        c=0
        err = 0
        while True:
            self.weight = np.random.uniform(-0.4,0.4,12).reshape([4,3])
            for epoch in range(self.epoch):
                c=c+1
                err = 0
                for input, target in zip(self.input,self.target):
                    net=np.dot(self.weight,input)
                    z=self.activationFunction(net)
                    outNet=z[0]*self.v[0]+z[1]*self.v[1]+z[2]*self.v[2]+z[3]*self.v[3]+self.b
                    output=self.activationFunction(outNet)
                    if output!=target:
                        err = err + 1
                        if output==+1:
                            if z[0]==+1:
                                self.weight[0] += self.learningRate*(-1-net[0])*input
                            if z[1]==+1:
                                self.weight[1] += self.learningRate*(-1-net[1])*input
                            if z[2]==+1:
                                self.weight[2] += self.learningRate*(-1-net[2])*input
                            if z[3]==+1:
                                self.weight[3] += self.learningRate*(-1-net[3])*input
                        elif output==-1:
                            arg=np.argmin(abs(net))
                            self.weight[arg,:]=self.weight[arg,:]+self.learningRate*(1-net[arg])*input
                print(c)
                if err ==0:
                    break
            if err ==0:
                break

    def plotModel(self):
        x = np.linspace(-6,6,100)
        y0 = -(self.weight[0][0]/self.weight[0][1])*x -(self.weight[0][2]/self.weight[0][1])
        y1 = -(self.weight[1][0]/self.weight[1][1])*x -(self.weight[1][2]/self.weight[1][1])
        y2 = -(self.weight[2][0]/self.weight[2][1])*x -(self.weight[2][2]/self.weight[2][1])
        y3 = -(self.weight[3][0]/self.weight[3][1])*x -(self.weight[3][2]/self.weight[3][1])
        plt.plot(x, y0, 'r')
        plt.plot(x, y1, 'b')
        plt.plot(x, y2, 'g')
        plt.plot(x, y3, 'y')

        plt.axis('equal')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim((-7,7))
        plt.ylim((-7,7))
        plt.title('TRAINING SET POINTS')

    def testModel(self):
        err = 0
        for input, target in zip(self.input,self.target):
            net=np.dot(self.weight,input)
            z=self.activationFunction(net)
            outNet=z[0]*self.v[0]+z[1]*self.v[1]+z[2]*self.v[2]+z[3]*self.v[3]+self.b
            output=self.activationFunction(outNet)
            if output!=target:
                err = err + 1
        print(err)



if __name__ == "__main__" :
    tic = time.clock()
    np.random.seed(11)
    model = Madaline()
    model.generateTrainSet()
    #model.plotTrainingSet()
    model.train()
    model.plotModel()
    #model.testModel()

    np.random.seed(11)
    model2 = Madaline2()
    model2.generateTrainSet()
    model2.train()
    model2.plotModel()
    toc = time.clock()
    print(toc - tic)
    plt.show()