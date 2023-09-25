import matplotlib.pyplot as plt
import numpy as np

class McCullochPitts:
    def __init__(self, w, v):
        self.w = w
        self.v = v

    def plotPolygon(self):
        x = [np.linspace(0,3,100), np.linspace(3,5,100), np.linspace(-1,5,100), np.linspace(-1,0,100)]  
        for weight, x_ in zip(self.w, x):
            if weight[1]==0:
                plt.scatter(x_,-(weight[2]/weight[0]),s=2, color = '#18b999')
            else:
                plt.scatter(x_,-(1.*weight[0]/weight[1])*x_ -(1.*weight[2]/weight[1]),s=2, color = '#18b999')
        plt.axis('equal')

    def activationFunction(self,x):
        return np.where(x >= 0, np.where(x < 0, x, 1), 0)

    def predect(self, x):
        x = np.append(x,[1])
        net = np.dot(self.w,x)
        out = np.append(self.activationFunction(net),[1])
        output = out.dot(self.v)
        return self.activationFunction(output)

    def predectTestSet(self, x = 11*np.random.random((1000,2)) - 4):
        for x_ in x:
            if model.predect(x_) == 1:
                plt.scatter(x_[0],x_[1],s=3, color = '#f92c50')
            else:
                plt.scatter(x_[0],x_[1],s=3, color = '#5db85d')

    def plot(self):
        plt.title("McCulloch-Pitts")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

if __name__ == "__main__":
    w = [[0, -1, 3], [-5, -2, 21], [0, 1, 2], [5, -1 ,3]]
    v = [0.25, 0.25, 0.25, 0.25, -1]        
    model = McCullochPitts(w,v)
    model.plotPolygon()
    model.predectTestSet()
    model.plot()