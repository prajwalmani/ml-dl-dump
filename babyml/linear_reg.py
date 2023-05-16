# demp of linear regression from stratch
import numpy as np
class linear_regression:
    def __init__(self,lr=0.01,num_of_iterations=100):
        self.lr=lr
        self.num_of_iterations=num_of_iterations
        self.weights=None
        self.bias=None
        
    def fit(self,x,y):
        num_of_samples,num_of_features=x.shape
        self.weights=np.zeros(num_of_features)
        self.bias=0

        for _ in range(self.num_of_iterations):
            #y=mx+c
            y_pred=np.dot(x,self.weights)+self.bias
            dw=(2/num_of_samples)*np.dot(x.T,(y_pred-y))
            db=(2/num_of_samples)*np.sum(y_pred-y)

            self.weights= self.weights- self.lr*dw
            self.bias= self.bias- self.lr*db

    def predict(self,x_test):
        y_pred=np.dot(x_test,self.weights)+self.bias
        return y_pred
