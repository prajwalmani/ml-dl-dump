# demp of logistic regression from stratch
import numpy as np
import math
class logistic_regression:
    def __init__(self,lr=0.01,num_of_iterations=100):
        self.lr=lr
        self.num_of_iterations=num_of_iterations
        self.weights=None
        self.bias=None
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def fit(self,x,y):
        num_of_samples,num_of_features=x.shape
        self.weights=np.zeros(num_of_features)
        self.bias=0

        for _ in range(self.num_of_iterations):
            #y=mx+c
            linear=np.dot(x,self.weights)+self.bias
            y_pred=self.sigmoid(y_pred)
            dw=(-1/num_of_samples)*np.dot(x.T,(y-y_pred))
            db=(-1/num_of_samples)*np.sum(y-y_pred)

            self.weights= self.weights- self.lr*dw
            self.bias= self.bias- self.lr*db

    def predict(self,x_test):
        linear=np.dot(x_test,self.weights)+self.bias
        y_pred=self.sigmoid(linear)
        y_pred_class=[1 if i >0.5 else 0 for i in y_pred]
        return y_pred_class
