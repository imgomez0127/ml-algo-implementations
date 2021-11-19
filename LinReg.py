import time
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow import reduce_sum,shape
def lin_reg(X,y,theta = [],epoch=1000,alpha=.1,L=0):
    #alpha is the learning rate and u need to change this based on what ur analyzing
    # the main tradeoff is that if u have it too low it will take 4ever to converge and if it is too high
    # it can fail to converge and u will get a bunch of infinte numbers
    # epoch is the amount of training ur algorithm will do and there is something to be said about overtraining dont know how to determine this yet
    # newThetaJ = thetaJ - alpha * d/d(thetaJ) cost
    # cost = 1/m * Sum((mx+b - y)^2)
    # d/d theta0 = 2/m * Sum(mx+b-y)
    # d/d theta 1 = 2/m * Sum(x *(mx+b-y))
    # Assumes that you have already added the bias term and the X is a MxN matrix
    # And that y is a Mx1 matrix
    if theta == []:
        theta = np.zeros(X.shape[1]).reshape(X.shape[1],1)
    m = X.shape[0]
    for i in range(0,epoch):
        regularizaiton_theta = np.copy(theta)
        regularizaiton_theta[0] = 0
        gradient = X.T.dot(((X.dot(theta))-y))
        theta -= alpha * (1/m) *  gradient + ((L/m)*regularizaiton_theta)
    return theta
def calculate_price(Theta,x):
    return Theta.T.dot(x[np.newaxis].T)
def costFunc(X,y,theta,L=0):
    therta = np.copy(theta)
    therta[0] = 0
    return (1/X.shape[0]) * sum((X.dot(theta) - y[np.newaxis].T)**2) + (L/X.shape[0]) * sum(therta * therta)
def stdErr(X,y,theta):
    return np.sqrt((1/X.shape[0]) * sum( ((y[np.newaxis].T - X.dot(theta)) ** 2) ))
def stdErrTF(predictions,labels,weights=None,name=None):
    return np.sqrt((1/shape(predictions)) * reduce_sum( ((labels - predictions) ** 2) ))
if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.rand(100)
    result_data = np.copy(data).reshape(100,1)
    X = [[1,x] for x in data]
    X = np.asarray(X)
    y = 2 + 3 * data 
    weights = lin_reg(X,y.reshape(100,1),epoch=1000)
    print(weights) 
