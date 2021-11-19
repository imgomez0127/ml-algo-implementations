"""
    Computes Fischer's Linear Discriminant for binary classification
    of a dataset
    @author Ian Gomez imgomez0127@github
"""
import numpy as np
from numpy.linalg import pinv, norm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def fischers_linear_discriminant(X, y):
    classes = [0, 1]
    n = len(X)
    classified_samples = [X[(y == category).flatten()] for category in classes]
    averages = [np.sum(samples, axis=0)/n for samples in classified_samples]
    covariances = []
    for average,samples in zip(averages, classified_samples):
        avg = np.repeat(average[np.newaxis], len(samples), axis=0)
        covariances.append((samples.T-avg.T)@(samples-avg))
    S = sum(covariances)
    classifier = pinv(S)@(averages[0][np.newaxis].T-averages[1][np.newaxis].T)
    classifier /= norm(classifier)
    return classifier

def print_accuracy(X,t,theta):
    print("Theta values:")
    print(theta)
    predicted_classes = (X@(theta)< 0.21).astype(float)
    print("Accuracy:")
    print((sum(predicted_classes == t)/len(t)))

def plot(X, y, theta):
    negative = np.asarray((y == 0).flatten())
    positive = np.asarray((y == 1).flatten())
    plt.scatter(list(X[negative,1]),list(X[negative,2]),color="r",label="first")
    plt.scatter(list(X[positive,1]),list(X[positive,2]),color="b",label="second")
    plt.ylim(1,5)
    plt.xlim(4,8)
    theta = np.asarray(theta)[:,0]
    x = np.linspace(0,10,50)
    omega = (theta[0] + theta[1]*x)/theta[2]
    plt.plot(x, omega)
    plt.show()

if __name__ == "__main__":
    IRIS_DATASET = pd.read_csv("iris.data", header=None)
    IRIS_DATASET = IRIS_DATASET.drop(columns=[2, 3])
    IRIS_DATASET = np.asarray(IRIS_DATASET)
    IRIS_DATASET[:, -1] = (IRIS_DATASET[:, -1] == "Iris-virginica").astype(float)
    X = IRIS_DATASET[:, 0:2]
    X = np.hstack((np.ones((len(X), 1)), X))
    X = X.astype(float)
    y = (IRIS_DATASET[:, -1][np.newaxis].T).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
    classifier = fischers_linear_discriminant(X_train, y_train)
    print_accuracy(X,y,classifier)
    plot(X,y,classifier)
