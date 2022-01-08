#!/usr/bin/env python3
"""
   Name    : nn.py
   Author  : Ian Gomez
   Date    : November 22, 2021
   Description : Neural Network implementation using JAX to test out a new framework.
   Github  : imgomez0127@github
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn
from jax.nn import initializers
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.manifold

key = random.PRNGKey(69)


def preprocess_data(data):
    data = np.array(data)
    flower_types = set(data[:, -1].reshape(-1))
    classes = {flower_type: n for n, flower_type in enumerate(flower_types)}
    data[:, -1] = np.array(list(map(lambda x: classes[x], data[:, -1].reshape(-1))))
    return data.astype(float)


def accuracy(predictions, true_labels):
    return np.sum(predictions == true_labels)/len(predictions)


def plot_results(iris_data, my_preds, true_preds):
    TSNE = sklearn.manifold.TSNE(init='pca', learning_rate='auto')
    embedding = TSNE.fit_transform(iris_data)
    colors = ['r', 'b', 'g']
    fig, axs = plt.subplots(2)
    fig.suptitle('Classifications')
    for i, color in enumerate(colors):
        cluster = my_preds == i
        axs[0].scatter(
            embedding[cluster, 0], embedding[cluster, 1], color=color)
    for i, color in enumerate(colors):
        cluster = true_preds == i
        axs[1].scatter(
            embedding[cluster, 0], embedding[cluster, 1], color=color)
    plt.show()


@jax.jit
def forward(layers, data):
    z = data
    for layer in layers[:-1]:
        a = jnp.dot(z, layer)
        z = jax.nn.relu(a)
        z = jnp.hstack((z, jnp.ones((z.shape[0], 1))))
    return jax.nn.softmax(jnp.dot(z, layers[-1]))


def loss(layers, data, targets):
    preds = forward(layers, data)
    return -jnp.mean(targets*jnp.log(preds))


class NeuralNetwork:

    def __init__(self, input_shape, hidden_shape, hidden_layers, output_shape):
        keys = random.split(key, 2+hidden_layers)
        glorot_uniform = initializers.glorot_uniform()
        self.layers = [glorot_uniform(keys[0], (input_shape, hidden_shape[1]))]
        self.layers.extend([glorot_uniform(key, hidden_shape)
                            for key in keys[1:-1]])
        self.layers.append(
            glorot_uniform(keys[-1], (hidden_shape[0], output_shape)))
        self.grad_function = jax.jit(jax.value_and_grad(loss))

    def update(self, data, targets, learning_rate):
        loss_val, grads = self.grad_function(self.layers, data, targets)
        for i, (grad, layer) in enumerate(zip(grads, self.layers)):
            self.layers[i] = layer - learning_rate * grad
        return loss_val

    def train(self, epochs, data, targets, learning_rate=0.1):
        for i in range(epochs):
            loss_val = self.update(data, targets, learning_rate)
            print(f'\rIteration {i} loss: {loss_val:.2f}', end='')
        print(f'\rIteration {i} loss: {loss_val:.2f}')

    def predict(self, data):
        return jnp.argmax(forward(self.layers, data), axis=1)

    def __call__(self, data):
        return self.predict(data)


def main():
    dataset = preprocess_data(pd.read_csv("iris.data", header=None))
    dataset = jnp.array(dataset)
    input_shape = dataset.shape[1]-1
    hidden_layer_width = 10
    hidden_layer_shape = (hidden_layer_width, hidden_layer_width-1)
    output_shape = 3
    nn = NeuralNetwork(input_shape, hidden_layer_shape, 3, output_shape)
    targets = jax.nn.one_hot(dataset[:, -1], 3)
    data = dataset[:, :-1]
    nn.train(1000, data, targets)
    plot_results(data, nn(data), jnp.argmax(targets, axis=1))
    print(accuracy(dataset[:, -1], nn(data)))


if __name__ == "__main__":
    main()
