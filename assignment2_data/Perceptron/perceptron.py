# https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, rate=0.01):
        # initialise the weight and rate
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = rate

    #define the linear layer
    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + + self.weights[0]
        return Z
    
    def Heaviside_step_fn(self, z):
        if z >= 0:
            return 1
        else:
            return 0
    
    def predict(self, inputs):
        Z = self.linear(inputs)
        try: 
            pred = []
            for z in Z:
                pred.append(self.Heaviside_step_fn(z))
        except:
            return self.Heaviside_step_fn(Z)
        return pred

    def loss(self, prediction, target):
        loss = (prediction-target)
        return loss
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for inputs, target in zip(X,y):
                self.train(inputs, target)

if __name__ == "__main__":
    data = pd.read_csv("ionosphere.data", sep=' ')
    X = data.iloc[:, :-1].to_numpy()
    y = np.where(data['class'] == 'g', 1, 0)

    np.random.seed(23)

    perceptron = Perceptron(num_inputs=X.shape[1])

    perceptron.fit(X,y, num_epochs=100)

    pred = perceptron.predict(X)

    accuracy = np.mean(pred != y)
    print("accuracy :", accuracy)

    print("pred", pred)
    print(y)
    print(X)
