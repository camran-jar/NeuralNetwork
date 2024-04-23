import pandas as pd
import numpy as np
import sys

class Perceptron:
    def __init__(self, num_inputs, rate=0.01):
        # initialise the weight and rate
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = rate

    # Define the linear layer
    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + self.weights[0]
        return Z
    
    def Heaviside_step_fn(self, z):
        return 1 if z >= 0 else 0
    
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
        loss = (prediction - target)
        return loss
    
    def train(self, inputs, target):
        for input_row, target_row in zip(inputs, target):
            prediction = self.predict(input_row)
            error = self.loss(prediction, target_row)
            self.weights[1:] += self.learning_rate * error * input_row
            self.weights[0] += self.learning_rate * error

    def fit(self, X, y, max_epochs=100):
        for epoch in range (max_epochs):
            converged = True
            for inputs, target in zip(X,y):
                prediction = self.predict(inputs)
                error = self.loss(prediction, target)
                if error !=0:
                    self.weights[1:] += self.learning_rate * error * inputs
                    self.weights[0] += self.learning_rate * error
                    converged = False
            if converged:
                print("Converged after {epoch+1} epochs")
                break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Prompt: python3 perceptron.py <ionosphere.data>")
        sys.exit(1)
        
    file = sys.argv[1]

    data = pd.read_csv(file, sep=' ')
    X = data.iloc[:, :-1].to_numpy()
    y = np.where(data['class'] == 'g', 1, 0)

    np.random.seed(23)

    perceptron = Perceptron(num_inputs=X.shape[1])

    perceptron.train(X, y)

    # Calculate accuracy
    pred = perceptron.predict(X)
    accuracy = np.mean(pred == y)
    print("Accuracy:", accuracy)

    # Print final weights
    print("Final Weights:", perceptron.weights)
