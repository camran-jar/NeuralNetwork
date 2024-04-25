
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, rate=0.01):
        # initialise the weights randomly and set the learnign rate
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = rate

    # Define the linear layer, calculate linear combinations of input and weights
    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + self.weights[0]
        return Z
    
    # Function that returns 1 if z >=0, else return 0
    def heaviside_step_fn(self, z):
        return np.where(z >= 0, 1, 0)

    # predict function, takes the linear function, applies the step function
    def predict(self, inputs):
            Z = self.linear(inputs)
            return self.heaviside_step_fn(Z)

    # Function to calculate the difference between target and prediction
    def loss(self, prediction, target):
        return target - prediction
    
    # function to train the dataset
    def fit(self, X, y, max_epochs=100):
        for epoch in range(max_epochs):
            converged = True
            misclassified = 0
            for input_row, target_row in zip(X, y):
                pred = self.predict(input_row)
                err = self.loss(pred, target_row)
                input_row = input_row.astype(float)
                self.weights[1:] += self.learning_rate * err * input_row
                self.weights[0] += self.learning_rate * err
                misclassified += np.abs(err)
                if err != 0:
                    converged = False
            if converged:
                print(f"Converged after {epoch + 1} epochs.")
                break
            elif epoch == max_epochs - 1:
                print(f"Did not converge after {max_epochs} epochs. {misclassified} instances still misclassified")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Prompt: python3 perceptron.py <ionosphere.data>")
        sys.exit(1)
        
    file = sys.argv[1]

    data = pd.read_csv(file, sep=' ')

    # split data into train and test, 80% of the data for training, 20% for testing
    train = data.sample(frac=0.8)
    test = data.drop(train.index)
    train = train.to_numpy()
    test = test.to_numpy()

    # set the split data to test and Train sets
    trainX, trainY = train[:, :-1], train[:, -1]
    testX, testY = test[:,:-1], test[:,-1]
    trainY = np.where(trainY == 'g', 1,0)
    testY = np.where(testY == 'g', 1,0)

    np.random.seed(29)

    perceptron = Perceptron(len(trainX[1]))

    perceptron.fit(trainX, trainY)

    # Trianing model evaluation
    pred_train = perceptron.predict(trainX)
    accuracy_train = np.mean(pred_train == trainY)
    print("Train Accuracy:", accuracy_train)

    # Test model evaluation
    pred_test = perceptron.predict(testX)
    accuracy_test = np.mean(pred_test == testY)
    print("Test Accuracy:", accuracy_test)

    # Print final weights
    print("Final Weights:", perceptron.weights)

   

    