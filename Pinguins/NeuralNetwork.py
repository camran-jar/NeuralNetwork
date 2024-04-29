import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []

        for i in range(self.num_hidden):
            # TODO: Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for j in range(len(inputs)):
                weighted_sum += inputs[j] * self.hidden_layer_weights[j, i]
            output = self.sigmoid(weighted_sum)        
            hidden_layer_outputs.append(output)

        output_layer_outputs = []

        for i in range(self.num_outputs):
            # TODO: Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for i in range(len(hidden_layer_outputs)):
                for j in range(len(self.output_layer_weights)):
                    weighted_sum += hidden_layer_outputs[j] * self.output_layer_weights[j][i]
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        # TODO! Calculate output layer betas.
        for i in range(self.num_outputs):
            output_layer_betas[i] = (desired_outputs[0] - output_layer_outputs[i]) * output_layer_outputs[i] * (1 - output_layer_outputs[i])
        print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        # TODO! Calculate hidden layer betas.
        for i in range(self.num_hidden):
            error = 0
            for j in range(self.num_outputs):
                error += self.output_layer_weights[i][j] * output_layer_betas[j]
            hidden_layer_betas[i] = error * hidden_layer_outputs[i] * (1 - hidden_layer_outputs[i])
        print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * output_layer_betas[j]

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                    delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * hidden_layer_betas[j]

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
            # TODO! Update the weights.
            self.output_layer_weights += delta_output_layer_weights

            self.hidden_layer_weights += delta_hidden_layer_weights
            print('Updated output layer weights', delta_output_layer_weights)
            print("updated hidden layer weights", delta_hidden_layer_weights)

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            correct_pred = 0
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = np.argmax(output_layer_outputs)  # TODO!
                predictions.append(predicted_class)

                if predicted_class == np.argmax(desired_outputs[i]):
                    correct_pred += 1

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch
            acc = correct_pred / len(instances)
            print('Accuracy for epoch {}: {:.2f}%'.format(epoch, acc))

        # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    def predict(self, instances):
        predictions = []

        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            #print(instances)
            #print("outputlayer ",output_layer_outputs)
            round = np.round(output_layer_outputs)
            print("rounded", round)
            if np.array_equal(round, [1,0,0]):
                predicted_class = 1
            elif np.array_equal(round, [0,1,0]):
                predicted_class = 2
            elif np.array_equal(round, [0,0,1]):
                predicted_class = 3
            else:
                print("Unable to make prediction")
                predicted_class = None
            predictions.append(predicted_class)
        return predictions
