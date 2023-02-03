from abc import ABC, abstractmethod
import random
from typing import Literal

from autograd import Value


class Neuron:
    """ Implementation of a single artificial neuron.
    """
    def __init__(self, input_dim: int, activation: Literal[None, 'relu', 'tanh']='relu'):
        """ Single aritifical neuron.

        Args:
            input_dim (int): Dimension of inputs to neuron.
            activation (Literal[None, 'relu', 'tanh'], optional): Non-linearity to apply. Defaults to 'relu'.
        """
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
        self.bias = Value(random.uniform(-1, 1))
        self.activation = activation
        
    def __call__(self, x: list[float, Value]):
        """ Forward pass."""
        linear_output = sum((w*xi for w, xi in zip(self.weights, x))) + self.bias
        if self.activation == 'relu':
            output = linear_output.relu()
            
        elif self.activation == 'tanh':
            output = linear_output.tanh()
            
        else:
            output = linear_output
        
        return output
    
    def parameters(self):
        """ Return list of neuron's parameters."""
        return self.weights + [self.bias]


class Layer:
    """ Implementation of a fully connected layer."""
    def __init__(self, input_dim: int, output_dim: int):
        """ Create a single fully connected layer.

        Args:
            input_dim (int): Dimension of inputs.
            output_dim (int): Dimension of outputs.
        """
        self.neurons = [Neuron(input_dim) for _ in range(output_dim)]
        
    def __call__(self, x:list[float, Value]):
        """ Forward pass."""
        output_list = [n(x) for n in self.neurons]
        output = output_list[0] if len(output_list) == 1 else output_list

        return output
    
    def parameters(self):
        """ Return list of parameters in layer."""
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class MLP:
    """ Implementation of a multilayer perceptron."""
    def __init__(self, input_dim: int, layer_widths: list[int]):
        """ Construct MLP with given widths for each layer.

        Args:
            input_dim (int): Dimension of each input sample.
            layer_widths (list[int]): The desired width of hidden and output layers.
        """
        layer_sizes = [input_dim] + layer_widths
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_widths))]
        
    def __call__(self, x: list[float, Value]):
        """ Forward pass."""
        for layer in self.layers:
            x = layer(x)

        return x
    
    def parameters(self):
        """ Return list of model parameters."""
        params = []
        for l in self.layers:
            params.extend(l.parameters())
            
        return params
    
    def zero_grad(self):
        """ Reset gradients to zero to avoid accumulation."""
        for p in self.parameters():
            p.grad = 0.

    def fit(self, X, y, loss_fn, learning_rate: float, n_iters: int):
        """ Perform simple training loop.
        
        Args:
            X: List of training features.
            y: List of labels.
            loss_fn: Loss function for optimizing network.
            learning rate: Learning rate for estimating derivatives.
            n_inters: Number of training iterations to perform.

        """
        for _ in range(n_iters):
            y_pred = [self.__call__(x) for x in X]
            loss = loss_fn(y, y_pred)
            
            self.zero_grad()
            loss.backward()

            for p in self.parameters():
                p.data += -learning_rate * p.grad