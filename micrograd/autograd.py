import math
from typing import Self, Union


class Value:
    """ Class that tracks the computational graph used to a value and allows for differentiation
    backwards through this graph.
    
    Methods:
        _backward: Calulate gradient of Value with respect to its children.
        backward: Call _backward on all children in topologically sorted order,
            then call _backward on self.
        tanh: Returns Value transformed by x -> tanh(x).
        relu: Returns Value transformed by x -> ReLU(x).
    """
    def __init__(self, data: Union[int, float], _children: tuple=(), _op: str='', label: str=''):
        """ Creates a Value with information needed to create and differentiate across
        the computational graph used to create it.

        Args:
            data (Union[int, float]): The numeric data contained by our Value.
            _children (tuple, optional): Children in computational graph. Defaults to ().
            _op (str, optional): String indicating the operation used to produce Value.
                Mainly used for visualization purposes. Defaults to ''.
            label (str, optional): A name to reference our Value. Defaults to ''.
        """
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
        # Initialize gradient and function for backward pass.
        self.grad = 0.
        self._backward = lambda: None
        
    def __repr__(self) -> str:
        """ Verbose string representation."""
        return f'Value(data={self.data})'
    
    def __add__(self, other: Union[Self, int, float]) -> Self:
        """ Returns sum as a Value with inherited children and correct backward function."""
        if not isinstance(other, (Value, int, float)):
            raise ValueError(f"Values can only perform addition with a Value, int, or float. Received {type(other)}")

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            """ Product rule for sums."""
            self.grad += output.grad
            other.grad += output.grad
        
        output._backward = _backward
        
        return output
    
    def __radd__(self, other: int) -> Self:
        """ Flips the order of addition when n + Value is called for some n."""
        return self + other
    
    def __neg__(self) -> Self:
        """ Negation of a Value."""
        return self * -1
    
    def __sub__(self, other: Union[Self, int, float]) -> Self:
        """ Subtract by adding the negation."""
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other: Union[Self, int, float]) -> Self:
        """ Returns multiplied Value with inherited children and correct backward function."""
        if not isinstance(other, (Value, int, float)):
            raise ValueError(f"Values can only perform multiplication with a Value, int, or float. Received {type(other)}")

        _input = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * _input.data, (self, _input), '*')
        
        def _backward():
            """ Product rule for n*x."""
            self.grad += _input.data * output.grad
            _input.grad += self.data * output.grad
        
        output._backward = _backward
        
        return output
    
    def __rmul__(self, other: int) -> Self:
        """ Flips the order of multiplication when n*Value is called for some n."""
        return self * other
    
    def __truedive__(self, other: Union[Self, int, float]) -> Self:
        """ Divide by multiplying the inverse."""
        return self * other**-1
    
    def __rtruediv__(self, other):
        """ Divides an int or float by Value."""
        return other * self**-1
    
    def __pow__(self, other: Union[int, float]) -> Self:
        """ Returns an exponentiated Value with inherited children and correct backward function."""
        assert isinstance(other, (int, float))
        output = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            """ Product rule for x^n."""
            self.grad += other * (self.data**(other-1)) * output.grad
            
        output._backward = _backward
        
        return output
    
    def exp(self):
        """ Returns a Value transformed by exponentiation with inherited children and correct backward function."""
        x = self.data
        output = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            """ Product rule for e^x."""
            self.grad = output.data * output.grad
            
        output._backward = _backward
    
    def tanh(self) -> Self:
        """ Returns a Value transformed by Tanh with inherited children and correct backward function."""
        x = self.data
        t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        output = Value(t, (self, ), 'tanh')
        
        def _backward():
            """ Product rule for tanh."""
            self.grad += output.grad * (1 - t**2)
            
        output._backward = _backward
        
        return output
    
    def relu(self) -> Self:
        """ Returns a Value transformed by ReLU with inherited children and correct backward function."""
        t = 0 if self.data < 0 else self.data
        output = Value(t, (self, ), 'ReLU')
        
        def _backward():
            """ Product rule for ReLU"""
            self.grad += (output.data > 0) * output.grad
            
        output._backward = _backward
            
        return output
    
    def backward(self):
        """ Constructs a topologically sorted list of graph of children in computational DAG, 
        then calls _backwards for each node in reverse order.
        
        A topologically sorted list of a DAG is a list where every vertex appears before any of its children,
        which ensures derivatives flow correctly.
        """
        graph = []
        visited = set()

        def topological_sort(vertex):    
            if vertex not in visited:
                visited.add(vertex)
                for child in vertex._prev:
                    topological_sort(child)
                graph.append(vertex)
                
            return graph, visited
                
        topological_sort(self)
        
        self.grad = 1.
        for node in reversed(graph):
            node._backward()
