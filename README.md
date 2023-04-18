# micrograd
Creating an Autograd engine and small deep learning library from scratch with the help of Andrej Karpathy and his walkthrough YouTube video [here](https://www.youtube.com/watch?v=VMj-3S1tku0). 
Inspired by PyTorch, we recreate autograd's Value class (now deprecated in PyTorch) which allows for building and differentiating on dynamically created DAGs. We also create a small deep learning module inspired by PyTorch's nn Module, which allows for building and optimizing basic neural networks just like in PyTorch (with a fit method thrown in for the Tensorflow fans). This is an educational project practicing my fundamentals, and there's nobody better than Andrej to learn from. See the original package [here](https://github.com/karpathy/micrograd), or check out my implementation in this repo.

### Creating and Training an MLP:
```python
from micrograd.nn import MLP


# Sample data.
x = [
    [2., 3., -1.],
    [3., -1., 0.5]
]
y = [1., -1.]

# Define network and loss function.
input_dim = 3 # Dimension of input.
hidden_layer_dims = [4, 4, 1] # Width of hidden and output layers.
net = MLP(input_dim, hidden_layer_dims)

def rmse(y_true, y_pred):
    return sum((y_obs - y_hat)**2 for y_hat, y_obs in zip(y_pred, y_true))

net.fit(x, y, loss_fn=rmse, learning_rate=0.01, n_iters=5)
```
