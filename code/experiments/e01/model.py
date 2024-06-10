import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from math import sqrt

class LinearNetwork(nn.Module):

    def __init__(self, num_layers: int, layer_width: int, Cw: float):
        super().__init__()
        self.activations = torch.zeros((num_layers, layer_width))
        ordered_dict = OrderedDict()

        def hook(_, __, y, layer_index: int):
            self.activations[layer_index, :] = y
        
        for i in range(num_layers):
            layer = nn.Linear(layer_width, layer_width, bias=False)

            partial_hook = partial(hook, layer_index=i)

            # Register hook to capture activations of each layer:
            layer.register_forward_hook(partial_hook)

            # Initialize weights from a normal distribution:
            # nn.init.ones_(layer.weight) # TODO remove
            nn.init.normal_(layer.weight, mean=0, std=sqrt(Cw/layer_width))
            ordered_dict[f"linear_{i}"] = layer
        
        self.stack_of_linear_layers = nn.Sequential(ordered_dict)            

    def forward(self, x):
        return self.stack_of_linear_layers(x)

    def dispose(self):
        del self.activations, self.stack_of_linear_layers, self
