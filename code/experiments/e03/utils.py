import torch
import sys
import os

def flatten_without_diagonal(square_tensor):
    assert square_tensor.shape[0] == square_tensor.shape[1]
    assert square_tensor.ndim == 2
    
    n = square_tensor.shape[0]
    flattened = square_tensor.flatten()
    concat_list = []
    for i in range(n-1):
        lower_bound = 1 + i * (n+1)
        upper_bound = (i+1) * (n+1)
        concat_list.append(flattened[lower_bound:upper_bound])

    result = torch.cat(concat_list)
    return result
