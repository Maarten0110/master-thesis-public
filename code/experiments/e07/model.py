from utils import free_memory
import torch
from math import sqrt
from typing import List, Union
from matplotlib import pyplot as plt
from math import factorial as fac, log, pi, sqrt
import seaborn as sns
sns.set_theme()

class MLPs_at_initialization_batched:

    def __init__(self,
        # architecture params:
        num_layers: int,
        layer_width: int,
        Cw: float,
        Cb: Union[float, None], # if none, biases are turned off
        activation: dict,
        
        # execution params:
        inputs: torch.Tensor, # expected to be 2 dimension. (dimension 0: batch, dimension 1: input)
        num_initializations_per_batch,
        ):

        assert (activation["name"] in ["linear", "relu", "relu-shifted", "tanh", "scale-invariant", "tanh-modified",
                                       "sin", "sin-modified", "polynomial-degree-4", "sigmoid",
                                       "sigmoid-shifted", "swish", "gelu", "softplus",
                                       "softplus-shifted"]), \
            f"Activation function \"{activation}\"not implemented!"

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.Cw = Cw
        self.Cb = Cb
        self.activation = activation
        self.num_initializations_per_batch = num_initializations_per_batch

        # Place input tensor on the GPU if possible. And prepare tensor for broadcasting during
        # matrix multiplication. (https://pytorch.org/docs/stable/notes/broadcasting.html)
        self.inputs = torch.unsqueeze(torch.unsqueeze(inputs, 0), -1)
        if torch.cuda.is_available():
            self.inputs = self.inputs.to("cuda")

        # Reserve memory for weight- and bias tensors (on the GPU if possible)
        self.list_of_weight_tensors: List[torch.Tensor] = []
        self.list_of_bias_tensors: List[torch.Tensor] = []
        for i in range(num_layers):
            num_features_in = inputs.size()[1] if i == 0 else layer_width
            num_features_out = layer_width
            # again, dimensions are in preparation of broadcasting during matrix multiplication
            weight_tensor = torch.zeros(
                (num_initializations_per_batch, 1, num_features_out, num_features_in),
                dtype=torch.float,
            )
            bias_tensor = torch.zeros(
                (num_initializations_per_batch, 1, num_features_out, 1),
                 dtype=torch.float,
            )

            if torch.cuda.is_available():
                weight_tensor = weight_tensor.to("cuda")
                bias_tensor = bias_tensor.to("cuda")

            self.list_of_weight_tensors.append(weight_tensor)
            self.list_of_bias_tensors.append(bias_tensor)

        # Reserve space to store the result of `execute_batch()`. The space will be reused
        # during each execution of `execute_batch()`.
        num_inputs = inputs.size()[0]
        self.preactivation_cache = torch.zeros(
            (num_initializations_per_batch, num_inputs, num_layers, layer_width),
            )
        if torch.cuda.is_available():
            self.preactivation_cache = self.preactivation_cache.to("cuda")

    def execute_batch(self) -> torch.Tensor:
        """
        Does the following:

            - Sample weights (and possibly biases) for a new batch
            - Execute the forward pass on the inputs for each initialization
            - Return the preactivations of the MLP in a tensor with these dimensions:

        `(num_initializations_per_batch, num_inputs, num_layers, layer_width)`
        """

        # sample the weights (in place)
        for i in range(self.num_layers):
            std = sqrt(self.Cw/self.layer_width)
            if i == 0:
                std = sqrt(1/self.layer_width)
            weight_tensor = self.list_of_weight_tensors[i]
            weight_tensor.normal_(0, std)
        
        # sample biases (in place) if Cb =/= None
        if self.Cb is not None:
            std = sqrt(self.Cb)
            for bias_tensor in self.list_of_bias_tensors:
                bias_tensor.normal_(0, std)

        # forward pass
        current_inputs = self.inputs
        for i in range(self.num_layers):
            weight_tensor = self.list_of_weight_tensors[i]
            bias_tensor = self.list_of_bias_tensors[i]
            
            # (num_initializations_per_batch, 1, num_features_out, num_features_in) x (1, num_inputs, num_features_in, 1)
            # result dims (broadcasting is used): (num_initializations_per_batch, num_inputs, num_features_out, 1)
            preactivations = torch.matmul(weight_tensor, current_inputs)

            # add bias if enabled
            if self.Cb is not None:
                preactivations = preactivations + bias_tensor

            # save preactivations to cache
            self.preactivation_cache[:, :, i, :] = torch.squeeze(preactivations, dim=3)

            # compute activations, which serve as the input for the next layers
            activations = self.compute_activations(preactivations)
            current_inputs = activations
        
        return self.preactivation_cache

    def compute_activations(self, preactivations: torch.Tensor) -> torch.Tensor:
        if self.activation["name"] == "linear":
            return preactivations
        if self.activation["name"] == "relu":
            return preactivations.clamp(min=0)
        if self.activation["name"] == "relu-shifted":
            a = self.activation["a"]
            return preactivations.clamp(min= -a)
        if self.activation["name"] == "scale-invariant":
            a_plus = self.activation["a_plus"]
            a_min = self.activation["a_min"]
            z_positive = preactivations.clamp(min = 0)
            z_negative = preactivations.clamp(max = 0)
            result = a_plus * z_positive + a_min * z_negative
            return result
        if self.activation["name"] == "tanh":
            return preactivations.tanh()
        if self.activation["name"] == "tanh-modified":
            b = self.activation["b"]
            return torch.tanh(b * preactivations)
        if self.activation["name"] == "sin":
            return preactivations.sin()
        if self.activation["name"] == "sin-modified":
            b = self.activation["b"]
            return torch.sin(b * preactivations)
        if self.activation["name"] == "polynomial-degree-4":
            c1 = self.activation["c1"]
            c2 = self.activation["c2"]
            c3 = self.activation["c3"]
            c4 = self.activation["c4"]
            z = preactivations
            result = c1 * z + c2 * (z**2) + c3 * (z**3) + c4 * (z**4)
            return result
        if self.activation["name"] == "sigmoid":
            return torch.sigmoid(preactivations)
        if self.activation["name"] == "sigmoid-shifted":
            return torch.sigmoid(preactivations) - 0.5
        if self.activation["name"] == "swish":
            z = preactivations
            return z * torch.sigmoid(z)
        if self.activation["name"] == "gelu":
            z = preactivations
            # approximation taken from: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
            return 0.5 * z * (1 + torch.tanh(sqrt(2/pi) * (z + 0.044715 * (z ** 3))))
        if self.activation["name"] == "softplus":
            return torch.log(1 + torch.exp(preactivations))
        if self.activation["name"] == "softplus-shifted":
            return torch.log(1 + torch.exp(preactivations)) - log(2)
        
        raise f"Activation unknown / not implemented: \"{self.activation}\""
    
    def dispose(self):
        """
        Deletes the resources used on the GPU. This instance becomes unusable afterwards!
        """
        to_delete = [self.preactivation_cache]
        for weight_tensor in self.list_of_weight_tensors:
            to_delete.append(weight_tensor)
        for bias_tensor in self.list_of_bias_tensors:
            to_delete.append(bias_tensor)
        free_memory(to_delete)


if __name__ == "__main__":
    print("Plots of activation functions to check their implementation!")
    bounds = 10
    input = torch.linspace(-bounds, bounds, 1001)

    # relu-shifted
    a = 0.5
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "relu-shifted", "a": a}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("shifted relu, $a=\\frac{1}{2}$")
    fig.savefig("report/images/activation_functions/shifted_relu.png")

    # scale-invariant
    a_plus = 1
    a_min = 0.2
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "scale-invariant", "a_plus": a_plus, "a_min": a_min}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title(f"scale invariant, $a_+={a_plus}$, $a_-={a_min}$")
    fig.savefig("report/images/activation_functions/scale_invariant.png")

    # tanh
    bs = [0.5, 1, 2]
    fig = plt.figure()
    for b in bs:
        model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "tanh-modified", "b": b}, torch.tensor([[1, 1],[1, 1]]), 10)
        y = model.compute_activations(input)
        sns.lineplot(x=input, y=y, label=f"$b={b}$")
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    plt.title("modified tanh, ($\\sigma(z) = \\tanh(bz)$)")
    fig.savefig("report/images/activation_functions/tanh_modified.png")

    # sin
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "sin"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title(f"$\\sigma(z) = \\sin(z)$")
    fig.savefig("report/images/activation_functions/sin.png")

    # sin-modified
    bs = [0.5, 1, 2]
    fig = plt.figure()
    for b in bs:
        model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "sin-modified", "b": b}, torch.tensor([[1, 1],[1, 1]]), 10)
        y = model.compute_activations(input)
        sns.lineplot(x=input, y=y, label=f"$b={b}$")
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    plt.title("modified sin, ($\\sigma(z) = \\sin(bz)$)")
    fig.savefig("report/images/activation_functions/sin_modified.png")

    # poly1 & poly2
    fig = plt.figure()

    eps = 0.01
    c1 = 1
    c2 = 1 / fac(2)
    c3 = -0.75 / fac(3)
    c4 = -(3/8 + 8/5 * eps) / fac(4)
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "polynomial-degree-4", "c1": c1, "c2": c2, "c3": c3, "c4": c4}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    
    sns.lineplot(x=input, y=y, label="poly1")
    eps_a = eps_b = 0.01
    c1 = 1
    c2 = sqrt(4*eps_a) / fac(2)
    c3 = -4 * eps_a / fac(3)
    c4 = - (eps_b + 12 * (eps_a ** 2)) / (sqrt(4*eps_a) * fac(4))
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "polynomial-degree-4", "c1": c1, "c2": c2, "c3": c3, "c4": c4}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)

    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y, label="poly2")
    plt.title("custom polynomial activation functions")
    plt.ylim([-10, 24])
    plt.xlabel("$z$")
    plt.ylabel("$\\sigma(z)$")
    fig.savefig("report/images/activation_functions/custom_polynomials.png")

    # sigmoid
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "sigmoid"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("$\\sigma(z) = 1/(1+e^{-z})$")
    fig.savefig("report/images/activation_functions/sigmoid.png")

    # sigmoid-shifted
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "sigmoid-shifted"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("shifted sigmoid: $\\sigma(z) = 1/(1+e^{-z}) - 1/2$")
    fig.savefig("report/images/activation_functions/sigmoid_shifted.png")

    # swish
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "swish"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("$\\sigma(z) = z/(1+e^{-z})$")
    fig.savefig("report/images/activation_functions/swish.png")

    # gelu
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "gelu"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("GELU")
    fig.savefig("report/images/activation_functions/gelu.png")

    # softplus
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "softplus"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("$\\sigma(z) = log(1+e^z)$")
    fig.savefig("report/images/activation_functions/softplus.png")

    # softplus-shifted
    model = MLPs_at_initialization_batched(1, 1, 1, None, {"name": "softplus-shifted"}, torch.tensor([[1, 1],[1, 1]]), 10)
    y = model.compute_activations(input)
    fig = plt.figure()
    plt.axline((-bounds, 0), (bounds, 0), color="black")
    plt.axline((0, 1.2*torch.min(y)), (0, 1.2*torch.max(y)), color="black")
    sns.lineplot(x=input, y=y)
    plt.title("$\\sigma(z) = log(1+e^z) - log(2)$")
    fig.savefig("report/images/activation_functions/softplus_shifted.png")
