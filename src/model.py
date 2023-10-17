"""Deep learning model for PINNS"""

import torch.nn as nn
import torch


# class to build the network
class BuildNetwork(nn.Module):
    def __init__(self, input_size, h_sizes, output_size, n_heads, activation="tanh"):
        super(BuildNetwork, self).__init__()
        # store the number of "heads" to use in the model
        self.n_heads = n_heads

        # build activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "cos":
            self.activation = torch.cos()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError("Invalid activation function. Use 'tanh'.")

        # build the layers to use for the forward pass
        self.hidden_layers = nn.ModuleList([nn.Linear(in_features, out_features)
                                           for in_features, out_features in zip([input_size] + h_sizes[:-1], h_sizes)])

        # build n_heads output layers, each corresponding to different conditions during training
        self.multi_head_output = nn.ModuleList([nn.Linear(h_sizes[-1], output_size)])
        self.multi_head_output.extend([nn.Linear(h_sizes[-1], output_size) for i in range(n_heads - 1)])

    def forward(self, x):
        # dictionary to store the output for each "head" in the model
        u_results = {}

        # all "heads" have the same pass through the hidden laers
        result = x
        for layer in self.hidden_layers:
            result = self.activation(layer(result))
        h = result

        # apply the corresponding output layer to each "head"
        for i in range(self.n_heads):
            result_i = self.multi_head_output[i](h)
            u_results[f"head {i + 1}"] = result_i

        return u_results, h


# class to build the network 
class BuildNetwork_previous(nn.Module):
    def __init__(self, input_size, h_size1, h_size2, h_size3, output_size, n_heads):
        super(BuildNetwork_previous, self).__init__()
        # store the number of "heads" to use in the model
        self.n_heads = n_heads

        # build the layers to use for the forward pass
        self.l1 = nn.Linear(input_size, h_size1)
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(h_size1, h_size2)
        self.l3 = nn.Linear(h_size2, h_size3)

        # build n_heads output layers, each corresponding to different conditions during training
        self.multi_head_output = nn.ModuleList([nn.Linear(h_size3, output_size)])
        self.multi_head_output.extend([nn.Linear(h_size3, output_size) for i in range(n_heads - 1)])

    def forward(self, x):
        # dictionary to store the output for each "head" in the model
        u_results = {}

        # all "heads" have the same pass through the hidden laers
        result = self.l1(x)
        result = self.tanh(result)
        result = self.l2(result)
        result = self.tanh(result)
        result = self.l3(result)
        h = self.tanh(result)

        # apply the corresponding output layer to each "head"
        for i in range(self.n_heads):
            result_i = self.multi_head_output[i](h)
            u_results[f"head {i + 1}"] = result_i

        return u_results, h
