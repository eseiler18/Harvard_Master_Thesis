import torch
import torch.nn as nn
from torchopt.update import apply_updates
from torchopt.typing import UninitializedState


class Multihead(nn.Module):
    def __init__(self, k, act=nn.Tanh(), IC_list=None):
        super().__init__()
        self.IC_list = IC_list
        self.act = act
        self.linear1 = nn.Linear(1, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 512)
        # define k final layers without bias
        self.final_layers = nn.ModuleList([nn.Linear(256, 1, bias=False) for _ in range(k)])
        self.k = k

    # it returns the output of the network and the hidden state
    def forward(self, x, reparametrization=False):
        out = self.act(self.linear1(x))
        out = self.act(self.linear2(out))
        out = self.act(self.linear3(out))
        out = self.act(self.linear4(out))
        out1 = out[:256]
        out2 = out[256:]
        output = []
        for i in range(self.k):
            first = self.final_layers[i](out1)
            second = self.final_layers[i](out2)
            concat = torch.cat((first, second))
            if reparametrization:
                N0 = self.forward(torch.tensor([0], dtype=torch.float32))[0][i, ...]
                concat = concat + ((self.IC_list[i].T - N0).expand(x.shape[0], -1) * torch.exp(-x))
                concat = concat.squeeze()
            output.append(concat)
        return torch.stack(output), out


def costum_step(
        optimizer,
        loss: torch.Tensor,
        params,
        inplace: bool | None = None,
        decay=False,
        gamma=1,
        every=100,
        num_iter=None):
    if isinstance(optimizer.optim_state, UninitializedState):
        optimizer.optim_state = optimizer.impl.init(params)

    if inplace is None:
        inplace = optimizer.inplace

    # Step parameter only
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=True)
    if decay:
        grads = tuple(grads[i] * (gamma**(num_iter / every)) for i in range(len(grads)))
    updates, optimizer.optim_state = optimizer.impl.update(
        grads,
        optimizer.optim_state,
        params=params,
        inplace=inplace,
    )
    return apply_updates(params, updates, inplace=inplace)
