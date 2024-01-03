import torch
import torch.nn as nn

from oneshot_transfer_learning.model import costum_step


# define the loss function
def loss_fn(params, f_multihead_vmap, dfdx_multihead,
            x, A_matrices, ICs, forcing_functions, X_BOUNDARY=0.0):

    f_values = f_multihead_vmap(x, params)
    f_gradients = dfdx_multihead(x, params)

    # Expand dimensions of 'A_matrices' to match the shape of 'f_values'
    expanded_A_matrices = A_matrices.unsqueeze(0).expand(f_values.shape[0], -1, -1, -1)
    # Perform element-wise matrix multiplication using broadcasting
    Au = torch.matmul(expanded_A_matrices, f_values.unsqueeze(-1).float()).squeeze(-1)
    del expanded_A_matrices

    forcing = torch.stack([f(x) for f in forcing_functions]).squeeze(1).transpose(1, 0)
    interior = f_gradients + Au - forcing
    del f_gradients, Au, forcing

    x_boundary = torch.tensor([X_BOUNDARY])
    f_boundary = torch.stack([torch.Tensor([i[0], i[1]]) for i in ICs])
    boundary = f_multihead_vmap(x_boundary, params) - f_boundary

    loss = nn.MSELoss()
    loss_colocation = loss(interior, torch.zeros_like(interior))
    loss_BC = loss(boundary, torch.zeros_like(boundary))
    loss_value = loss_colocation + loss_BC
    return loss_value, loss_colocation, loss_BC


# define the function that trains the model
def train_model(num_iter, params, f_multihead_vmap, dfdx_multihead,
                batch_size, domain, A_matrices, ICs, forcing_functions,
                optimizer, display_every=100,
                decay=False, gamma=1, every=100):
    loss_trace = []
    colocation_trace = []
    BC_trace = []
    for i in range(num_iter):
        # sample colocation points in the domain randomly at each iteration
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # update the parameters using the functional API
        loss, loss_colocation, loss_BC = loss_fn(params, f_multihead_vmap, dfdx_multihead,
                                                 x, A_matrices, ICs, forcing_functions)

        params = costum_step(optimizer, loss, params, decay=decay, gamma=gamma, every=every, num_iter=i + 1)

        loss_trace.append(float(loss))
        colocation_trace.append(float(loss_colocation))
        BC_trace.append(float(loss_BC))

        if ((i + 1) % display_every == 0):
            print(f"Iteration {i} with loss {float(loss)}, colocation: {float(loss_colocation)}, BC: {float(loss_BC)}")

    del loss, loss_colocation, loss_BC
    return params, loss_trace, colocation_trace, BC_trace
