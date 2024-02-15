"""Loss functions"""

import torch
import torch.nn as nn


# calculate du^n / dt^n for arbitrary n and use this to form loss
def compute_derivative(u, t, num_u_vectors):

    # compute derivative of outputs with respect to inputs
    derivs_list = []
    for i in range(num_u_vectors):
        # individually compute du/dt for each u and combine them all together afterwards
        du_dt = torch.autograd.grad(
            u[:, i, :], t, grad_outputs=torch.ones_like(u[:, i, :]), create_graph=True)[0]
        derivs_list.append(du_dt)

    deriv_to_t = torch.stack(derivs_list, dim=1)

    return deriv_to_t


# build the loss function
def calc_loss_new(x, A_list, v_list, force, model, numerical_solution, t_eval, device):

    num_heads = len(A_list)

    # dictionary to store loss results for plotting
    loss_dict = {}

    # compute the PINNS solution
    model_result = lambda t: model(t)[0]
    u_results = model_result(x)

    # compute du/dt
    num_u_vectors = u_results.shape[2]
    du_dt = []
    for i in range(u_results.shape[1]):
        du_dt.append(compute_derivative(u_results[:, i, :].unsqueeze(dim=-1), x, num_u_vectors))
    du_dt = torch.stack(du_dt, dim=1).reshape(-1, num_heads, num_u_vectors)

    # Expand dimensions of A_list to match the shape of u_results
    expanded_A_list = torch.stack(A_list).unsqueeze(0).expand(u_results.shape[0], -1, -1, -1)

    # Compute ODE loss
    # Perform element-wise matrix multiplication using broadcasting
    Au = torch.matmul(expanded_A_list, u_results.unsqueeze(-1)).squeeze(-1)
    del expanded_A_list
    if callable(force[0]):
        forcing = []
        for f in force:
            forcing.append(f(x.squeeze()))
        forcing = torch.stack(forcing, dim=1)
    else:
        forcing = torch.stack(force, dim=1).T.expand(u_results.shape[0], -1, -1).reshape(-1, num_heads, num_u_vectors)
    interior = du_dt + Au - forcing
    del du_dt, Au, forcing

    # Compute IC loss
    f_boundary = torch.stack([torch.tensor([i[j] for j in range(v_list[0].shape[0])], dtype=torch.float64, device=device) for i in v_list])
    u_0 = model_result(torch.tensor([[0]], dtype=torch.float64, device=device))[0]
    boundary = u_0 - f_boundary

    # Compute Data Loss
    if t_eval is not None:
        u_error = model_result(t_eval.unsqueeze(1))
        error = u_error - numerical_solution

    # All together
    loss = nn.MSELoss()
    loss_ODE = loss(interior, torch.zeros_like(interior))

    loss_IC = loss(boundary, torch.zeros_like(boundary))
    total_loss = loss_ODE + loss_IC
    loss_dict["head"] = interior.mean(0).mean(1).pow(2) + boundary.mean(1).pow(2)
    if t_eval is not None:
        loss_error = loss(error, torch.zeros_like(error))
        total_loss += loss_error
        loss_dict["head"] += error.mean(0).mean(1).pow(2)

    loss_dict['L_total'] = total_loss
    loss_dict['L_ODE'] = loss_ODE
    loss_dict['L_IC'] = loss_IC
    if t_eval is not None:
        loss_dict['L_error'] = loss_error
    return loss_dict


# build the loss function
def calc_loss_nonlinear(x, equation_list, v_list, model, numerical_solution, t_eval, reparametrization, device):

    num_heads = len(equation_list)

    # dictionary to store loss results for plotting
    loss_dict = {}

    # compute the PINNS solution
    model_result = lambda t: model(t, reparametrization=reparametrization)[0]
    u_results = model_result(x)
    num_u_vectors = u_results.shape[2]

    # compute du/dt and equation(u)
    du_dt = []
    equation_u = []
    for i in range(num_heads):
        du_dt.append(compute_derivative(u_results[:, i, :].unsqueeze(dim=-1), x, num_u_vectors))
        equation_u.append(equation_list[i](x.squeeze(), u_results[:, i, :])) 
    du_dt = torch.stack(du_dt, dim=1).reshape(-1, num_heads, num_u_vectors)
    equation_u = torch.stack(equation_u, dim=1).reshape(-1, num_heads, num_u_vectors)

    interior = du_dt - equation_u
    del du_dt, equation_u

    # Compute IC loss
    if ~reparametrization:
        f_boundary = torch.stack([torch.tensor([i[j] for j in range(v_list[0].shape[0])], dtype=torch.float64, device=device) for i in v_list])
        u_0 = model_result(torch.tensor([[0]], dtype=torch.float64, device=device))[0]
        boundary = u_0 - f_boundary
    else:
        boundary = torch.zeros((1, 2)).double().to(device)

    # Compute Data Loss
    if t_eval is not None:
        u_error = model_result(t_eval.unsqueeze(1))
        error = u_error - numerical_solution

    # All together
    loss = nn.MSELoss()
    loss_ODE = loss(interior, torch.zeros_like(interior))

    loss_IC = loss(boundary, torch.zeros_like(boundary))
    total_loss = loss_ODE + loss_IC
    loss_dict["head"] = interior.mean(0).mean(1).pow(2) + boundary.mean(1).pow(2)
    if t_eval is not None:
        loss_error = loss(error, torch.zeros_like(error))
        total_loss += loss_error
        loss_dict["head"] += error.mean(0).mean(1).pow(2)

    loss_dict['L_total'] = total_loss
    loss_dict['L_ODE'] = loss_ODE
    loss_dict['L_IC'] = loss_IC
    if t_eval is not None:
        loss_dict['L_error'] = loss_error
    return total_loss, loss_dict
