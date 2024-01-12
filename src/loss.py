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
def calc_loss(x, A_list, v_list, force, model, is_A_time_dep, is_force_time_dep, device):

    # dictionary to store loss results for plotting
    loss_dict = {}
    loss_dict["head"] = {}
    # current loss
    L = 0

    # create the trial solution
    model_result = lambda t: model(t)[0]
    u_results = model_result(x)

    # determine the number of u vectors
    num_u_vectors = u_results['head 1'].shape[1]

    L_t_tot = 0
    # loss will be the sum of the terms from the "multi-head" model, hence we iterate over each head's outputs
    for i, head_i in enumerate(u_results.keys()):
        # extract the u for the current "head", corresponding to one of the initial conditions
        # u shape (sample, num_equation, 1)
        u = u_results[head_i].unsqueeze(dim=-1)

        # compute A * u if A is time dependent
        if is_A_time_dep:
            A_times_u = torch.matmul(
                A_list[i](x).reshape(-1, num_u_vectors, 1), u)
        # compute A * u if A is not time dependent
        else:
            A_times_u = torch.matmul(A_list[i], u)

        # compute du/dt (Note: each u is computed separately to avoid torch.autograd.grad() summing them together)
        du_dt = compute_derivative(u, x, num_u_vectors)

        # compute the L_T term
        if is_force_time_dep:
            force_calc = torch.cat(
                force[i](x.detach().cpu()), axis=1).unsqueeze(2).to(device)
        else:
            force_calc = torch.stack([force[i]] * len(x))
        L_t_term = du_dt + A_times_u - force_calc
        L_t = torch.matmul(L_t_term.mT, L_t_term)

        # compute the L_0 term
        u_0 = model_result(torch.tensor([[0]], dtype=torch.float64, device=device))[
            head_i][0].unsqueeze(dim=-1)
        L_0_term = u_0 - v_list[i]
        L_0 = torch.matmul(L_0_term.T, L_0_term)

        # compute head loss
        loss_dict["head"][head_i] = torch.mean(L_t) + L_0
        L_t_tot = L_t_tot + L_t

        # compute the overall loss
        L += (torch.mean(L_t) + L_0)
    loss_dict["L_t_tot"] = L_t_tot
    loss_dict['L_total'] = L

    return loss_dict


# build the loss function
def calc_loss_new(x, A_list, v_list, force, model, device):

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
    du_dt = torch.stack(du_dt, dim=1).squeeze()

    # Expand dimensions of A_list to match the shape of u_results
    expanded_A_list = torch.stack(A_list).unsqueeze(0).expand(u_results.shape[0], -1, -1, -1)

    # Compute ODE loss
    # Perform element-wise matrix multiplication using broadcasting
    Au = torch.matmul(expanded_A_list, u_results.unsqueeze(-1)).squeeze(-1)
    del expanded_A_list
    forcing = torch.stack(force).expand(u_results.shape[0], -1, -1, -1).squeeze()
    interior = du_dt + Au - forcing
    del du_dt, Au, forcing

    # Compute IC loss
    f_boundary = torch.stack([torch.tensor([i[0], i[1]], dtype=torch.float64, device=device) for i in v_list])
    u_0 = model_result(torch.tensor([[0]], dtype=torch.float64, device=device))[0]
    boundary = u_0 - f_boundary

    # All together
    loss = nn.MSELoss()
    loss_ODE = loss(interior, torch.zeros_like(interior))
    loss_IC = loss(boundary, torch.zeros_like(boundary))
    total_loss = loss_ODE + loss_IC

    loss_dict["head"] = interior.mean(0).mean(1).pow(2) + boundary.mean(1).pow(2)
    loss_dict['L_total'] = total_loss
    return loss_dict
