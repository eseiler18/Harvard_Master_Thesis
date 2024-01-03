"""Loss functions"""

import torch


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
        u_0 = model_result(torch.tensor([[0]], dtype=torch.float32, device=device))[
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
def calc_loss_non_linear(x, equation_list, v_list, model, reparametrization, device):

    # dictionary to store loss results for plotting
    loss_dict = {}
    loss_dict["head"] = {}
    # current loss
    L = 0
    L_t_tot = 0

    # create the trial solution
    model_result = lambda t, reparametrization: model(t, reparametrization)[0]
    u_results = model_result(x, reparametrization)

    # determine the number of u vectors
    num_u_vectors = u_results['head 1'].shape[1]

    # loss will be the sum of the terms from the "multi-head" model, hence we iterate over each head's outputs
    for i, head_i in enumerate(u_results.keys()):
        # extract the u for the current "head", corresponding to one of the initial conditions
        # u shape (sample, num_equation, 1)
        u = u_results[head_i].unsqueeze(dim=-1)

        # compute equation(t, u)
        equation = equation_list[i](x, u)

        # compute du/dt (Note: each u is computed separately to avoid torch.autograd.grad() summing them together)
        du_dt = compute_derivative(u, x, num_u_vectors)

        # compute the L_T term
        L_t_term = du_dt - equation
        L_t = torch.matmul(L_t_term.mT, L_t_term)

        # compute the L_0 term
        if not reparametrization:
            u_0 = model_result(torch.tensor([[0]], dtype=torch.float32, device=device), reparametrization)[head_i][0].unsqueeze(dim=-1)
            L_0_term = u_0 - v_list[i]
            L_0 = torch.matmul(L_0_term.T, L_0_term)

        # compute the overall and head loss loss
        L_t_tot = L_t_tot + L_t
        if reparametrization:
            loss_dict["head"][head_i] = torch.mean(L_t)
            L += torch.mean(L_t)
        else:
            loss_dict["head"][head_i] = torch.mean(L_t) + L_0
            L += (torch.mean(L_t) + L_0)

    loss_dict["L_t_tot"] = L_t_tot
    loss_dict['L_total'] = L

    return loss_dict


# build the loss function
def calc_transfer_loss_non_linear(x, equation_transfer, v, model, reparametrization, device):

#    """  # dictionary to store loss results for plotting
#     loss_dict = {}

#     # create the trial solution
#     model_result = lambda t, reparametrization: model(t, reparametrization)[0]
#     u_results = model_result(x, reparametrization)

#     # determine the number of u vectors
#     num_u_vectors = u_results['head 1'].shape[1]

#     u = u_results['head 1'].unsqueeze(dim=-1)

#     # compute equation(t, u)
#     equation = equation_transfer(x, u)

#     # compute du/dt (Note: each u is computed separately to avoid torch.autograd.grad() summing them together)
#     du_dt = compute_derivative(u, x, num_u_vectors)

#     # compute the L_T term
#     L_t_term = du_dt - equation
#     L_t = torch.matmul(L_t_term.mT, L_t_term)
#     L = torch.mean(L_t)
#     loss_dict["L_t"] = L_t.squeeze(1).squeeze(1)

#     # compute the L_0 term if no reparametrization
#     if not reparametrization:
#         u_0 = model_result(torch.tensor([[0]], dtype=torch.float32, device=device), reparametrization=False)['head 1'][0].unsqueeze(dim=-1)
#         L_0_term = u_0 - v
#         L_0 = torch.matmul(L_0_term.T, L_0_term)
#         L += L_0.item()
#         loss_dict['L_0'] = L_0.squeeze(1)

#     loss_dict['L_total'] = L
#     loss_dict['L_total'].backward()

#     return loss_dict['L_total'], loss_dict """

    # dictionary to store loss results for plotting
    loss_dict = {}

    # create the trial solution
    model_result = lambda t, reparametrization: model(t, reparametrization)[0]
    u_results = model_result(x, reparametrization)

    # determine the number of u vectors
    num_u_vectors = u_results['head 1'].shape[1]

    # extract the u for the current "head", corresponding to one of the initial conditions
    # u shape (sample, num_equation, 1)
    u = u_results['head 1'].unsqueeze(dim=-1)

    # compute equation(t, u)
    equation = equation_transfer(x, u)

    # compute du/dt (Note: each u is computed separately to avoid torch.autograd.grad() summing them together)
    du_dt = compute_derivative(u, x, num_u_vectors)

    # compute the L_T term
    L_t_term = du_dt - equation
    L_t = torch.matmul(L_t_term.mT, L_t_term)
    loss_dict["L_t"] = L_t.squeeze(1).squeeze(1)

    # compute the L_0 term
    if not reparametrization:
        u_0 = model_result(torch.tensor([[0]], dtype=torch.float32, device=device), reparametrization)['head 1'][0].unsqueeze(dim=-1)
        L_0_term = u_0 - v
        L_0 = torch.matmul(L_0_term.T, L_0_term)
        loss_dict['L_0'] = L_0.squeeze(1)

    if reparametrization:
        loss_dict["L_total"] = torch.mean(L_t)
    else:
        loss_dict["L_total"] = torch.mean(L_t) + L_0

    loss_dict['L_total'].backward()

    return  loss_dict['L_total'], loss_dict
