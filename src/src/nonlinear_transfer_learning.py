from collections import defaultdict
import time
import torch
import torch.nn as nn
from tqdm.auto import trange
import numpy as np
from src.transfer_learning import compute_M_inv, compute_W_with_IC_and_force
from src.loss import calc_loss_nonlinear


def compute_initial(x0, beta, p):
    denominator = 0
    for i in range(p + 1):
        denominator += beta**i
    return x0 / denominator


def build_force_function(n, force_func_index):
    list_force_index = []
    for i in range(n + 1):
        list_force_index.append(force_func_index(i))
    return list_force_index


def solve_perturbation_TL(beta, p, t_eval,
                          alpha, A, force, IC,
                          H, H_0, dH_dt, dev,
                          force_func_index,
                          numerical_sol_fct,
                          force_function_PINNS,
                          force_function_numerical,
                          compute_numerical_pert,
                          numerical_perturbation_fct,
                          verbose, M_inv=None):

    # compute IC of the p systems
    IC_pertubation = torch.tensor([[compute_initial(IC[i], beta, p)] for i in range(len(IC))], device=dev)

    # compute the inverse of the M matrix associate with the equation

    if verbose:
        print("Solving the 0th linear ode system...")
    total_time = 0
    if M_inv is None:
        start_time = time.time()
        M_inv = compute_M_inv(dH_dt, H, H_0, t_eval, A)
        total_time = time.time() - start_time
        if verbose:
            print(f"Time to compute inverse of M: {total_time:.2e}")

    # compute the analytic W0
    W0, comp_time = compute_W_with_IC_and_force(t_eval, A, IC_pertubation, force, H, H_0, dH_dt, M_inv)
    total_time += comp_time
    if verbose:
        print(f"Time to compute W0: {comp_time:.2e}")
        print("=======================================================")

    # store the solution in t_eval for the force function of next systems
    u0 = torch.matmul(H, W0)
    PINNS_list = [u0]

    # compute numerical solution of the first system (numercial solution bu perturbation)
    if compute_numerical_pert:
        numerical_pert = numerical_sol_fct(t_eval.detach().cpu().numpy(),
                                           IC_pertubation.detach().cpu().numpy() if IC_pertubation.shape[0] != 1 else IC_pertubation.detach().cpu().numpy().squeeze(1),
                                           A.detach().cpu().numpy(), force)
        # store the numerical solution
        numerical_pert_list = [numerical_pert]

    # find the index of each systems force function
    list_force_index = build_force_function(p, force_func_index)

    # solve the p systems
    for i in range(1, p + 1):
        if verbose:
            print(f"Solving the {i}th linear ode system...")

        # compute the analytic Wi
        fi = force_function_PINNS(i, alpha, list_force_index, PINNS_list)
        Wi, computational_time = compute_W_with_IC_and_force(t_eval, A, IC_pertubation, fi, H, H_0, dH_dt, M_inv)
        total_time += computational_time

        if verbose:
            print(f"Time to compute W{i}: {computational_time:.2e}")
        # store the solution in t_eval for the force function of next systems
        ui = torch.matmul(H, Wi)
        PINNS_list.append(ui)

        if compute_numerical_pert:
            start_time = time.time()
            fi_numerical = force_function_numerical(i, alpha, list_force_index, numerical_pert_list)
            numerical_pert_list.append(numerical_perturbation_fct(t_eval.detach().cpu(),
                                                                  IC_pertubation.detach().cpu().numpy() if IC_pertubation.shape[0] != 1 else IC_pertubation.detach().cpu().numpy().squeeze(1),
                                                                  A.detach().cpu(), fi_numerical).y)
            if verbose:
                print(f"Time to compute the perturbation numerical solution {time.time()-start_time:.2e}")

        if verbose:
            print("=======================================================")
    if verbose:
        print(f"{p+1} systems solved in {total_time:.3e} seconds")

    # Compute PINNS general perturbation solution
    solution_PINNS = sum([beta**j * u for j, u in enumerate(PINNS_list)])
    solution_PINNS = solution_PINNS.detach().cpu().numpy().squeeze()
    # detach each solution of the p systems
    PINNS_list = [u.detach().cpu().numpy().squeeze() for u in PINNS_list]

    # Compute numerical general perturbation solution
    if compute_numerical_pert:
        solution_numerical = sum([beta**j * u for j, u in enumerate(numerical_pert_list)]).T
        numerical_pert_list = [u.T for u in numerical_pert_list]
    else:
        solution_numerical = None
        numerical_pert_list = None

    return solution_PINNS, solution_numerical, PINNS_list, numerical_pert_list, total_time


def GD_transfer_learning(iterations, x_range, N, equation_transfer, IC, num_equations,
                         dev, hid_lay, pretrained_model, lr, optimizer_name,
                         decay=True, gamma=0.1, reparametrization=False, tqdm_bool=False):

    for i, pretrained_layer in enumerate(pretrained_model.hidden_layers):
        if isinstance(pretrained_layer, nn.Linear):
            for param in pretrained_layer.parameters():
                param.requires_grad = False  # Freeze the layer
    pretrained_model.multi_head_output = nn.ModuleList([nn.Linear(hid_lay[-1], num_equations)]).double()
    pretrained_model.multi_head_output[0].bias = None
    pretrained_model.n_heads = 1
    pretrained_model.to(dev)

    # set-up the optimizer
    if optimizer_name == "LBFGS":
        optimizer = torch.optim.LBFGS(pretrained_model.parameters(), history_size=100, max_iter=20, lr=lr)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=lr)

    # store loss and mse values
    loss_history = defaultdict(list)
    start_time = time.time()

    # training loop
    for i in trange(iterations) if tqdm_bool else range(iterations):

        rng = np.random.default_rng()
        x = torch.arange(x_range[0], x_range[1], 0.001, requires_grad=True, device=dev).double()
        x = x[rng.choice(range(0, len(x)), size=N, replace=False)]
        x = x.reshape(-1, 1)
        x, _ = x.sort(dim=0)

        global curr_loss
        if optimizer_name == "LBFGS":
            def closure():
                optimizer.zero_grad()
                global curr_loss
                L, curr_loss = calc_loss_nonlinear(x, [equation_transfer], [IC], pretrained_model, numerical_solution=None, t_eval=None, device=dev, reparametrization=reparametrization)
                if (i % 1 == 0):
                    info_loss = f"Iterations {i}"
                    for k, v in curr_loss.items():
                        if k != "head":
                            info_loss += f", {k} = {v}"
                print(info_loss)
                L.backward(retain_graph=True)
                return L
            optimizer.step(closure)

        if optimizer_name == "Adam":
            _, curr_loss = calc_loss_nonlinear(x, [equation_transfer], [IC], pretrained_model, numerical_solution=None, t_eval=None, device=dev, reparametrization=reparametrization)
            if (i % 100 == 0):
                info_loss = f"Iterations {i}"
                for k, v in curr_loss.items():
                    if k != "head":
                        info_loss += f", {k} = {v}"
                print(info_loss)
            curr_loss['L_total'].backward()
            if decay:
                gamma = 0.95  # Adjust the decay factor accordingly
                every = 50  # Adjust the decay interval accordingly
                for param in pretrained_model.multi_head_output[0].parameters():
                    param.grad *= (gamma**((i + 1) / every))
            optimizer.step()
            optimizer.zero_grad()

        # store individual loss terms for plotting
        loss_history['L_IC'].append(curr_loss['L_IC'].detach().item())
        loss_history['L_ODE'].append(curr_loss['L_ODE'].detach().item())
        loss_history['L_total'].append(curr_loss['L_total'].detach().item())

    end_time = time.time()
    total_time = end_time - start_time
    return loss_history, pretrained_model, total_time
