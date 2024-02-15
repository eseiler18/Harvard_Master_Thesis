import torch
import time
import numpy as np
from tqdm.auto import trange
from collections import defaultdict


from src.loss import calc_loss_new, calc_loss_nonlinear
from src.model import BuildNetworkNew
from src.load_save import save_model


def run_model_new(iterations, x_range, lr, A_list, v_list, force, hid_lay, activation, num_equations, num_heads, sample_size, decay, dev, verbose, true_functs=None, save=False):

    assert num_equations > 0, 'num_equations must be >= 1'
    assert len(v_list) == num_heads, 'num_heads must equal the length of v_list'
    assert len(A_list) == num_heads, 'num_heads must equal the length of A_list'
    assert len(v_list[0]) == num_equations, 'num_equations does not match equation set-up'
    assert hid_lay[-1] % num_equations == 0, 'last hidden layer does not evenly divide num_equations for transfer learning'

    # build the neural net model
    model = BuildNetworkNew(1, hid_lay, num_equations, num_heads, activation).to(dev, dtype=torch.double)
    # set-up the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=lr))

    # extract the min and max range of x values to sample
    min_x, max_x = x_range

    # create a random number generator for loss computation
    rng = np.random.default_rng()

    # store loss and mse values
    loss_history = defaultdict(list)
    loss_head = []

    start_time = time.time()

    if true_functs is not None:
        t_eval = torch.linspace(min_x, max_x, sample_size, device=dev, requires_grad=True).double()
        numerical_solution = []
        for i, (A, v, f) in enumerate(zip(A_list, v_list, force)):
            numerical_solution.append(torch.tensor(true_functs(t_eval.detach().cpu().numpy(),
                                                   v.detach().cpu().numpy(),
                                                   A.detach().cpu().numpy(),
                                                   f), device=dev).double().T)
        numerical_solution = torch.stack(numerical_solution, dim=1)
    else:
        t_eval = None
        numerical_solution = None

    # training loop
    for i in trange(iterations):

        # every batch, randomly sample from min and max range
        x = torch.arange(min_x, max_x, 0.001, requires_grad=True, device=dev).double()
        x = x[rng.choice(range(0, len(x)), size=sample_size, replace=False)]
        x = x.reshape(-1, 1)
        x, _ = x.sort(dim=0)

        # forward: compute loss
        curr_loss = calc_loss_new(x, A_list, v_list, force, model, numerical_solution, t_eval, device=dev)

        if (verbose) & (i % 100 == 0):
            info_loss = f"Iterations {i}"
            for k, v in curr_loss.items():
                if k != "head":
                    info_loss += f", {k} = {v}"
            print(info_loss)

        if torch.isnan(curr_loss["L_total"]).item():
            print(f"Training stop after {i} because of diverge loss")
            end_time = time.time()
            total_time = end_time - start_time
            return loss_history, model, total_time

        # store individual loss terms for plotting
        loss_head.append(curr_loss["head"].detach())
        loss_history['L_IC'].append(curr_loss['L_IC'].detach().item())
        loss_history['L_ODE'].append(curr_loss['L_ODE'].detach().item())
        loss_history['L_total'].append(curr_loss['L_total'].detach().item())

        # backward: backpropagation
        curr_loss['L_total'].backward()

        if decay:
            gamma = 0.98  # Adjust the decay factor accordingly
            every = 100  # Adjust the decay interval accordingly
            for param in model.parameters():
                param.grad *= (gamma**((i + 1) / every))

        optimizer.step()
        optimizer.zero_grad()

        if (isinstance(save, np.ndarray)) & (i % 5000 == 0):
            if i > 0:
                print("Save model")
                loss_history["head"] = torch.stack(loss_head, dim=0).cpu().numpy()
                save_model(model, i, "VDP", "BIG_inference",
                           x_range, iterations, hid_lay, num_equations, num_heads, A_list,
                           v_list, force, save, loss_history)

    # Stack all head loss
    loss_history["head"] = torch.stack(loss_head, dim=0).cpu().numpy()

    # Compute time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Model Training Complete in{total_time: .3f} seconds")

    return loss_history, model, total_time


def run_model_non_linear(iterations, x_range, equation_list, v_list, lr, hid_lay, activation,
                         num_equations, num_heads, sample_size, decay, dev, verbose,
                         true_functs=None, reparametrization=False):

    # build the neural net model
    model = BuildNetworkNew(input_size=1, h_sizes=hid_lay, output_size=num_equations,
                            n_heads=num_heads, dev=dev, activation=activation,
                            IC_list=v_list).to(dev, dtype=torch.double)
    # set-up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # extract the min and max range of x values to sample
    min_x, max_x = x_range

    # create a random number generator for loss computation
    rng = np.random.default_rng()

    # store loss and mse values
    loss_history = defaultdict(list)
    loss_head = []

    start_time = time.time()

    if true_functs is not None:
        t_eval = torch.linspace(min_x, max_x, sample_size, device=dev, requires_grad=True).double()
        numerical_solution = []
        for true_funct in true_functs:
            numerical_solution.append(torch.tensor(true_funct(t_eval)))
        numerical_solution = torch.stack(numerical_solution, dim=1)
    else:
        t_eval = None
        numerical_solution = None

    # training loop
    for i in trange(iterations):

        # every batch, randomly sample from min and max range
        x = torch.arange(min_x, max_x, 0.001, requires_grad=True, device=dev).double()
        x = x[rng.choice(range(0, len(x)), size=sample_size, replace=False)]
        x = x.reshape(-1, 1)
        x, _ = x.sort(dim=0)

        # forward: compute loss
        _, curr_loss = calc_loss_nonlinear(x, equation_list, v_list, model, numerical_solution, t_eval, reparametrization, device=dev)

        if (verbose) & (i % 100 == 0):
            info_loss = f"Iterations {i}"
            for k, v in curr_loss.items():
                if k != "head":
                    info_loss += f", {k} = {v}"
            print(info_loss)

        if torch.isnan(curr_loss["L_total"]).item():
            print(f"Training stop after {i} because of diverge loss")
            end_time = time.time()
            total_time = end_time - start_time
            return loss_history, model, total_time

        # store individual loss terms for plotting
        loss_head.append(curr_loss["head"].detach())
        loss_history['L_IC'].append(curr_loss['L_IC'].detach().item())
        loss_history['L_ODE'].append(curr_loss['L_ODE'].detach().item())
        loss_history['L_total'].append(curr_loss['L_total'].detach().item())

        # backward: backpropagation
        curr_loss['L_total'].backward()

        if decay:
            gamma = 0.98  # Adjust the decay factor accordingly
            every = 100  # Adjust the decay interval accordingly
            for param in model.parameters():
                param.grad *= (gamma**((i + 1) / every))

        optimizer.step()
        optimizer.zero_grad()

    # Stack all head loss
    loss_history["head"] = torch.stack(loss_head, dim=0).cpu().numpy()

    # Compute time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Model Training Complete in{total_time: .3f} seconds")

    return loss_history, model, total_time
