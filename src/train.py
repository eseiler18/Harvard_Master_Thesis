import torch
import time
import numpy as np
from tqdm import trange
from collections import defaultdict

from src.loss import calc_loss, calc_loss_new
from src.model import BuildNetwork, BuildNetworkNew


# train and evaluate the model
def run_model(iterations, x_range, lr, A_list, v_list, force, true_functs, hid_lay, activation, num_equations, num_heads, head_to_track, sample_size, is_A_time_dep, is_force_time_dep, dev, verbose):

    assert num_equations > 0, 'num_equations must be >= 1'
    assert len(v_list) == num_heads, 'num_heads must equal the length of v_list'
    assert len(A_list) == num_heads, 'num_heads must equal the length of A_list'
    assert len(v_list[0]) == num_equations, 'num_equations does not match equation set-up'
    assert hid_lay[-1] % num_equations == 0, 'last hidden layer does not evenly divide num_equations for transfer learning'

    if not is_A_time_dep:
        assert len(A_list[0]) == num_equations, 'num_equations does not match equation set-up'

    # build the neural net model
    model = BuildNetwork(1, hid_lay, num_equations, num_heads, activation).to(dev)
    # set-up the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # extract the min and max range of x values to sample
    min_x, max_x = x_range

    # index of the head being tracked for MSE
    head_idx = int(head_to_track.split()[-1]) - 1

    # create a random number generator for loss computation
    rng = np.random.default_rng()

    # store loss and mse values
    loss_history = defaultdict(list)
    loss_history["head"] = defaultdict(list)
    MSEs = []

    start_time = time.time()

    # training loop
    for i in trange(iterations):

        # every batch, randomly sample from min and max range
        x = torch.arange(min_x, max_x, 0.001, requires_grad=True, device=dev)
        x = x[rng.choice(range(0, len(x)), size=sample_size, replace=False)]
        x = x.reshape(-1, 1)
        x, _ = x.sort(dim=0)

        # forward: compute loss
        curr_loss = calc_loss(x, A_list, v_list, force, model, is_A_time_dep, is_force_time_dep, device=dev)

        if (verbose) & (i % 500 == 0):
            print(f"Iterations {i}: loss = {curr_loss['L_total'].item()}")

        if torch.isnan(curr_loss["L_total"]).item():
            print(f"Training stop after {i} because of diverge loss")
            end_time = time.time()
            total_time = end_time - start_time
            return loss_history, model, total_time, MSEs

        # store individual loss terms for plotting
        for head_i in curr_loss["head"].keys():
            loss_history["head"][head_i].append(curr_loss["head"][head_i].detach().item())
        loss_history['Ltotal_losses'].append(curr_loss['L_total'].detach().item())
        loss_history['L_t_tot'] = curr_loss["L_t_tot"].detach().cpu().view(-1).tolist()

        # backward: backpropagation
        curr_loss['L_total'].backward()

        optimizer.step()
        optimizer.zero_grad()

        # compute the mse for the head that is being monitored ('head_to_track')
        with torch.no_grad():
            current_mse = 0
            for j in range(num_equations):
                network_sol_j = model(x)[0][head_to_track][:, j].cpu().unsqueeze(dim=1).numpy()

                # compute the true solution if A is not time dependent
                true_sol_j = true_functs(x.detach().cpu(),
                                         v_list[head_idx].detach().cpu(),
                                         A_list[head_idx] if is_A_time_dep else A_list[head_idx].detach().cpu(),
                                         force[head_idx] if is_force_time_dep else force[head_idx].detach().cpu())[j]

                true_sol_j = np.expand_dims(true_sol_j, axis=1)
                current_mse += np.mean((true_sol_j - network_sol_j) ** 2)
            MSEs.append(current_mse)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Model Training Complete in{total_time: .3f} seconds")

    return loss_history, model, total_time, MSEs


def run_model_new(iterations, x_range, lr, A_list, v_list, force, hid_lay, activation, num_equations, num_heads, sample_size, decay, dev, verbose):

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
    MSEs = []

    start_time = time.time()

    # training loop
    for i in trange(iterations):

        # every batch, randomly sample from min and max range
        x = torch.arange(min_x, max_x, 0.001, requires_grad=True, device=dev).double()
        x = x[rng.choice(range(0, len(x)), size=sample_size, replace=False)]
        x = x.reshape(-1, 1)
        x, _ = x.sort(dim=0)

        # forward: compute loss
        curr_loss = calc_loss_new(x, A_list, v_list, force, model, device=dev)

        if (verbose) & (i % 500 == 0):
            print(f"Iterations {i}: loss = {curr_loss['L_total'].item()}")

        if torch.isnan(curr_loss["L_total"]).item():
            print(f"Training stop after {i} because of diverge loss")
            end_time = time.time()
            total_time = end_time - start_time
            return loss_history, model, total_time, MSEs

        # store individual loss terms for plotting
        loss_history["head"].append(curr_loss["head"].detach())
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
    loss_history["head"] = torch.stack(loss_history["head"], dim=0).cpu().numpy()

    # Compute time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Model Training Complete in{total_time: .3f} seconds")

    return loss_history, model, total_time
