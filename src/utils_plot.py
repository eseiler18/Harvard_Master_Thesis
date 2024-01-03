"""Utils plot function for vizualisation"""

import matplotlib.pyplot as plt
import torch
import numpy as np


# 1) Plot Solution, Loss, and MSE Information
# function to plot the neural network vs exact solution
def plot_solution(x_range, true_functs, trained_model, v_list, A_list, force, axis, head_to_track, is_A_time_dep, is_force_time_dep, device):

    # function to extract the model results
    model_result = lambda t: trained_model(t)[0]

    # x values to predict on
    min_x, max_x = x_range
    xx = np.linspace(min_x, max_x, 200)[:, None]

    # find the model results
    u = model_result(torch.tensor(xx, dtype=torch.float32, device=device))[head_to_track]
    # determine the number of curves to plot
    num_curves = u.shape[1]
    # store the true solutions and network solutions
    yys, yts = [], []

    # save the network solutions in a list for plotting
    with torch.no_grad():
        head_idx = int(head_to_track.split()[-1]) - 1
        for i in range(num_curves):
            yys.append(u[:, i].cpu().numpy())

            # find the true solution if A is not time independent

            yts.append(true_functs(xx,
                                   v_list[head_idx].detach().cpu(),
                                   A_list[head_idx] if is_A_time_dep else A_list[head_idx].detach().cpu(),
                                   force[head_idx] if is_force_time_dep else force[head_idx].detach().cpu())[i])

    # plot the network solutions
    for i in range(num_curves):
        axis.plot(xx, yys[i], 'x', markersize=8, label=f'Network Solution $u_{{{i+1}}}(t)$ ({head_to_track})',
                  linewidth=3.5)

    # plot the true solutions
    for i in range(num_curves):
        axis.plot(xx, yts[i].reshape(-1, 1), label=f'Numerical Solution $u_{{{i+1}}}(t)$', linewidth=2.5)

    axis.set_title("$u(t)$ for Network and Numerical Solutions", fontsize=20)
    axis.set_xlabel('$t$', fontsize=16)
    axis.set_ylabel('$u(t)$', fontsize=16)
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=16)
    axis.grid()
    axis.legend(loc='best', fontsize=8)


# function to plot the overall loss of the network solution
def plot_total_loss(train_losses, axis, loss_label):
    axis.plot(range(len(train_losses)), train_losses, label=loss_label)
    axis.set_yscale("log")
    axis.set_title("Total Loss vs Iterations", fontsize=20)
    axis.set_xlabel('Iterations', fontsize=16)
    axis.set_ylabel('Loss', fontsize=16)
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=16)
    axis.grid()
    axis.legend(loc='best', fontsize=16)


def plot_solution_non_linear(x_range, true_functs, trained_model, v_list, reparametrization, axis, head_to_track, device):

    # function to extract the model results
    model_result = lambda t, reparametrization: trained_model(t, reparametrization)[0]

    # x values to predict on
    min_x, max_x = x_range
    xx = np.linspace(min_x, max_x, 200)[:, None]

    # find the model results
    u = model_result(torch.tensor(xx, dtype=torch.float32, device=device), reparametrization)[head_to_track]
    # determine the number of curves to plot
    num_curves = u.shape[1]
    # store the true solutions and network solutions
    yys, yts = [], []

    # save the network solutions in a list for plotting
    with torch.no_grad():
        head_idx = int(head_to_track.split()[-1]) - 1
        for i in range(num_curves):
            yys.append(u[:, i].cpu().numpy())

            # find the true solution if A is not time independent
            true_funct = true_functs[head_idx]
            yts.append(true_funct(xx,
                                  v_list[head_idx].detach().cpu())[i])

    # plot the network solutions
    for i in range(num_curves):
        axis.plot(xx, yys[i], 'x', markersize=8, label=f'Network Solution $u_{{{i+1}}}(t)$ ({head_to_track})',
                  linewidth=3.5)

    # plot the true solutions
    for i in range(num_curves):
        axis.plot(xx, yts[i].reshape(-1, 1), label=f'Numerical Solution $u_{{{i+1}}}(t)$', linewidth=2.5)

    axis.set_title("$u(t)$ for Network and Numerical Solutions", fontsize=20)
    axis.set_xlabel('$t$', fontsize=16)
    axis.set_ylabel('$u(t)$', fontsize=16)
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=16)
    axis.grid()
    axis.legend(loc='best', fontsize=8)


# function to plot the MSE
def plot_mse(mses, axis, head_to_track):
    axis.plot(range(len(mses)), mses, label=f'MSE ({head_to_track})')
    axis.set_yscale("log")
    axis.set_title("MSE vs Iterations", fontsize=20)
    axis.set_xlabel('Iterations', fontsize=16)
    axis.set_ylabel('MSE', fontsize=16)
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=16)
    axis.grid()
    axis.legend(loc='best', fontsize=16)


# plot head losses
def plot_head_loss(head_loss, axis):
    for _, head in enumerate(head_loss):
        axis.plot(range(len(head_loss[head])), head_loss[head], label=f'{head}')
    axis.set_yscale("log")
    axis.set_title("Head loss vs Iterations", fontsize=20)
    axis.set_xlabel('Iterations', fontsize=16)
    axis.set_ylabel('Loss', fontsize=16)
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=16)
    axis.grid()
    axis.legend(loc='best', fontsize=16)


# wrapper function to plot the solution and the overall loss & MSE of the network solution
def plot_loss_mse_and_solution(x_range, true_functs, iterations, trained_model, v_list,
                               A_list, force, train_losses, loss_label, mses, head_to_track, is_A_time_dep, device):

    fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(24, 8))

    plot_total_loss(train_losses=train_losses,
                    axis=axs[0], loss_label=loss_label)

    plot_solution(x_range=x_range, true_functs=true_functs,
                  trained_model=trained_model, v_list=v_list,
                  A_list=A_list, force=force, axis=axs[1],
                  head_to_track=head_to_track, is_A_time_dep=is_A_time_dep, device=device)

    plot_mse(mses=mses, axis=axs[2],
             head_to_track=head_to_track)

    plt.show()


# wrapper function to plot all heads and the overall loss & MSE of the network solution
def plot_loss_mse_and_all_solution(x_range, true_functs, iterations, trained_model, v_list,
                                   A_list, force, train_losses, loss_label, mses, is_A_time_dep, is_force_time_dep, device):

    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(24, 16))

    plot_total_loss(train_losses=train_losses,
                    axis=axs[0, 0], loss_label=loss_label)

    plot_mse(mses=mses, axis=axs[0, 1],
             head_to_track='head 1')

    plot_solution(x_range=x_range, true_functs=true_functs,
                  trained_model=trained_model, v_list=v_list,
                  A_list=A_list, force=force, axis=axs[0, 2],
                  head_to_track='head 1', is_A_time_dep=is_A_time_dep,
                  is_force_time_dep=is_force_time_dep, device=device)

    plot_solution(x_range=x_range, true_functs=true_functs,
                  trained_model=trained_model, v_list=v_list,
                  A_list=A_list, force=force, axis=axs[1, 0],
                  head_to_track='head 2', is_A_time_dep=is_A_time_dep,
                  is_force_time_dep=is_force_time_dep, device=device)

    plot_solution(x_range=x_range, true_functs=true_functs,
                  trained_model=trained_model, v_list=v_list,
                  A_list=A_list, force=force, axis=axs[1, 1],
                  head_to_track='head 3', is_A_time_dep=is_A_time_dep,
                  is_force_time_dep=is_force_time_dep, device=device)

    plot_solution(x_range=x_range, true_functs=true_functs,
                  trained_model=trained_model, v_list=v_list,
                  A_list=A_list, force=force, axis=axs[1, 2],
                  head_to_track='head 4', is_A_time_dep=is_A_time_dep,
                  is_force_time_dep=is_force_time_dep, device=device)

    plt.show()


# wrapper function to plot all heads and the overall loss & MSE of the network solution
def plot_loss_mse_and_all_solution_non_linear(x_range, true_functs, iterations, trained_model, reparametrization,
                                              v_list, train_losses, loss_label, mses, device):

    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(24, 16))

    plot_total_loss(train_losses=train_losses, axis=axs[0, 0], loss_label=loss_label)

    plot_mse(mses=mses, axis=axs[0, 1], head_to_track='head 1')

    plot_solution_non_linear(x_range=x_range, true_functs=true_functs,
                             trained_model=trained_model, v_list=v_list,
                             reparametrization=reparametrization, axis=axs[0, 2],
                             head_to_track='head 1', device=device)

    plot_solution_non_linear(x_range=x_range, true_functs=true_functs,
                             trained_model=trained_model, v_list=v_list,
                             reparametrization=reparametrization, axis=axs[1, 0],
                             head_to_track='head 2', device=device)

    plot_solution_non_linear(x_range=x_range, true_functs=true_functs,
                             trained_model=trained_model, v_list=v_list,
                             reparametrization=reparametrization, axis=axs[1, 1],
                             head_to_track='head 3', device=device)

    plot_solution_non_linear(x_range=x_range, true_functs=true_functs,
                             trained_model=trained_model, v_list=v_list,
                             reparametrization=reparametrization, axis=axs[1, 2],
                             head_to_track='head 4', device=device)

    plt.show()


# 2) Plot Transfer Learned and Analytical Solutions
# function to plot the transfer learned and analytical solutions on the same graph
def plot_transfer_learned_and_analytical_non_linear(H, W_out, t_eval, v, num_equations, true_funct):

    fig, axs = plt.subplots(1, 2, tight_layout=False, figsize=(16, 8))

    # compute the transfer learned solution
    #u_transfer = torch.matmul(H, W_out)
    u_transfer = W_out(H)

    # plot the transfer learned solutions
    for i in range(num_equations):
        axs[0].plot(t_eval.detach().cpu().numpy(), u_transfer[:, i].detach().cpu().numpy(), 'x',
                    markersize=8, label=f'Transfer Learned $U_{{{i+1}}}$', linewidth=3.5)

    # plot the true solutions
    for i in range(num_equations):
        axs[0].plot(t_eval.detach().cpu().numpy(),
                    true_funct(t_eval.detach().cpu().numpy(),
                               v.detach().cpu())[i],
                    label=f'True $U_{{{i+1}}}$', linewidth=2.5)

    axs[0].set_title("$u(t)$ for Transfer Learned and Numerical Solutions", fontsize=20)
    axs[0].set_xlabel("t", fontsize=16)
    axs[0].set_ylabel("$u(t)$", fontsize=16)
    axs[0].tick_params(axis='x', labelsize=16)
    axs[0].tick_params(axis='y', labelsize=16)
    axs[0].grid()
    axs[0].legend()

    # plot the residuals
    for i in range(num_equations):
        x_vals = t_eval.detach().cpu().numpy()
        predicted_vals = u_transfer[:, i].detach().cpu().numpy().squeeze()
        true_vals = true_funct(x_vals,
                               v.detach().cpu())[i]
        residuals = (predicted_vals - true_vals) ** 2
        print(f"mean {residuals.mean()}")
        print(f"max {residuals.max()}")
        axs[1].plot(t_eval.detach().cpu().numpy(), residuals, label=f'Residual $U_{{{i+1}}}$')

    axs[1].set_title("Plot of Residuals vs Network Input $t$", fontsize=20)
    axs[1].set_xlabel("Network Input $t$", fontsize=16)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Residual Value', fontsize=16)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    axs[1].grid()
    axs[1].legend()


# 2) Plot Transfer Learned and Analytical Solutions
# function to plot the transfer learned and analytical solutions on the same graph
def plot_transfer_learned_and_analytical(H, W_out, t_eval, v, A, force, num_equations, true_funct, is_A_time_dep, is_force_time_dep):

    fig, axs = plt.subplots(1, 2, tight_layout=False, figsize=(16, 8))

    # compute the transfer learned solution
    u_transfer = torch.matmul(H.double(), W_out.double())

    # plot the transfer learned solutions
    for i in range(num_equations):
        axs[0].plot(t_eval.detach().cpu().numpy(), u_transfer[:, i, :].detach().cpu().numpy(), 'x',
                    markersize=8, label=f'Transfer Learned $U_{{{i+1}}}$', linewidth=3.5)

    # plot the true solutions
    for i in range(num_equations):
        axs[0].plot(t_eval.detach().cpu().numpy(),
                    true_funct(t_eval.detach().cpu().numpy(), v.detach().cpu(),
                               A if is_A_time_dep else A.detach().cpu(),
                               force if is_force_time_dep else force.detach().cpu())[i],
                    label=f'True $U_{{{i+1}}}$', linewidth=2.5)

    axs[0].set_title("$u(t)$ for Transfer Learned and Numerical Solutions", fontsize=20)
    axs[0].set_xlabel("t", fontsize=16)
    axs[0].set_ylabel("$u(t)$", fontsize=16)
    axs[0].tick_params(axis='x', labelsize=16)
    axs[0].tick_params(axis='y', labelsize=16)
    axs[0].grid()
    axs[0].legend()

    # plot the transfer learned solutions
    for i in range(num_equations):
        x_vals = t_eval.detach().cpu().numpy()
        predicted_vals = u_transfer[:, i, :].detach().cpu().numpy().squeeze()
        true_vals = true_funct(x_vals, v.detach().cpu(),
                               A if is_A_time_dep else A.cpu(),
                               force if is_force_time_dep else force.detach().cpu())[i]

        residuals = (predicted_vals - true_vals) ** 2
        print(f"mean {residuals.mean()}")
        print(f"max {residuals.max()}")
        axs[1].plot(t_eval.detach().cpu().numpy(), residuals, label=f'Residual $U_{{{i+1}}}$')

    axs[1].set_title("Plot of Residuals vs Network Input $t$", fontsize=20)
    axs[1].set_xlabel("Network Input $t$", fontsize=16)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Residual Value', fontsize=16)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    axs[1].grid()
    axs[1].legend()


# function to plot only the residuals of the equations
def plot_residuals(H, W_out, t_eval, v, A, force, num_equations, true_funct, is_A_time_dep):

    # compute the transfer learned solution
    u_transfer = torch.matmul(H, W_out)

    # plot the transfer learned solutions
    for i in range(num_equations):
        x_vals = t_eval.detach().cpu().numpy()
        predicted_vals = u_transfer[:, i, :].detach().cpu().numpy().squeeze()
        true_vals = true_funct(x_vals, v.detach().cpu(), A.cpu(), force.detach().cpu())[i]
        residuals = (predicted_vals - true_vals) ** 2
        plt.plot(t_eval.detach().cpu().numpy(), residuals, label=f'Equation {i + 1}')

    # plot the true solutions
    plt.title("Plot of Residuals vs Network Input $t$", fontsize=20)
    plt.xlabel("Network Input $t$", fontsize=16)
    plt.ylabel('Residual Value', fontsize=16)
    plt.yscale('log')
    plt.grid()
    plt.legend()


def plot_transfer_loss(t_eval, transfer_loss):
    fig, axis = plt.subplots(1, 2, figsize=(16, 5))
    axis[0].plot(range(len(transfer_loss["loss"])), transfer_loss["loss"], label='Total Loss')
    axis[0].plot(range(len(transfer_loss["loss"])), transfer_loss["L_0"], label='L_0')
    axis[0].plot(range(len(transfer_loss["loss"])), transfer_loss["L_t_mean"], label='L_t mean')
    axis[0].set_yscale("log")
    axis[0].set_title("Transfer Loss", fontsize=20)
    axis[0].set_xlabel('Iterations', fontsize=16)
    axis[0].set_ylabel('Loss', fontsize=16)
    axis[0].tick_params(axis='x', labelsize=14)
    axis[0].tick_params(axis='y', labelsize=16)
    axis[0].grid()
    axis[0].legend(loc='best', fontsize=16)

    for i, loss in transfer_loss["L_t"].items():
        axis[1].plot(t_eval.detach().cpu().numpy(), loss, label=f'{i}')
    axis[1].set_yscale("log")
    axis[1].set_title("L_t term at during training", fontsize=20)
    axis[1].set_xlabel('t', fontsize=16)
    axis[1].set_ylabel('Loss', fontsize=16)
    axis[1].tick_params(axis='x', labelsize=14)
    axis[1].tick_params(axis='y', labelsize=16)
    axis[1].grid()
    axis[1].legend(loc='best', fontsize=16)
