import time
import torch
from src.transfer_learning import compute_M_inv, compute_W_with_IC_and_force


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
    x0_initials = compute_initial(IC[0], beta, p)
    v0_initials = compute_initial(IC[1], beta, p)
    IC_pertubation = torch.tensor([[x0_initials], [v0_initials]], device=dev)

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
                                           IC_pertubation.detach().cpu().numpy(),
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
                                                                  IC_pertubation.detach().cpu(),
                                                                  A.detach().cpu(), fi_numerical))
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
