import time
import numpy as np
import torch


# function to compute H and dH_dt components necessary for transfer learning
def compute_H_and_dH_dt(min_x, max_x, trained_model, num_equations, hid_lay, size, dev):

    start_time = time.time()

    # generate a set of times to evaluate with
    rng = np.random.default_rng()
    t_eval = torch.arange(min_x, max_x, 0.001, requires_grad=True, device=dev).double()
    t_eval = t_eval[np.concatenate(([0], rng.choice(range(1, len(t_eval)), size=size - 1, replace=False)))]
    t_eval = t_eval.reshape(-1, 1)
    t_eval, _ = t_eval.sort(dim=0)

    # forward pass with t_eval to extract H
    _, H = trained_model(t_eval)
    # reshape "H" to batch_size X num_equations X d // num_equations
    H = H.reshape(-1, num_equations, hid_lay[-1] // num_equations)
    H = torch.cat((torch.ones(len(t_eval), num_equations, 1, device=dev), H), 2)

    # forward pass with t = 0 to extract H_0
    _, H_0 = trained_model(torch.tensor([[0.]], dtype=torch.float64,
                                        requires_grad=True, device=dev))
    # reshape "H_0" to batch_size X num_equations X d // num_equations
    H_0 = H_0.reshape(-1, num_equations, hid_lay[-1] // num_equations)
    H_0 = torch.cat((torch.ones(1, num_equations, 1, device=dev), H_0), 2).squeeze()
    H_0 = H_0.unsqueeze(dim=0) if num_equations == 1 else H_0

    # compute dH_dt
    _, H_orig = trained_model(t_eval)
    dH_dt = [torch.autograd.grad(H_orig[:, i], t_eval, grad_outputs=torch.ones_like(H_orig[:, i]), create_graph=True)[0] for i in range(H_orig.shape[1])]
    dH_dt = torch.hstack(dH_dt)
    dH_dt_new = dH_dt.reshape(-1, num_equations, hid_lay[-1] // num_equations)
    dH_dt_new = torch.cat((torch.zeros(len(t_eval), num_equations, 1, device=dev), dH_dt_new), 2)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time to compute H and dH_dt: {total_time: .3f} seconds")

    return H.double(), H_0.double(), dH_dt_new.double(), t_eval.double()


# function to analytically compute W_0 (transfer learned weights)
def analytically_compute_weights(dH_dt, H, H_0, t_eval, v, A, force, verbose=True):

    start_time = time.time()

    # compute dH_dt * dH_dt.T
    dH_dt_times_dH_dt_T = torch.matmul(dH_dt.mT, dH_dt)

    # compute dH_dt * A * H
    dH_dt_times_A_times_H = torch.matmul(torch.matmul(dH_dt.mT, A), H)

    # compute H.T * A.T * dH_dt
    H_times_A_T_times_dH_dt = torch.matmul(torch.matmul(H.mT, A.T), dH_dt)

    # compute H.T * A.T * A * H
    H_T_times_A_T_times_A_times_H = torch.matmul(torch.matmul(torch.matmul(H.mT, A.T), A), H)

    # compute the "summation portion" of the M matrix
    M_sum_terms = dH_dt_times_dH_dt_T + dH_dt_times_A_times_H + H_times_A_T_times_dH_dt + H_T_times_A_T_times_A_times_H
    M_sum_terms = M_sum_terms.sum(axis=0)
    M_sum_terms = M_sum_terms / len(t_eval)

    # compute H_0.T * H_0
    H_0_T_times_H_0 = torch.matmul(H_0.mT, H_0)

    # compute the "M" matrix and invert it
    M = M_sum_terms + H_0_T_times_H_0
    M_inv = torch.linalg.pinv(M)

    # compute dH_dt * force
    if callable(force):
        force = force(t_eval.squeeze())
    dH_dt_times_force = torch.matmul(dH_dt.mT, force if force.shape[-1] == 1 else force.unsqueeze(2))
    # dH_dt_times_force = torch.matmul(dH_dt.mT, torch.cat(force(t_eval.detach().cpu()), axis=1).unsqueeze(2).to(dev).double() if is_force_time_dep else force)

    # compute H * A.T * force
    H_times_A_T_times_f = torch.matmul(torch.matmul(H.mT, A.T), force if force.shape[-1] == 1 else force.unsqueeze(2))
    # H_times_A_T_times_f = torch.matmul(torch.matmul(H.mT.double(), A.T.double()), torch.cat(force(t_eval.detach().cpu()), axis=1).unsqueeze(2).to(dev).double() if is_force_time_dep else force.double())

    # sum the force-contributing terms and add them to H_0.T * v
    force_terms = dH_dt_times_force + H_times_A_T_times_f
    force_terms = force_terms.sum(axis=0)
    force_terms = force_terms / len(t_eval)
    rhs_terms = force_terms + torch.matmul(H_0.T, v)

    # compute the output weights by W_out = M ^ -1 * H_0 * u_0
    W_out = torch.matmul(M_inv, rhs_terms)

    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"Time to compute weights (given H and dH_dt): {total_time: .3f} seconds")
    return M_inv, W_out, force_terms, total_time


def compute_M_inv(dH_dt, H, H_0, t_eval, A):

    # compute dH_dt * dH_dt.T
    dH_dt_times_dH_dt_T = torch.matmul(dH_dt.mT, dH_dt)

    # compute dH_dt * A * H
    dH_dt_times_A_times_H = torch.matmul(torch.matmul(dH_dt.mT, A), H)

    # compute H.T * A.T * dH_dt
    H_times_A_T_times_dH_dt = torch.matmul(torch.matmul(H.mT, A.T), dH_dt)

    # compute H.T * A.T * A * H
    H_T_times_A_T_times_A_times_H = torch.matmul(torch.matmul(torch.matmul(H.mT, A.T), A), H)

    # compute the "summation portion" of the M matrix
    M_sum_terms = dH_dt_times_dH_dt_T + dH_dt_times_A_times_H + H_times_A_T_times_dH_dt + H_T_times_A_T_times_A_times_H
    M_sum_terms = M_sum_terms.sum(axis=0)
    M_sum_terms = M_sum_terms / len(t_eval)

    # compute H_0.T * H_0
    H_0_T_times_H_0 = torch.matmul(H_0.mT, H_0)

    # compute the "M" matrix and invert it
    M = M_sum_terms + H_0_T_times_H_0
    M_inv = torch.linalg.pinv(M)

    return M_inv


def compute_W_with_IC(M_inv, force_terms, v, H_0):

    start_time = time.time()

    rhs_terms = force_terms + torch.matmul(H_0.T, v)
    #rhs_terms = force_terms + torch.matmul(H_0.T, v)
    # compute the output weights by W_out = M ^ -1 * H_0 * u_0
    W_out = torch.matmul(M_inv, rhs_terms)

    total_time = time.time() - start_time
    return W_out, total_time


def compute_force_term(t_eval, A, force, H, dH_dt):
    # compute dH_dt * force
    if callable(force):
        force = force(t_eval.squeeze())
    # compute dH_dt * force
    # dH_dt_times_force = torch.matmul(dH_dt.mT, force if force.shape[-1] == 1 else force.unsqueeze(2))
    if force.shape[0] != t_eval.shape[0]:
        force = force.unsqueeze(0)
    else:
        force = force.unsqueeze(2)
    dH_dt_times_force = torch.matmul(dH_dt.mT, force)
    # dH_dt_times_force = torch.matmul(dH_dt.mT, torch.cat(force(t_eval.detach().cpu()), axis=1).unsqueeze(2).to(dev).double() if is_force_time_dep else force)

    # compute H * A.T * force
    H_times_A_T_times_f = torch.matmul(torch.matmul(H.mT, A.T), force)
    #H_times_A_T_times_f = torch.matmul(torch.matmul(H.mT, A.T), force if force.shape[-1] == 1 else force.unsqueeze(2))
    # H_times_A_T_times_f = torch.matmul(torch.matmul(H.mT.double(), A.T.double()), torch.cat(force(t_eval.detach().cpu()), axis=1).unsqueeze(2).to(dev).double() if is_force_time_dep else force.double())

    # sum the force-contributing terms and add them to H_0.T * v
    force_terms = dH_dt_times_force + H_times_A_T_times_f
    force_terms = force_terms.sum(axis=0)
    force_terms = force_terms / len(t_eval)
    # return force_terms.squeeze()
    return force_terms


def compute_W_with_IC_and_force(t_eval, A, v, force, H, H_0, dH_dt, M_inv):
    start_time = time.time()
    force_terms = compute_force_term(t_eval, A, force, H, dH_dt)
    W_out, _ = compute_W_with_IC(M_inv, force_terms, v, H_0)
    total_time = time.time() - start_time
    return W_out, total_time


