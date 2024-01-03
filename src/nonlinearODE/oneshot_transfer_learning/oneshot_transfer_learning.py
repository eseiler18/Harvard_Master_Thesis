import torch
import numpy as np


def M_matrix(x, params, hidden_state_vmap, dHdx, A):
    """this function returns the matrix M

    Args:
        x (Tensot): x is the batch of time
        params: parameters of the neural network
        A (np.array): New linear system of equation 

    Returns:
        _type_: _description_
    """
    # compute the batched Hidden states
    Hs = hidden_state_vmap(x, params).detach().numpy()
    # compute the batched gradients of the hidden states
    H_primes = dHdx(x, params).detach().numpy()
    # compute the first part of the matrix M
    results = []
    for i in range(Hs.shape[0]):
        h = Hs[i].reshape(2, 256)
        # h = np.hstack((np.ones((h.shape[0], 1)), h)) ##add the bias column
        hp = H_primes[i].reshape(2, 256)
        # hp = np.hstack((np.zeros((hp.shape[0], 1)), hp)) ##add the bias column
        B = hp.T @ A @ h
        matrix = hp.T @ hp + B + B.T + h.T @ A.T @ A @ h
        results.append(matrix)
    results = np.array(results)
    results = np.mean(results, axis=0)

    # compute the second part of the matrix M
    h0 = hidden_state_vmap(torch.Tensor([0]), params).detach().numpy()[0].reshape(2, 256)
    return results + h0.T @ h0


def compute_W(x, params, hidden_state_vmap, dHdx, A, force_function, IC, Minv):
    """Compute the analytic W that minimizes the loss function

    Args:
        x (_type_): _description_
        params (_type_): _description_
        hidden_state_vmap (_type_): _description_
        dHdx (_type_): _description_
        A (_type_): _description_
        force_function (_type_): _description_
        IC (_type_): _description_
        Minv (_type_): _description_

    Returns:
        _type_: _description_
    """
    h0 = hidden_state_vmap(torch.Tensor([0]), params).detach().numpy()[0].reshape(2, 256)
    u0 = np.array([[IC[0]], [IC[1]]])

    Hs = hidden_state_vmap(x, params).detach().numpy()
    H_primes = dHdx(x, params).detach().numpy()

    fs = force_function(x)
    if type(fs) is not np.ndarray:
        fs = fs.numpy()
    results = []
    for i in range(Hs.shape[0]):
        h = Hs[i].reshape(2, 256)
        hp = H_primes[i].reshape(2, 256)
        f = fs[i].reshape(2, 1)
        results.append((hp.T + h.T @ A.T) @ f)
    results = np.array(results)
    results = np.mean(results, axis=0) + h0.T @ u0
    W = Minv @ results
    return W


def one_shot_TL(x, params, hidden_state_vmap, dHdx,
                A, force_function, IC, reparametrization):
    """This function gives you the solution of an unseen ODE using one-shot transfer learning

    Args:
        x (Tensor): batched tensors of inputs to compute the M and W
        params : is the trained network's parameters
        hidden_state_vmap (_type_): _description_
        dHdx (_type_): _description_
        A (_type_): _description_
        force_function (_type_): _description_
        IC (_type_): _description_
        reparametrization (_type_): _description_

    Returns:
        _type_: _description_
    """
    M = M_matrix(x, params, hidden_state_vmap, dHdx, A)
    M = M + np.diag([0.000001 for i in range(M.shape[0])])
    Minv = np.linalg.pinv(M)
    print(np.linalg.cond(M))

    # compute the analytic W
    W = compute_W(x, params, hidden_state_vmap, dHdx, A, force_function, IC, Minv)
    Hs = hidden_state_vmap(x, params).detach().numpy()
    analytic_result = []
    for i in range(x.shape[0]):
        h = Hs[i].reshape(2, 256)
        analytic_result.append(h @ W)
    analytic_result = np.array(analytic_result)
    if reparametrization:
        H0 = hidden_state_vmap(torch.Tensor([0]), params).detach().numpy()[0].reshape(2, 256)
        u0 = H0 @ W
        reparametrization_factor = (np.tile(([IC[0], IC[1]] - u0.T).T, x.shape[0]) * np.exp(-np.array(x))).T[..., np.newaxis]
        analytic_result = analytic_result + reparametrization_factor
        analytic_result = analytic_result[:, :, 0]
    return analytic_result, W


def loss_TL(x, params, hidden_state_vmap, dHdx, A, force_function, IC, W):
    """this function computes the loss of the PINN after one-shot TL

    Args:
        x (_type_): _description_
        params (_type_): _description_
        hidden_state_vmap (_type_): _description_
        dHdx (_type_): _description_
        A (_type_): _description_
        force_function (_type_): _description_
        IC (_type_): _description_
        W (_type_): _description_

    Returns:
        _type_: _description_
    """
    H_prime = dHdx(x, params).detach().numpy()
    H = hidden_state_vmap(x, params).detach().numpy()
    # colocation loss
    colo = []
    for i in range(H.shape[0]):
        hp = H_prime[i].reshape(2, 256)
        h = H[i].reshape(2, 256)
        colo.append((hp + A @ h) @ W - np.array(force_function(x)[i].reshape(2, 1)))
    colo = np.array(colo)
    loss_colocation = (colo**2).sum() / H.shape[0]
    # BC loss
    H0 = hidden_state_vmap(torch.Tensor([0]), params).detach().numpy()[0].reshape(2, 256)
    u0 = H0 @ W
    loss_BC = ((u0 - np.array([[IC[0]], [IC[1]]]))**2).sum()
    return loss_colocation + loss_BC, loss_colocation, loss_BC
