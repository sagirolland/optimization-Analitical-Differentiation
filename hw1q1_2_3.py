import numpy as np


def numerical_diff(f, x, epsilon, *args):
    # Column vectors (n,1) broadcast with (n,) x to (n,n) and break f; keep 1d length n.
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    f0, _, _ = f(x, *args, True, False, False)
    grad = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        f_plus, _, _ = f(x + epsilon * ei, *args, True, False, False)
        f_minus, _, _ = f(x - epsilon * ei, *args, True, False, False)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    hess = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        _, g_plus, _ = f(x + epsilon * ei, *args, False, True, False)
        _, g_minus, _ = f(x - epsilon * ei, *args, False, True, False)
        g_plus = np.asarray(g_plus, dtype=float).ravel()
        g_minus = np.asarray(g_minus, dtype=float).ravel()
        hess[:, i] = (g_plus - g_minus) / (2 * epsilon)
    return f0, grad, hess
