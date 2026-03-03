import cvxpy as cp
import numpy as np
import torch

def get_quad_params(points, func_values):
    n = points.shape[0]

    c = cp.Variable(1)
    g = cp.Variable(n)
    H = cp.Variable((n, n))
    H.value

    objective = cp.Minimize(cp.sum_squares(H)) # ||H||F ^ 2
    
    # f(xi) = m(xi) = c + gTx + 1/2 xT H x
    constraints = [f_xi == c + g @ xi + 0.5 * xi @ H @ xi for xi, f_xi in zip(points, func_values)] # CVXPY should handle @...

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return (c.value, g.value, H.value)


def solve_quad_in_ball(c, g, H, x_k, delta):
    n = x_k.shape[0]

    x = cp.Variable(n)

    objective = cp.Minimize(c + x @ g + 0.5 * x @ H @ x)

    constraints = [np.sum((x - x_k)**2) < delta**2]

    prob = cp.Problem(objective, constraints)
    f_tilda_x_hat = prob.solve()
    x_hat = x.value

    return torch.from_numpy(x_hat), f_tilda_x_hat


def get_quad_model_and_solution(x_k, points, func_values, delta): # return a function
    # convert to numpy
    x_k = x_k.numpy()
    points = points.numpy()
    func_values = func_values.numpy()

    # step 1 - build a quadratic model using minimum Frobenius norm of the Hessian
    c, g, H = get_quad_params(points, func_values)

    c_t = torch.from_numpy(c)
    g_t = torch.from_numpy(g)
    H_t = torch.from_numpy(g)

    def f_tilda(x): # tensor
        return c_t + x @ g_t + 0.5 * x @ H_t @ x

    # step 2 - solve quadratic model and return point (in tensor form)
    x_hat, f_tilda_x_hat = solve_quad_in_ball(c, g, H, x_k, delta)

    return x_hat, f_tilda_x_hat, f_tilda