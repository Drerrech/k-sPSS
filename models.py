import cvxpy as cp
import numpy as np
import torch

def get_quad_params(points, func_values):
    n = points.shape[1]

    c = cp.Variable(1)
    g = cp.Variable(n)
    H = cp.Variable((n, n), symmetric=True)

    objective = cp.Minimize(cp.sum_squares(H)) # ||H||F ^ 2
    
    # f(xi) = m(xi) = c + gTx + 1/2 xT H x
    constraints = [f_xi == c + g @ xi + 0.5 * cp.quad_form(xi, H) for xi, f_xi in zip(points, func_values)] # CVXPY should handle @...

    prob = cp.Problem(objective, constraints)
    prob.solve()
    # print("status:", prob.status)
    # print("H:", H.value)
    # print("points centered:", points)  # need >= 6 for n=2
    # print("func_vals:", func_values)
    return (c.value, g.value, H.value)


def solve_relative_quad_in_ball(c, g, H, delta): # assumes open-ball is centered at 0
    n = g.shape[0]

    x = cp.Variable(n)

    objective = cp.Minimize(c + g @ x + 0.5 * cp.quad_form(x, H))

    constraints = [cp.sum(x**2) <= delta**2]

    prob = cp.Problem(objective, constraints)
    f_tilda_x_hat = prob.solve()
    x_hat = x.value

    return torch.from_numpy(x_hat), f_tilda_x_hat


def get_quad_model_and_solution(points, func_values, delta): # assuming the first points is x_k, model will be cetnered at x_k
    # NOTE: will break if number of points is > (n+1)(n+2)/2 -> to handle disregard
    n = points.shape[1]
    max_num_points = (n+1)*(n+2)//2
    # trim
    points = points[:max_num_points]
    func_values = func_values[:max_num_points]

    x_k = points[0]
    # convert to numpy
    points = points.numpy()
    # center points at x_k
    rel_points = points - points[0]
    func_values = func_values.numpy()

    # step 1 - build a quadratic model using minimum Frobenius norm of the Hessian
    c, g, H = get_quad_params(rel_points, func_values)
    # Force H to be PSD - TODO: whatever that means
    eigvals = np.linalg.eigvalsh(H)
    if eigvals.min() < 0:
        H = H - eigvals.min() * np.eye(H.shape[0])

    c_t = torch.from_numpy(c)
    g_t = torch.from_numpy(g)
    H_t = torch.from_numpy(H)
#will this function f tilda still work even when we switched to relative points????
    def f_tilda(x): # NOTE: the model was built assuming p0 is 0 (centered at x_k)
        x_rel = x - x_k
        return c_t + g_t @ x_rel + 0.5 * x_rel @ H_t @ x_rel

    # step 2 - solve quadratic model and return point (in tensor form)
    rel_x_hat, f_tilda_x_hat = solve_relative_quad_in_ball(c, g, H, delta)
    x_hat = torch.from_numpy(points[0]) + rel_x_hat

    return x_hat, f_tilda_x_hat, g_t, f_tilda


def get_lin_model_and_solution(points, func_values, delta):
    # NOTE: solving a linear subproblem is the same as taking a delta step in -grad of f
    x_k = points[0]
    # p+1 points given
    p = points.shape[0] - 1
    # print(points)
    # print("p " + str(p))

    rel_points = points - points[0]

    # build delta f, x_bar will be the first point
    delta_f = -func_values[0] * torch.ones(p)
    for i in range(p):
        delta_f[i] += func_values[i+1]
    delta_f = delta_f.to(torch.float64)
    
    # D is [x1-x0, x2-x0 ... xp-x0]
    D = rel_points[1:] - rel_points[0]
    D_t_pinv = torch.linalg.pinv(D.t())
    
    # calculate grad and build linear model (already in torch)
    # print(D_t_pinv, delta_f)
    g = D_t_pinv @ delta_f
    c = func_values[0] - g@rel_points[0] # c + gxi = fi, so for x0: c = f0 - gx0

    def f_tilda(x):
        rel_x = x - x_k
        return c + g @ rel_x
    
    # solve quadratic model and return point (in tensor form)
    # as mentioned, step in -g with size delta
    x_hat = x_k - g * delta
    f_tilda_x_hat = f_tilda(x_hat)

    return x_hat, f_tilda_x_hat, g, f_tilda


def get_random_unit_D(p, n): # p - number of points, returns p randomly directed unit vectors
    vectors = torch.rand(p, n)
    vectors /= torch.sqrt(torch.pow(vectors, 2).sum(1)).view((-1, 1))
    return vectors
