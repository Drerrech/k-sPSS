import cvxpy as cp
import numpy as np
import torch
from scipy.optimize import minimize, brentq

def get_quad_params(points, func_values):
    # n = points.shape[1]

    # c = cp.Variable(1)
    # g = cp.Variable(n)
    # H = cp.Variable((n, n), symmetric=True)

    # objective = cp.Minimize(cp.sum_squares(H)) # ||H||F ^ 2
    
    # # f(xi) = m(xi) = c + gTx + 1/2 xT H x
    # constraints = [f_xi == c + g @ xi + 0.5 * (xi @ H @ xi) for xi, f_xi in zip(points, func_values)] # CVXPY should handle @...

    # prob = cp.Problem(objective, constraints)
    # prob.solve()
    # print("status:", prob.status)
    # # print("H:", H.value)
    # # print("points centered:", points)  # need >= 6 for n=2
    # # print("func_vals:", func_values)
    # return (c.value, g.value, H.value)



    n = points.shape[1]
    c = cp.Variable(1)
    g = cp.Variable(n)
    H = cp.Variable((n, n), symmetric=True)
    
    residuals = [f_xi - (c + g @ xi + 0.5 * (xi @ H @ xi)) 
                 for xi, f_xi in zip(points, func_values)]
    
    objective = cp.Minimize(
        cp.sum_squares(cp.hstack(residuals)) +  # fit the points
        1e-6 * cp.sum_squares(H)                # minimum ||H||_F
    )
    
    prob = cp.Problem(objective)
    prob.solve()
    return c.value, g.value, H.value


def get_quad_params_simplex_hessian(points, func_values):

    return c.value, g.value, H.value


def solve_relative_quad_in_ball(c, g, H, delta): # assumes open-ball is centered at 0
    n = g.shape[0]

    # x = cp.Variable(n)

    # objective = cp.Minimize(c + g @ x + 0.5 * cp.quad_form(x, H)) quad form is bad?

    # constraints = [cp.sum(x**2) <= delta**2]

    # prob = cp.Problem(objective, constraints)
    # f_tilda_x_hat = prob.solve()
    # x_hat = x.value

    # return torch.from_numpy(x_hat), f_tilda_x_hat


    # optim_result = minimize(fun=lambda x: c + g @ x + 0.5 * x @ H @ x, x0=np.zeros(n), jac=lambda x: g + H @ x, hess=lambda x: H, method='trust-exact', options={'initial_trust_radius': 0.5 * delta, 'max_trust_radius': delta})
    
    # print(torch.from_numpy(optim_result.x), optim_result.fun)
    
    # if not optim_result.success:
    #     raise Exception(f"scipy trust-exact could not solve the MBTR subproblem: {optim_result.message}")
    
    # return torch.from_numpy(optim_result.x), optim_result.fun


    def x_of_lam(lam):
        return np.linalg.solve(H + lam * np.eye(n), -g)
    
    eigvals = np.linalg.eigvalsh(H)
    lam_min = max(0, -eigvals.min() + 1e-8)  # ensure H + lam*I is PD
    
    # check if unconstrained solution is inside ball
    try:
        x_unc = x_of_lam(lam_min)
        if np.linalg.norm(x_unc) <= delta:
            return torch.from_numpy(x_unc), c + g @ x_unc + 0.5 * x_unc @ H @ x_unc
    except np.linalg.LinAlgError:
        pass
    
    # otherwise find lambda that places x on the boundary
    secular = lambda lam: np.linalg.norm(x_of_lam(lam)) - delta
    lam_star = brentq(secular, lam_min, lam_min + 1e6)
    x = x_of_lam(lam_star)
    return torch.from_numpy(x), c + g @ x + 0.5 * x @ H @ x



def get_quad_model_and_solution(raw_points, raw_func_values, raw_delta): # assuming the first points is x_k, model will be cetnered at x_k
    try:
        # NOTE: will break if number of points is > (n+1)(n+2)/2 -> to handle disregard
        n = raw_points.shape[1]
        max_num_points = (n+1)*(n+2)//2
        # trim
        print("GOT:", raw_points)
        points = raw_points[:max_num_points]
        func_values = raw_func_values[:max_num_points]

        print("TRIMMED:", points)
        print("trimmed func_values", func_values)

        x_k = points[0]
        # convert to numpy
        points = points.numpy()
        # center points at x_k
        rel_points = points - points[0]
        func_values = func_values.numpy()

        # TODO TESTING
        func_values = func_values.astype(np.float64)

        # step 1 - build a quadratic model using minimum Frobenius norm of the Hessian
        print("RELATIVE POINTS", rel_points)
        print("points dtype:", points.dtype)
        print("func_values dtype:", func_values.dtype)
        print("points type:", type(points))
        print("func_values type:", type(func_values))
        print("n:", n)
    
        c, g, H = get_quad_params(rel_points, func_values)

        print("HESSIAN", H)
        # # Force H to be PSD - ensures that the problem is convex but isn't that bs... it is, so i'll switch to scipy for solving the trust region
        # eigvals = np.linalg.eigvalsh(H)
        # if eigvals.min() < 0:
        #     H = H - eigvals.min() * np.eye(H.shape[0])

        c_t = torch.from_numpy(c)
        g_t = torch.from_numpy(g)
        H_t = torch.from_numpy(H)
        #will this function f tilda still work even when we switched to relative points????
        def f_tilda(x): # NOTE: the model was built assuming p0 is 0 (centered at x_k)
            x_rel = x - x_k
            return c_t + g_t @ x_rel + 0.5 * x_rel @ H_t @ x_rel

        # step 2 - solve quadratic model and return point (in tensor form)
        rel_x_hat, f_tilda_x_hat = solve_relative_quad_in_ball(c, g, H, raw_delta)
        x_hat = torch.from_numpy(points[0]) + rel_x_hat
        # raise Exception("some exception")
        return x_hat, f_tilda_x_hat, g_t, f_tilda
    except Exception as e: # for some reason, either building or solving the model failed, just do linear
        print(f"WARNING: failed to build quad model for MBTR, doing linear: {e}")
        print(raw_points)
        return get_lin_model_and_solution(raw_points, raw_func_values, raw_delta)



def get_lin_model_and_solution(points, func_values, delta):
    # NOTE: solving a linear subproblem is the same as taking a delta step in -grad of f
    x_k = points[0]
    # p+1 points given
    p = points.shape[0] - 1
    print(points)
    print(func_values)
    # print("p " + str(p))

    rel_points = points - points[0]

    # build delta f, x_bar will be the first point
    delta_f = -func_values[0] * torch.ones(p)
    for i in range(p):
        delta_f[i] += func_values[i+1]
    delta_f = delta_f.to(torch.float64)
    
    # D is [x1-x0, x2-x0 ... xp-x0]
    D_t = rel_points[1:] - rel_points[0]
    D_t_pinv = torch.linalg.pinv(D_t)
    
    # calculate grad and build linear model (already in torch)
    # print(D_t_pinv, delta_f)
    print("D_t_pinv:", D_t_pinv)
    print("delta_f", delta_f)
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
    vectors = (torch.rand(p, n) - 0.5) * 2  # uniform in [-1, 1]^n
    norms = torch.sqrt((vectors ** 2).sum(1))
    while (norms < 1e-8).any():
        mask = norms < 1e-8
        vectors[mask] = (torch.rand(mask.sum(), n) - 0.5) * 2
        norms = torch.sqrt((vectors ** 2).sum(1))
    return vectors
