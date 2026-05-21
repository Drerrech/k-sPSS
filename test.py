import pycutest
from tqdm import tqdm

# # Find unconstrained, variable-dimension problems
probs = pycutest.find_problems(objective="sum of squares other", constraints="unconstrained") # the query for dim contrain simply doesn't work, so implemented manually
max_dim = 1000


# for i, p_name in enumerate(tqdm(probs)):
#     try:
#         p = pycutest.import_problem(p_name)
#         if p.n > max_dim:
#             continue
#     except Exception as e:
#         continue


max_dim = 1000
raw_problem_selection = pycutest.find_problems(objective="sum of squares other", constraints="unconstrained") # the query for dim contrain simply doesn't work, so implemented manually
# load problems into self.problems and load the pytorch compativle functions into self.problem_functions
for p_name in tqdm(raw_problem_selection):
    try:
        p = pycutest.import_problem(p_name)
        if p.n > max_dim:
            continue
        
        #self.problems.append(p)
        def f(x):
            return p.obj(x.numpy())
        #self.problem_functions.append(f)
    except Exception as e:
        # if print_load_status:
        #     print(f"{p_name} caused an exception while loading")
        continue

# # Properties of problem ROSENBR
# print(pycutest.problem_properties('TRO21X5'))



# problem_collection = BB_wrapper.BB_cutest_collection()

# ros_idx = problem_collection.problems.index('ROSENBR')
# print(ros_idx)

# p = problem_collection.get_problem(888)
# f = problem_collection.get_problem_function(888)

# print("Rosenbrock function in %gD" % p.n)

# iters = 0

# x = torch.tensor([0.9999957, 0.99999139], dtype=torch.float64)

# result1 = f(x)
# result2 = p.obj(x.numpy())

# print(f"f(x):     {result1:.17f}")
# print(f"p.obj(x): {result2:.17f}")
# print(f"Equal: {result1 == result2}")
# print(f"Close: {np.isclose(result1, result2)}")

# print(type(result1), type(result2))  # Both should be float
# print(x.numpy().dtype)  # Should be float64
# f, g = p.obj(x, gradient=True)  # objective and gradient
# H = p.hess(x)  # Hessian

# while iters < 100 and np.linalg.norm(g) > 1e-10:
#     print("Iteration %g: objective value is %g with norm of gradient %g at x = %s" % (iters, f, np.linalg.norm(g), str(x)))
#     s = np.linalg.solve(H, -g)  # Newton step
#     x = x + s  # used fixed step length
#     f, g = p.obj(x, gradient=True)
#     H = p.hess(x)
#     iters += 1

# print("Found minimum x = %s after %g iterations" % (str(x), iters))
# print("Done")
