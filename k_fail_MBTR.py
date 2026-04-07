import models

import torch

class MBTR_k_fail:
    def __init__(self, x, bb_k_fail_wrapper, delta, mu, eta, gamma, eps_stop, prediction_software, log_file_path, preferred_model_order=2): # f_omega will not be used as no problems in the test set have contraints
        self.bb_k_fail_wrapper = bb_k_fail_wrapper
        self.x = x
        self.cur_f_val = self.bb_k_fail_wrapper.p_reuse.evaluate(self.x) # using hashing (or not at step 0) get the current value of f
        self.delta = delta
        self.mu = mu
        self.eta = eta
        self.gamma = gamma
        self.eps_stop = eps_stop
        self.prediction_software = prediction_software
        self.preferred_model_order = preferred_model_order

        self.n = x.shape[0]

        self.log_file_path = log_file_path # path to .txt where info should be stored
        open(log_file_path, "w").close() # clear log file
        with open(self.log_file_path, "a") as _f: # add columns
            _f.write("k       | x" + 78*" " + "| f(x)               | delta          | n_function_calls, n_failed_function_calls, n_1_batch_calls, n_batch_calls | message\n")
            # k:8, x:80, f(x):16, delta:16, delta_m:16, f_calls:18
        
        # counters
        self.k = 0
        self.n_function_calls = 0
        self.n_failed_function_calls = 0
        self.n_1_batch_calls = 0
        self.n_batch_calls = 0
    
    def log_current(self, message=""): # log information about the current state
        s = f"{self.k:8}|{str([round(i, 16) for i in self.x.tolist()]):80}|{self.cur_f_val:20.16f}|{self.delta:16.8f}|{str([self.n_function_calls, self.n_failed_function_calls, self.n_1_batch_calls, self.n_batch_calls]):75}| {message}"
        with open(self.log_file_path, "a") as _f:
            _f.write(s + "\n")
    
    def step_default(self):
        message = ""

        # 1 - model
        # for a quadratic model we need (n+1)(n+2)/2 points
        # for a linear only (n+1)
        
        # 1.1 - build points to build the model
        
        k_fail_predicted = self.prediction_software.predict_k() # predict k
        if self.preferred_model_order == 1: # NOTE p+1 points, not p!
            p = self.n + k_fail_predicted # linear model +k failed points TODO: as discussed with Clement this can fail
        else:
            p = (self.n+1)*(self.n+2)//2 + k_fail_predicted - 1 # TODO: as discussed with Clement this can fail
        # TODO: test for oversupply
        # print("P:", p)
        points = self.x + self.delta*models.get_random_unit_D(p+0, self.n) # NOTE: without x_k for now, will add later (so we don't loose it)

        # 1.2 get function value at points
        f_vals, completed, actual_batch_calls = self.bb_k_fail_wrapper.batch_call(points)
        actual_k = points.shape[0] - completed.shape[0]
        # print("raw points", points)
        # print("completed idx", completed)
        # print("x_k", self.x)
        points = points[completed] # NOTE: important step
        
        # points only consistend of random_D, so adding x_k as first element back in, also adding its' function
        points = torch.cat((self.x.unsqueeze(0), points)) # add x_k back in now p+1 points
        f_vals = torch.cat((torch.tensor(self.cur_f_val).unsqueeze(0), f_vals)) # add f(x_k)
        
        # print("f(x_k)", self.cur_f_val)
        # print("POINTS:", points)
        # print("func vals", f_vals)

        selected_order = 1
        if points.shape[0] > self.n: # capable of a poor quad model (n+1 - linear, (n+1)(n+2)/2 - quad)
            selected_order = 2

        self.prediction_software.add_actual_k(actual_k)
        # update counters NOTE: not all of them are update here (self.n_1_batch_calls)
        self.n_function_calls = self.bb_k_fail_wrapper.p_reuse.get_n_f_evals()
        self.n_failed_function_calls += actual_k
        self.n_batch_calls += actual_batch_calls

        # 1.3 build model
        x_hat, f_tilda_x_hat, g_tilda, f_tilda = None, None, None, None
        if selected_order == 1:
            x_hat, f_tilda_x_hat, g_tilda, f_tilda = models.get_lin_model_and_solution(points, f_vals, self.delta)
            message += " | used linear model with " + str(p+1) + " points"
        elif selected_order == 2:
            x_hat, f_tilda_x_hat, g_tilda, f_tilda = models.get_quad_model_and_solution(points, f_vals, self.delta)
            message += " | used quadratic model with " + str(p+1) + " points"
        
        # 2 - model accuracy checks
        g_tilda_norm = torch.linalg.vector_norm(g_tilda)
        # 2.a - success?
        if self.delta < self.eps_stop and g_tilda_norm < self.eps_stop:
            message += " | DECALRED SUCCESS"
            self.log_current(message=message)
            return 1 # on success return 1, on fail (step with no succes) return 0 by default
        # 2.b - insufficient accuracy?
        if self.delta > self.mu * g_tilda_norm:
            self.delta *= self.gamma
            # go to 5
        
        else:
            # 3 - trust region problem
            # already solved

            # 4 - candidate test and trust region update
            # TODO: FOR NOW ASSUME k=0, becuase what if f(x_hat) fails
            f_x_hat = self.bb_k_fail_wrapper.p_reuse.evaluate(x_hat) # TODO: counters
            self.n_1_batch_calls += 1 # update counter
            rho = (self.cur_f_val - f_x_hat) / (f_tilda(self.x) - f_tilda_x_hat) # NOTE: CHEATING!!!

            if rho > self.eta: # iterate success
                self.x = x_hat
                self.cur_f_val = f_x_hat
                self.delta /= self.gamma
                message += " | iterate success"
            else: # iterate failure
                self.delta *= self.gamma
                message += " | iterate failure"
        
        # step 5 - termination test
        # not included

        # update params and log
        message += " | actual_k=" + str(actual_k)
        self.k += 1
        self.log_current(message=message)
