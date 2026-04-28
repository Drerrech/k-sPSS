import models

import torch

class MBTR_k_fail:
    def __init__(self, x, bb_k_fail_wrapper, delta, mu, eta, gamma, eps_stop, prediction_software, log_file_path, preferred_model_order=2, use_opportunistic_cpu_exploitation=True, opportunistic_cpu_exploitation_manual_point_limit=1e12): # f_omega will not be used as no problems in the test set have contraints
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

        self.use_opportunistic_cpu_exploitation = use_opportunistic_cpu_exploitation
        self.opportunistic_cpu_exploitation_manual_point_limit = opportunistic_cpu_exploitation_manual_point_limit

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
        

        k_fail_predicted = self.prediction_software.predict_k() # predict k
        if self.preferred_model_order == 1: # NOTE p+1 points, not p!
            p_total = self.n + k_fail_predicted # linear model +k failed points
        else:
            p_total = (self.n+1)*(self.n+2)//2 + k_fail_predicted - 1
        
        if not self.use_opportunistic_cpu_exploitation:
            # build the full model, n_1_batch called later as a separate batch NOTE: must also update normal batch count when updating n_1_batch!

            # 1.1 - build points to build the model
            # TODO: test for oversupply - though get model and sltn methods might take care of this already?
            # print("P:", p)
            points = self.x + self.delta*models.get_random_unit_D(p_total+0, self.n) # NOTE: without x_k for now, will add later (so we don't loose it)

            # 1.2 get function value at points
            f_vals, completed, actual_batch_calls, actual_k = self.bb_k_fail_wrapper.batch_call(points)
            points = points[completed] # NOTE: important step
            
            # points only consistend of random_D, so adding x_k as first element back in, also adding its' function
            points = torch.cat((self.x.unsqueeze(0), points)) # add x_k back in now p+1 points
            # print(self.cur_f_val)
            f_vals = torch.cat((torch.tensor(self.cur_f_val).unsqueeze(0), f_vals)) # add f(x_k)
            
            selected_order = 1
            if points.shape[0] > self.n + k_fail_predicted: # capable of a poor quad model (n+1 + k - linear, (n+1)(n+2)/2 + k - quad)
                selected_order = 2

            self.prediction_software.add_actual_k(-1)
            # update counters NOTE: not all of them are update here (self.n_1_batch_calls)
            self.n_function_calls = self.bb_k_fail_wrapper.p_reuse.get_n_f_evals()
            self.n_failed_function_calls += points.shape[0] - completed.shape[0]
            self.n_batch_calls += actual_batch_calls

            # 1.3 build model
            x_hat, f_tilda_x_hat, g_tilda, f_tilda = None, None, None, None
            if selected_order == 1:
                x_hat, f_tilda_x_hat, g_tilda, f_tilda = models.get_lin_model_and_solution(points, f_vals, self.delta)
                message += " | used linear model with " + str(p_total+1) + " points"
            elif selected_order == 2:
                x_hat, f_tilda_x_hat, g_tilda, f_tilda = models.get_quad_model_and_solution(points, f_vals, self.delta)
                message += " | used quadratic model with " + str(p_total+1) + " points"
        
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
                f_x_hat = self.bb_k_fail_wrapper.p_reuse.evaluate(x_hat)
                self.n_batch_calls += 1
                self.n_1_batch_calls += 1
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
        
        else: # exploitation
            num_cpus = self.bb_k_fail_wrapper.num_cpus

            additional_points = self.x + self.delta*models.get_random_unit_D(num_cpus, self.n) # NOTE: without x_k for now, will add later (so we don't loose it)
            additional_f_vals, completed, actual_batch_calls, actual_k = self.bb_k_fail_wrapper.batch_call(additional_points)
            
            
            self.n_function_calls = self.bb_k_fail_wrapper.p_reuse.get_n_f_evals()
            self.n_failed_function_calls += additional_points.shape[0] - completed.shape[0]
            self.n_batch_calls += actual_batch_calls
            
            additional_points = additional_points[completed] # NOTE: important step
            
            # starting with [x0, num_cpus points]
            points = torch.cat((self.x.unsqueeze(0), additional_points))
            # print(points, additional_points)
            f_vals = torch.cat((torch.tensor(self.cur_f_val).unsqueeze(0), additional_f_vals))
            # print("STARTING", points.shape, f_vals.shape)

            # keep adding model points + checking point -> see if passes accuracy check
            got_sufficient_acc = False
            starting_iteration_done = False
            iterate_success = False
            
            while (points.shape[0] < p_total+1 and points.shape[0] < self.opportunistic_cpu_exploitation_manual_point_limit) or not starting_iteration_done:
                # print("POITNS AND LIMIT:", points.shape[0], p_total)
                starting_iteration_done = True

                # try to build model with what we have right no (no additional), and check
                selected_order = 1
                if points.shape[0] > self.n + k_fail_predicted: # capable of a poor quad model (n+1 + k - linear, (n+1)(n+2)/2 + k - quad)
                    selected_order = 2
                
                x_hat, f_tilda_x_hat, g_tilda, f_tilda = None, None, None, None
                if selected_order == 1:
                    x_hat, f_tilda_x_hat, g_tilda, f_tilda = models.get_lin_model_and_solution(points, f_vals, self.delta)
                    # print("LIN", points.shape, f_vals.shape)
                    message += " | trying linear partial model with " + str(points.shape[0]) + " points"
                elif selected_order == 2:
                    x_hat, f_tilda_x_hat, g_tilda, f_tilda = models.get_quad_model_and_solution(points, f_vals, self.delta)
                    print("QUAD", points.shape, f_vals.shape)
                    message += " | trying quadratic partial model with " + str(points.shape[0]) + " points"
                
                # 2 - model accuracy checks
                g_tilda_norm = torch.linalg.vector_norm(g_tilda)
                # 2.a - success?
                if self.delta < self.eps_stop and g_tilda_norm < self.eps_stop:
                    message += " | DECALRED SUCCESS"
                    self.log_current(message=message)
                    return 1 # on success return 1, on fail (step with no succes) return 0 by default
                
                # evaluate additional points: (num_cpus-(k+1)) new model points + (k+1) f_x_hat points so at least one makes it out
                # if f_x_hat does not make it out due to underpredicted k, one full batch will be used up for f_x_hat
                additional_points = self.x + self.delta*models.get_random_unit_D(num_cpus - (k_fail_predicted+1), self.n)
                additional_points = torch.cat([additional_points, torch.ones((k_fail_predicted+1, self.n)) * x_hat]) # idxs [num_cpus-(k+1), num_cpus-1] are x_hat points
                
                additional_f_vals, completed, actual_batch_calls, actual_k = self.bb_k_fail_wrapper.batch_call(additional_points)
                self.prediction_software.add_actual_k(-1)
                self.n_function_calls = self.bb_k_fail_wrapper.p_reuse.get_n_f_evals()
                self.n_failed_function_calls += additional_points.shape[0] - completed.shape[0]
                self.n_batch_calls += actual_batch_calls

                # check if any x_hat points made it back
                found_x_hat = False
                f_x_hat = None
                for i, idx_completed in enumerate(completed):
                    if idx_completed >= num_cpus - (k_fail_predicted+1): # point was x_hat
                        found_x_hat = True
                        f_x_hat = additional_f_vals[i].item()
                        break
                
                if not found_x_hat: # will have to use an additional batch call
                    message += " FAILED TO GET X_HAT, using additional batch call"
                    # print("FAILED TO GET X_HAT, using additional batch call")
                    f_x_hat = self.bb_k_fail_wrapper.p_reuse.evaluate(x_hat)
                    self.n_batch_calls += 1
                    self.n_1_batch_calls += 1
                
                # get additional model points, will be cat-ed if needed in 2.b
                additional_model_points = []
                additional_model_f_vals = []
                for i, idx_completed in enumerate(completed):
                    if idx_completed < num_cpus - (k_fail_predicted+1): # model point
                        additional_model_points.append(additional_points[idx_completed])
                        additional_model_f_vals.append(additional_f_vals[i])
                
                additional_model_points = torch.stack(additional_model_points)
                additional_model_f_vals = torch.stack(additional_model_f_vals)

                # 2.b - insufficient accuracy?
                if self.delta > self.mu * g_tilda_norm:
                    # NOTE: instead of self.delta *= self.gamma, try again with more points
                    points = torch.cat((points, additional_model_points)) 
                    f_vals = torch.cat((f_vals, additional_model_f_vals))
                    continue
                else:
                    got_sufficient_acc = True
                
                # 4 - candidate test and trust region update
                rho = (self.cur_f_val - f_x_hat) / (f_tilda(self.x) - f_tilda_x_hat)

                if rho > self.eta: # iterate success
                    self.x = x_hat
                    self.cur_f_val = f_x_hat
                    self.delta /= self.gamma
                    message += " | iterate success"
                    iterate_success = True
                    break
                else: # iterate failure - instead of the normal delta decrease try more points for the model
                    points = torch.cat((points, additional_model_points)) 
                    f_vals = torch.cat((f_vals, additional_model_f_vals))
                    message += " | (iterate failure) trying more points before decreasing delta"
                    continue
            
            if (not got_sufficient_acc) or (not iterate_success): # 2b - insufficient accuracy OR 4 - in the iterations success never happened, so decrease delta
                message += " | exploitation failed, decreasing delta"
                self.delta *= self.gamma

        # update params and log
        message += " | actual_k=" + str(actual_k)
        self.k += 1
        self.log_current(message=message)
