import polytope_k_sPSS

import torch

def select_k_spss(predicted_k_fail, num_dim, delta):
    return polytope_k_sPSS.default_k_sPSS(num_dim, predicted_k_fail) * delta

def get_random_rotation_matrix(n):
    i_hat = torch.randint(0, n-1, (1,)) # [0, n-1) (idx starts at 0)
    j_hat = torch.randint(i_hat, n, (1,))
    theta = torch.rand((1,)) * 2 * torch.pi
    
    m = torch.eye(n)

    m[i_hat, i_hat] = torch.cos(theta)
    m[j_hat, j_hat] = torch.cos(theta)

    m[i_hat, j_hat] = torch.sin(theta)
    
    m[j_hat, i_hat] = -torch.sin(theta)

    return m


class GPS_k_fail:
    def __init__(self, x, bb_k_fail_wrapper, delta, tao, prediction_software, log_file_path, use_opportunistic_cpu_exploitation=True): # f_omega will not be used as no problems in the test set have contraints
        self.x = x
        self.bb_k_fail_wrapper = bb_k_fail_wrapper
        self.delta = delta
        self.tao = tao
        self.prediction_software = prediction_software

        self.use_opportunistic_cpu_exploitation = use_opportunistic_cpu_exploitation

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

        self.cur_f_val = self.bb_k_fail_wrapper.p_reuse.evaluate(self.x) # using hashing (or not at step 0) get the current value of f
    
    def log_current(self, message=""): # log information about the current state
        s = f"{self.k:8}|{str([round(i, 16) for i in self.x.tolist()]):80}|{self.cur_f_val:20.16f}|{self.delta:16.8f}|{str([self.n_function_calls, self.n_failed_function_calls, self.n_1_batch_calls, self.n_batch_calls]):75}| {message}"
        with open(self.log_file_path, "a") as _f:
            _f.write(s + "\n")
    
    def step_default(self, random_rotate=True): # delta is updated nomally
        # 2 - poll
        k_fail_predicted = self.prediction_software.predict_k() # predict k
        
        P = select_k_spss(k_fail_predicted, self.x.shape[0], self.delta) # tensor of points to eval
        # rotate the matrix (optional)
        if random_rotate:
            P = (get_random_rotation_matrix(self.x.shape[0]) @ P.T).T
        

        actual_batch_calls = 0
        actual_k = 0
        total_failed = 0
        
        if not self.use_opportunistic_cpu_exploitation:
            f_vals, completed, actual_batch_calls, actual_k = self.bb_k_fail_wrapper.batch_call(P)

            total_failed = P.shape[0] - completed.shape[0]

            # complete polling
            min_f_val_idx = torch.argmin(f_vals) # note, this is an idx of returned values, not an index of P
            if f_vals[min_f_val_idx] < self.cur_f_val: # found a better value -> update point and tao
                self.x = P[completed[min_f_val_idx]]
                self.cur_f_val = f_vals[min_f_val_idx]
                self.delta = self.delta / self.tao
            else: # failed to find a better point
                self.delta = self.delta * self.tao
        else:
            # reorder P randomly so calling explores in all directions each batch
            P = P[torch.randperm(P.size(0))]
            # print(P)

            num_cpus = self.bb_k_fail_wrapper.num_cpus
            # call 1 batch, check for improbement, if not call again until checked all points in P
            for i in range(0, P.size(0), num_cpus):
                # print("selection:", P[i : min(P.size(0), i+num_cpus)])
                sub_batch_f_vals, sub_batch_completed, sub_batch_actual_batch_calls, sub_batch_actual_k = self.bb_k_fail_wrapper.batch_call(P[i : min(P.size(0), i+num_cpus)])
                total_failed += P[i : min(P.size(0), i+num_cpus)].shape[0] - sub_batch_completed.shape[0]

                actual_batch_calls += sub_batch_actual_batch_calls # update total count on this iteration
                actual_k += sub_batch_actual_k # update actual_k (total) as well

                # check if we actually got anything, if not try again with next call
                if sub_batch_f_vals.shape[0] == 0:
                    continue

                # opportunistic - check for improvement
                min_sub_batch_f_val_idx = torch.argmin(sub_batch_f_vals) # note, this is an idx of returned values, not an index of P
                if sub_batch_f_vals[min_sub_batch_f_val_idx] < self.cur_f_val: # found a better value -> update point and tao
                    self.x = P[sub_batch_completed[min_sub_batch_f_val_idx]]
                    self.cur_f_val = sub_batch_f_vals[min_sub_batch_f_val_idx]
                    self.delta = self.delta / self.tao
                    break # exit loop, point found, values updated

            # opportunistic - if this step was reached failed to find improvement
            self.delta = self.delta * self.tao
        

        # update k_fail and number of f evals
        self.prediction_software.add_actual_k(-1)
        self.n_function_calls = self.bb_k_fail_wrapper.p_reuse.get_n_f_evals()
        self.n_failed_function_calls += total_failed
        self.n_batch_calls += actual_batch_calls

        
        # update params
        self.k += 1
        self.log_current()
        
