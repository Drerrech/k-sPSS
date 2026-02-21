import BB_wrapper
import polytope_k_sPSS

import torch
import math

class constant_prediction_software():
    def __init__(self, k_pred):
        self.k_pred = k_pred # constant value to predict
    
    def predict_k(self):
        return self.k_pred
    
    def add_actual_k(self, k):
        pass    

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
    def __init__(self, x, bb_k_fail_wrapper, delta, tao, select_k_spss, prediction_software, log_file_path): # f_omega will not be used as no problems in the test set have contraints
        self.x = x
        self.bb_k_fail_wrapper = bb_k_fail_wrapper
        self.delta = delta
        self.tao = tao
        self.select_k_spss = select_k_spss
        self.prediction_software = prediction_software

        self.log_file_path = log_file_path # path to .txt where info should be stored
        open(log_file_path, "w").close() # clear log file
        with open(self.log_file_path, "a") as _f: # add columns
            _f.write("k       | x" + 78*" " + "| f(x)               | delta          | n_function_calls | message\n")
            # k:8, x:80, f(x):16, delta:16, delta_m:16, f_calls:18
        
        self.k = 0
        self.n_function_calls = 0

        self.cur_f_val = self.bb_k_fail_wrapper.p_reuse.evaluate(self.x) # using hashing (or not at step 0) get the current value of f
    
    def log_current(self, message=""): # log information about the current state
        s = f"{self.k:8}|{str([round(i, 16) for i in self.x.tolist()]):80}|{self.cur_f_val:20.16f}|{self.delta:16.8f}|{self.n_function_calls:18}| {message}"
        with open(self.log_file_path, "a") as _f:
            _f.write(s + "\n")
    
    def step_default(self, random_rotate=True): # delta is updated nomally
        # 2 - poll
        k_fail_predicted = self.prediction_software.predict_k() # predict k
        
        P = select_k_spss(k_fail_predicted, self.x.shape[0], self.delta) # tensor of points to eval
        # rotate the matrix (optional)
        if random_rotate:
            P = (get_random_rotation_matrix(self.x.shape[0]) @ P.T).T

        f_vals, completed = self.bb_k_fail_wrapper.batch_call(P)
        actual_k = completed.shape[0]

        # update k_fail and number of f evals
        self.prediction_software.add_actual_k(actual_k)
        self.n_function_calls = self.bb_k_fail_wrapper.p_reuse.get_n_f_evals()
        # given that the points are evaluated in batches, using oppotrunistic or ordered polling will not make any sense, only complete, which does raise some questions...
        # complete polling
        min_f_val_idx = torch.argmin(f_vals) # note, this is an idx of returned values, not an index of P
        if f_vals[min_f_val_idx] < self.cur_f_val: # found a better value -> update point and tao
            self.x = P[completed[min_f_val_idx]]
            self.cur_f_val = f_vals[min_f_val_idx]
            self.delta = self.delta / self.tao
        else: # failed to find a better point
            self.delta = self.delta * self.tao
        
        # update params
        self.k += 1
        
