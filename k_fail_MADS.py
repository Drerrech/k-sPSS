import BB_wrapper

import torch

class constant_prediction_software():
    def __init__(self, k_pred):
        self.k_pred = k_pred # constant value to predict
    
    def predict_k(self):
        return self.k_pred
    
    def add_actual_k(self):
        pass

def select_k_spss(predicted_k_fail, D, delta, delta_mesh):
    pass

class MADS_k_fail:
    def __init__(self, x, bb_k_fail_wrapper, delta, tao, D, select_k_spss, prediction_software, log_file_path): # f_omega will not be used as no problems in the test set have contraints
        self.x = x
        self.bb_k_fail_wrapper = bb_k_fail_wrapper
        self.delta = delta
        self.delta_0 = delta
        self.delta_mesh = min(self.delta, self.delta**2 / self.delta_0)
        self.D = D
        self.tao = tao
        self.select_k_spss = select_k_spss
        self.prediction_software = prediction_software

        self.log_file_path = log_file_path # path to .txt where info should be stored
        open(log_file_path, "w").close() # clear log file
        with open(self.log_file_path, "a") as _f: # add columns
            _f.write("k       | x" + 78*" " + "| f(x)           | delta          | delta_mesh     | n_function_calls | message")
            # k:8, x:80, f(x):16, delta:16, delta_m:16, f_calls:18
        
        self.k = 0
        self.n_function_calls = 0

        self.cur_f_val = self.bb_k_fail_wrapper.p_reuse.evaluate(self.x) # using hashing (or not at step 0) get the current value of f
    
    def log_current(self, message=""): # log information about the current state
        s = f"{self.k:8}|{str([round(i, 16) for i in self.x.tolist()]):80}|{self.cur_f_val:16.4f}|{self.delta:16.8}|{self.delta_mesh:16.8}|{self.n_function_calls:18}| {message}"
        with open(self.log_file_path, "a") as _f:
            _f.write(s + "\n")
    
    def step_default(self): # delta is updated nomally
        # 1 - parameter update
        self.delta_mesh = min(self.delta, self.delta**2 / self.delta_0)

        # 2 - search
        # skip, unless Mr. Hare wants to personally suggest a point t such that f(t) < f(x)

        # 3 - poll
        k_fail_predicted = self.prediction_software.predict_k() # predict k

        P = select_k_spss(k_fail_predicted, self.D, self.delta, self.delta_mesh) # tensor of points to eval
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
        
