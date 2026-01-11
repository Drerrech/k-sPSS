import torch
import time

class point_reuse:
        def __init__(self, f):
            self.f = f
            self.f_points = {}
            self.points_raw = []
        
        def evaluate(self, x):
            x_hash = x.numpy().tobytes()

            if x_hash in self.f_points.keys(): # already evaluated at this exact point
                return self.f_points[x_hash]
            else: # must evaluate from scratch
                val = self.f(x)
                self.f_points[x_hash] = val
                self.points_raw.append(x)
                return val # 1 stands for 1 evaluation of the function
        
        def get_n_f_evals(self):
            return len(self.f_points)
        
        def get_evals(self): # returns dict of all evaluations and corresponding values
            points = [[], []]
            
            for x in self.points_raw:
                x_hash = x.numpy().tobytes()
                points[0].append(x)
                points[1].append(self.f_points[x_hash])
            
            return points

class BB_wrapper:
    def __init__(self, f, pattern, time_based=False, random_seed=42): # set random seed to None to disable reproducibility
        self.batch_calls = 0
        self.function_raw_calls = 0 # NOTE: different from point_reuse evals, this counts every time a function is called
        self.function_raw_succesfull_calls = 0

        self.f = f
        self.pattern = pattern # 2xn tensor (n - max number of batch calls if iteration based OR number of time slots if time based)
        self.time_based = time_based
        self.current_pattern_idx = 0

        self.start_time = -1 # time will be set on first batch_call and then used for pattern

        self.p_reuse = point_reuse(f)

        if not random_seed is None:
            torch.manual_seed(seed=random_seed)
    
    def batch_call(self, points, overwrite_k=-1): # optional k argument to overwrite the pattern
        p = points.shape[0] # p - |D| where D' ( D and D is k-sPSS
        
        # check if time should be set
        if self.start_time == -1:
            self.start_time = time.time() # NOTE: system must be able to provide fractions of seconds time for proper use
        
        # get k from the pattern
        if self.current_pattern_idx != self.pattern.shape[0]-1: # check if hasn't reached end of pattern    
            if self.time_based:
                # update idx, skip if over the activation time
                while self.pattern[self.current_pattern_idx + 1, 0] <= time.time - self.start_time:
                    self.current_pattern_idx += 1
            else: # batch_calls based
                self.current_pattern_idx = min(self.batch_call, self.pattern.shape[0]-1)
        
        if overwrite_k == -1:
            k = self.current_pattern_idx[self.current_pattern_idx][1]
        else:
            k = overwrite_k
        
        # get binary tensor for failure
        completed = torch.randperm(torch.arange(0, p-1, p, dtype=torch.int16))[:p-k] # mask of indexes of p-k elements

        # evaluate
        f_vals = torch.zeros(p-k)
        for i, point in enumerate(points[completed]):
            f_vals[i] = self.p_reuse.evaluate(point)
        
        return (f_vals, completed)

