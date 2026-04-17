import pycutest
import torch
import time
import math

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


class BB_cutest_collection:
    def __init__(self, print_load_status=True, write_to_file="", cap_n_problems=10000, max_dim = 1000, problem_names=None):
        self.problems = []
        self.problem_functions = []
        
        if print_load_status:
            print("loading problems...")
        
        # problem criteria
        raw_problem_selection = pycutest.find_problems(objective="sum of squares other", constraints="unconstrained") if problem_names is None else problem_names # the query for dim contrain simply doesn't work, so implemented manually
        # load problems into self.problems and load the pytorch compativle functions into self.problem_functions
        timings = []
        for i, p_name in enumerate(raw_problem_selection):
            if len(self.problems) == cap_n_problems:
                break
            
            if print_load_status and i%20 == 0:
                print(f"{i}/{len(raw_problem_selection)}")
            
            try:
                time_start = time.time()
                p = pycutest.import_problem(p_name)
                if p.n > max_dim:
                    print(f"skipping {p_name}, dimension too high")
                    continue
                
                self.problems.append(p)
                self.problem_functions.append(lambda x, p=p: p.obj(x.numpy())) # default argument voodoo becuase python will look at the reference of p and not the actual value otherwise

                timings.append(time.time() - time_start)
            except Exception as e:
                if print_load_status:
                    print(f"{p_name} caused an exception while loading: {e}")
                continue

        if print_load_status:
            print(f"loaded {len(self.problems)} problems")
        
        # optional: save the loaded problems to a txt file
        if write_to_file != "":
            with open(write_to_file, "w") as f:
                f.write("id      | name           | load time (s)  | n      | m      | n_fixed| n_free | vartype\n")
                for i, p in enumerate(self.problems):
                    f.write(f"{i:8}|{p.name:16}|{timings[i]:16.4}|{p.n:8}|{p.m:8}|{p.n_fixed:8}|{p.n_free:8}|{' '.join([str(i) for i in p.vartype])}\n")
    


class BB_k_fail_wrapper:
    def __init__(self, f, pattern, num_cpus, time_based=False, random_seed=42): # set random seed to None to disable reproducibility
        self.f = f
        self.pattern = pattern # nx2 tensor (n - max number of batch calls if iteration based OR number of time slots if time based)
        self.num_cpus = num_cpus # used for calculating num_batch_calls on batch_call
        self.time_based = time_based
        self.current_pattern_idx = 0

        self.batch_calls = 0
        self.function_raw_calls = 0 # NOTE: different from point_reuse evals, this counts every time a function is called
        self.function_raw_succesfull_calls = 0

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
                self.current_pattern_idx = min(self.batch_calls, self.pattern.shape[0]-1)
        
        if overwrite_k == -1:
            k = self.pattern[self.current_pattern_idx][1]
        else:
            k = overwrite_k
        
        
        # calculate, actual_batch_calls, how many it would take in a cluster
        actual_batch_calls = math.ceil(p / self.num_cpus)
        
        # completed idxs
        completed = torch.tensor([], dtype=torch.int)
        for b_idx in range (actual_batch_calls - 1):
            completed_batch = b_idx * self.num_cpus + torch.randperm(self.num_cpus, dtype=torch.int)[:-k]
            completed = torch.cat([completed, completed_batch.int()])
        # tail
        _tail_idxs = []
        _sub_batch_completion = torch.randperm(self.num_cpus, dtype=torch.int)[:-k] # completed idx (starting with 0)
        _tail_start_idx = (actual_batch_calls-1) * self.num_cpus
        # print("_tail_start_idx:", _tail_start_idx)
        _n_tail_elems = p - _tail_start_idx
        # print("_n_tail_elems:", _n_tail_elems)
        for sub_batch_idx in _sub_batch_completion:
            # print("sub idx", sub_batch_idx)
            if sub_batch_idx >= _n_tail_elems:
                continue
            # print("appending")
            _tail_idxs.append(_tail_start_idx + sub_batch_idx)
        completed = torch.cat([completed, torch.tensor(_tail_idxs, dtype=torch.int)])

        # completed = torch.randperm(p, dtype=torch.int)[:p-k] # mask of indexes of p-k elements

        # print("completed:", completed)

        # evaluate
        f_vals = torch.zeros(completed.shape[0])
        for i, point in enumerate(points[completed]):
            f_vals[i] = min(1e20, max(-1e20, self.p_reuse.evaluate(point)))

        return (f_vals, completed, actual_batch_calls, k.item())

