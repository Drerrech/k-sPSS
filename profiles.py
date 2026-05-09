import torch
import ast
from matplotlib import pyplot as plt

def parse_log_file(log_path):
    info_dict = {
        "f_x": [],
        "n_function_calls": [],
        "n_failed_function_calls": [],
        "n_1_batch_calls": [],
        "n_batch_calls": [],
    }
    
    with open(log_path, "r") as f:
        skipped_first = False
        for line in f:
            if not skipped_first or not line.strip():
                skipped_first = True
                continue

            elems = line.split("|")

            f_x = ast.literal_eval(elems[2].strip())

            function_call_info_list = ast.literal_eval(elems[4].strip())
            n_function_calls = function_call_info_list[0]
            n_failed_function_calls = function_call_info_list[1]
            n_1_batch_calls = function_call_info_list[2]
            n_batch_calls = function_call_info_list[3]
            
            info_dict["f_x"].append(f_x)
            info_dict["n_function_calls"].append(n_function_calls)
            info_dict["n_failed_function_calls"].append(n_failed_function_calls)
            info_dict["n_1_batch_calls"].append(n_1_batch_calls)
            info_dict["n_batch_calls"].append(n_batch_calls)
    
    # convert to tensor for argmin and so on
    for k, v in info_dict.items():
        info_dict[k] = torch.tensor(v)
    
    return info_dict

def performance_profile(num_algs: int, num_problems: int, alg_log_paths: list[list], tau: float, alpha_vals: torch.tensor=torch.linspace(1, 8, 8), looking_at="n_batch_calls"):
    T_vals = torch.zeros((num_algs, num_problems))
    N_vals = torch.ones((num_algs, num_problems)) * torch.inf
    r_vals = torch.zeros((num_algs, num_problems))
    profile_vals = torch.zeros((num_algs, alpha_vals.shape[0]))

    for problem_idx in range(num_problems):
        # find f* for this problem
        f_star = torch.min(parse_log_file(alg_log_paths[0][problem_idx])["f_x"])
        for alg_idx in range(1, num_algs):
            f_star = min(f_star, torch.min(parse_log_file(alg_log_paths[alg_idx][problem_idx])["f_x"]))
        
        for alg_idx in range(num_algs):
            info_dict = parse_log_file(alg_log_paths[alg_idx][problem_idx])
            
            # fill in N and T
            if info_dict["f_x"][0] - f_star == 0:
                # already optimal at start
                N_vals[alg_idx, problem_idx] = 0
                T_vals[alg_idx, problem_idx] = 1
                continue
            for i in range(info_dict["f_x"].shape[0]-1, -1, -1):
                acc = (info_dict["f_x"][0] - info_dict["f_x"][i]) / (info_dict["f_x"][0] - f_star)
                
                if acc >= 1 - tau:
                    N_vals[alg_idx, problem_idx] = info_dict[looking_at][i]
                    T_vals[alg_idx, problem_idx] = 1
                else:
                    # T is 0 by default
                    break
    
    # fill in r_vals and profile vals
    for alg_idx in range(num_algs):
        for problem_idx in range(num_problems):
            min_N = torch.min(N_vals[T_vals[:, problem_idx] == 1, problem_idx])
            r_vals[alg_idx, problem_idx] = N_vals[alg_idx, problem_idx] / min_N if T_vals[alg_idx, problem_idx] == 1 else torch.inf
        
        for i, alpha_val in enumerate(alpha_vals):
            profile_vals[alg_idx, i] = 1/num_problems * torch.count_nonzero(r_vals[alg_idx] <= alpha_val)
    
    return profile_vals

def data_profile(num_algs: int, num_problems: int, alg_log_paths: list[list], tau: float, problem_dims: torch.tensor, k_vals: torch.tensor=torch.linspace(1, 8, 8), looking_at="n_batch_calls"):
    T_vals = torch.zeros((num_algs, num_problems)) # technically as N_vals is now default as inf we don't need T_vals, as inf <= anything == false
    N_vals = torch.ones((num_algs, num_problems)) * torch.inf
    
    profile_vals = torch.zeros((num_algs, k_vals.shape[0]))

    for problem_idx in range(num_problems):
        # find f* for this problem
        f_star = torch.min(parse_log_file(alg_log_paths[0][problem_idx])["f_x"])
        for alg_idx in range(1, num_algs):
            f_star = min(f_star, torch.min(parse_log_file(alg_log_paths[alg_idx][problem_idx])["f_x"]))
        
        for alg_idx in range(num_algs):
            info_dict = parse_log_file(alg_log_paths[alg_idx][problem_idx])
            
            # fill in N and T
            if info_dict["f_x"][0] - f_star == 0:
                # already optimal at start
                N_vals[alg_idx, problem_idx] = 0
                T_vals[alg_idx, problem_idx] = 1
                continue
            for i in range(info_dict["f_x"].shape[0]-1, -1, -1):
                acc = (info_dict["f_x"][0] - info_dict["f_x"][i]) / (info_dict["f_x"][0] - f_star)
                
                if acc >= 1 - tau:
                    N_vals[alg_idx, problem_idx] = info_dict[looking_at][i]
                    T_vals[alg_idx, problem_idx] = 1
                else:
                    # T is 0 by default
                    break
    
    # fill in profile vals
    for alg_idx in range(num_algs):
        for i, k_val in enumerate(k_vals):
            profile_vals[alg_idx, i] = 1/num_problems * torch.count_nonzero(N_vals[alg_idx] <= k_val * (problem_dims + 1) * T_vals[alg_idx])
    
    return profile_vals

def accuracy_profile(num_algs: int, num_problems: int, alg_log_paths: list[list], d_vals: torch.tensor=torch.linspace(1, 8, 8)):
    f_tot_N_acc = torch.zeros((num_algs, num_problems))
    profile_vals = torch.zeros((num_algs, d_vals.shape[0]))

    for problem_idx in range(num_problems):
        # find f* for this problem
        f_star = torch.min(parse_log_file(alg_log_paths[0][problem_idx])["f_x"])
        for alg_idx in range(1, num_algs):
            f_star = min(f_star, torch.min(parse_log_file(alg_log_paths[alg_idx][problem_idx])["f_x"]))
        
        for alg_idx in range(num_algs):
            info_dict = parse_log_file(alg_log_paths[alg_idx][problem_idx])
            
            # fill in N and T
            if info_dict["f_x"][0] - f_star == 0:
                # already optimal at start
                f_tot_N_acc[alg_idx, problem_idx] = 1
                continue
            
            f_tot_N_acc[alg_idx, problem_idx] = (info_dict["f_x"][0] - info_dict["f_x"][-1]) / (info_dict["f_x"][0] - f_star)
        
        # DEBUG
        print(f"problem {problem_idx:2d} | f_star={float(f_star):.4f} | f_x[0]={float(parse_log_file(alg_log_paths[0][problem_idx])['f_x'][0]):.4f}", end="")
        for alg_idx in range(num_algs):
            info_dict = parse_log_file(alg_log_paths[alg_idx][problem_idx])
            print(f" | alg{alg_idx}: f[-1]={float(info_dict['f_x'][-1]):.4f} acc={float(f_tot_N_acc[alg_idx, problem_idx]):.4f}", end="")
        print()
    
    f_tot_N_acc = torch.clamp(f_tot_N_acc, 0.0, 1.0)
    
    # fill in profile vals
    for alg_idx in range(num_algs):
        for i, d_val in enumerate(d_vals):
            profile_vals[alg_idx, i] = 1/num_problems * torch.count_nonzero(-torch.log10(1 - f_tot_N_acc[alg_idx]) >= d_val)
    
    return profile_vals


MARKERS = [
    "o",   # circle
    "v",   # triangle down
    "^",   # triangle up
    "<",   # triangle left
    ">",   # triangle right
    "1",   # tri down
    "2",   # tri up
    "3",   # tri left
    "4",   # tri right
    "8",   # octagon
    "s",   # square
    "p",   # pentagon
    "*",   # star
    "h",   # hexagon1
    "H",   # hexagon2
    "+",   # plus
    "x",   # x
    "D",   # diamond
    "d",   # thin diamond
    "|",   # vline
    "_",   # hline
    "P",   # plus (filled)
    "X",   # x (filled)
    ".",   # point
    ",",   # pixel
]

def plot_performance_profile(profile_vals, tau, alg_names, alpha_vals: torch.tensor=torch.linspace(1, 8, 8)):
    plt.figure(figsize=(8, 5))

    for alg_idx, alg_name in enumerate(alg_names):
        plt.plot(alpha_vals, profile_vals[alg_idx], label=alg_name, drawstyle="steps-post", linewidth=2.5, marker=MARKERS[alg_idx%len(MARKERS)], markersize=10)

    
    plt.xlabel("α")
    
    plt.ylabel("Portion of τ-solved problems")
    plt.title(f"Performance Profile (τ = {tau})")

    # y is a fraction so clamp to [0, 1]
    plt.ylim(0, 1)
    plt.xlim(alpha_vals[0], alpha_vals[-1])

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_data_profile(profile_vals, tau, alg_names, k_vals: torch.tensor=torch.linspace(1, 8, 8)):
    plt.figure(figsize=(8, 5))

    for alg_idx, alg_name in enumerate(alg_names):
        plt.plot(k_vals, profile_vals[alg_idx], label=alg_name, drawstyle="steps-post", linewidth=2.5, marker=MARKERS[alg_idx%len(MARKERS)], markersize=10)

    
    plt.xlabel("groups of k(n+1) evaluations")
    
    plt.ylabel("Portion of τ-solved problems")
    plt.title(f"Data Profile (τ = {tau})")

    # y is a fraction so clamp to [0, 1]
    plt.ylim(0, 1)
    plt.xlim(k_vals[0], k_vals[-1])

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_accuracy_profile(profile_vals, alg_names, d_vals: torch.tensor=torch.linspace(1, 8, 8)):
    plt.figure(figsize=(8, 5))

    for alg_idx, alg_name in enumerate(alg_names):
        plt.plot(d_vals, profile_vals[alg_idx], label=alg_name, drawstyle="steps-post", linewidth=2.5, marker=MARKERS[alg_idx%len(MARKERS)], markersize=10)

    
    plt.xlabel("Relative accuracy d")
    
    plt.ylabel("Portion of instances solved to rel. acc. d")
    plt.title(f"Accuracy profile")

    # y is a fraction so clamp to [0, 1]
    plt.ylim(0, 1)
    plt.xlim(d_vals[0], d_vals[-1])

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
