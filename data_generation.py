#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["MASTSIF"] = "/home/karima04/cutest/mastsif/"

import k_fail_GPS
import k_fail_MBTR
import k_fail_GPS_MBTR_hybrid
import BB_wrapper
import k_fail_prediction

import torch


# In[ ]:


alg_names = [
    "GPS-opp-optimistic",
    "GPS-comp-optimistic",
    "GPS-opp-realistic",
    "GPS-comp-realistic",
    "GPS-opp-pessimistic",
    "GPS-comp-pessimistic",

    "MBTR-1-opp-orth-optimistic",
    "MBTR-1-opp-purerand-optimistic",
    "MBTR-1-comp-orth-optimistic",
    "MBTR-1-comp-purerand-optimistic",
    "MBTR-2-opp-orth-optimistic",
    "MBTR-2-opp-purerand-optimistic",
    "MBTR-2-comp-orth-optimistic",
    "MBTR-2-comp-purerand-optimistic",
    "MBTR-1-opp-orth-realistic",
    "MBTR-1-opp-purerand-realistic",
    "MBTR-1-comp-orth-realistic",
    "MBTR-1-comp-purerand-realistic",
    "MBTR-2-opp-orth-realistic",
    "MBTR-2-opp-purerand-realistic",
    "MBTR-2-comp-orth-realistic",
    "MBTR-2-comp-purerand-realistic",
    "MBTR-1-opp-orth-pessimistic",
    "MBTR-1-opp-purerand-pessimistic",
    "MBTR-1-comp-orth-pessimistic",
    "MBTR-1-comp-purerand-pessimistic",
    "MBTR-2-opp-orth-pessimistic",
    "MBTR-2-opp-purerand-pessimistic",
    "MBTR-2-comp-orth-pessimistic",
    "MBTR-2-comp-purerand-pessimistic",

    "Hybrid-1-opp-orth-optimistic",
    "Hybrid-1-opp-purerand-optimistic",
    "Hybrid-1-comp-orth-optimistic",
    "Hybrid-1-comp-purerand-optimistic",
    "Hybrid-2-opp-orth-optimistic",
    "Hybrid-2-opp-purerand-optimistic",
    "Hybrid-2-comp-orth-optimistic",
    "Hybrid-2-comp-purerand-optimistic",
    "Hybrid-1-opp-orth-realistic",
    "Hybrid-1-opp-purerand-realistic",
    "Hybrid-1-comp-orth-realistic",
    "Hybrid-1-comp-purerand-realistic",
    "Hybrid-2-opp-orth-realistic",
    "Hybrid-2-opp-purerand-realistic",
    "Hybrid-2-comp-orth-realistic",
    "Hybrid-2-comp-purerand-realistic",
    "Hybrid-1-opp-orth-pessimistic",
    "Hybrid-1-opp-purerand-pessimistic",
    "Hybrid-1-comp-orth-pessimistic",
    "Hybrid-1-comp-purerand-pessimistic",
    "Hybrid-2-opp-orth-pessimistic",
    "Hybrid-2-opp-purerand-pessimistic",
    "Hybrid-2-comp-orth-pessimistic",
    "Hybrid-2-comp-purerand-pessimistic",
]

NUM_ALGS = len(alg_names)

NUM_PROBLEMS = 32


# # Test Set

# In[3]:


cutest_wrapper = BB_wrapper.BB_cutest_collection(write_to_file="alg_logs_data_run/cutest_problem_selection_hyperparam_tuning.txt", max_dim=100, cap_n_problems=NUM_PROBLEMS,
                problem_names=[
    "SISSER",
    "CHWIRUT2LS", # abnormal
    "DENSCHNB",
    "GBRAINLS",
    "MGH10LS", # abnormal
    "QING",
    "BOX3", # abnormal
    "DENSCHNC",
    # "LANCZOS3LS", # abnormal - even with good params
    # "GAUSS1LS", # abnormal - even with good params
    "MGH09LS",
    # "LANCZOS1LS", # abnormal - even with good params
    # "BA-L1LS", # abnormal - even with good params
    # "GAUSS3LS", # abnormal - even with good params
    "DENSCHNA",
    # "NELSONLS", # abnormal - even with good params
    "LUKSAN21LS",
    "VANDANMSLS",
    "SENSORS",
    "DENSCHND", # abnormal
    # "DANWOODLS", # abnormal - even with good params
    "CUBE",
    "MUONSINELS",
    "DENSCHNE", # abnormal
    "DMN15333LS",
    "DIAMON3DLS",
    "EXPFIT", # abnormal
    "HELIX",
    "LSC2LS",
    "KOWOSB",
    "S308",
    "DANIWOODLS",
    "DENSCHNF",
    "LRIJCNN1",
    "YFITU",
    # "GAUSS2LS", # abnormal - even with good params
    "CHWIRUT1LS", # abnormal
    "COOLHANSLS",
    "HAHN1LS",
    "HIMMELBB", # abnormal
                ])

# 8 problems that break even with good params
# problem |         max norm |                 f(x) | point
# ----------------------------------------------------------------------------------------------------
#          BA-L1LS |         990.5243 |         9.219979e+21 | [-0.7446797668400358, -0.5794640963847512, -3.500486499158188, -1.323064012068379, -0.8821493419036788, -1.153266032451207, -1.037392399990249, -1.7370471505998548, -0.17264920003296513, -0.4199224842103658, 0.554345516022883, 400.32507031390696, 0.5155897025421708, 0.3353901881454257, -1.0909267718088276, 0.6574185667019381, 0.004708728385872163, 1.559525356418754, -0.25605862627329956, 0.6835794392982492, 403.2747372135689, -1.2256417192317774, 1.3594889258775074, -1.5979727812800866, -1.0860927258990598, 1.5105830155390567, -0.29574298935971965, 0.18619405401275946, -1.5642689391963902, 402.0687289118412, -0.2616010582518473, 0.06891134488748785, -0.9446102477864426, -1.0731416075359488, -1.3379491284948843, -0.99565881271732, -1.1728883749045516, 0.5812987787646552, 408.29874934964937, -0.11559198901642201, 0.13525565863270428, -1.3489024022805156, -2.3137113543778423, 0.49262514747475517, -0.785366139956941, -0.506207567601328, -1.9268780779095296, 406.02145117197347, -0.5096040443748667, -0.6721877138836985, 0.47587316574278804, -0.17102787064680636, -1.8429799248670111, 0.07077613383914438, 0.8619971105045817, 1.6253086384384314, 406.155553922073]
#        DANWOODLS |           9.3062 |                  nan | [-2.834344446659088, 8.864126235246658]
#         GAUSS1LS |         247.8049 |         2.742551e+56 | [97.36571794748306, -0.23966244196891784, 100.08577281236649, 66.2136663198471, 19.28817927837372, 69.28813630342484, 179.0735146999359, 16.897746235132217]
#         GAUSS2LS |         245.4660 |         2.686505e+56 | [96.36571794748306, -0.23966244196891784, 103.08577281236649, 107.2136663198471, 17.28817927837372, 71.28813630342484, 152.0735146999359, 18.397746235132217]
#         GAUSS3LS |         237.7268 |        2.236339e+114 | [95.27756401300431, -0.5074717435836792, 90.63045934438705, 113.69683742523193, 20.402080088853836, 74.77810739278793, 140.9900802373886, 20.220720440149307]
#       LANCZOS1LS |          67.8451 |         1.918248e+31 | [19.08627117872238, -28.728440307825803, 33.668587347865106, 33.67619502544403, 19.488707780838013, 27.674590772390367]
#       LANCZOS3LS |          67.8451 |         1.918248e+31 | [19.08627117872238, -28.728440307825803, 33.668587347865106, 33.67619502544403, 19.488707780838013, 27.674590772390367]
#         NELSONLS |           6.3727 |        2.832579e+284 | [4.01845096051693, 4.806541366672516, -1.1663485860824585]


# In[4]:


# evaluating n problem functions
for i in range(NUM_PROBLEMS):
    p = cutest_wrapper.problems[i]
    f = cutest_wrapper.problem_functions[i]
    print(f"idx: {i} | {p.name} | n: {p.n}: {f(torch.from_numpy(p.x0))}")



# # Algorithm Wrappers

# In[6]:


import time
import os

TIME_LIMIT_S = 600.0

def read_last_fx(filename):
    last_fx = None
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # Skip header and empty lines
            if not line or line.startswith("k"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    last_fx = float(parts[2])
                except ValueError:
                    pass
    return last_fx


# In[7]:


def gps_path_batch_constrained(problem_idx, cluster_size, k, k_assumed, delta, tao, n_batch_calls_limit, opportunistic=True):
    log_file_path = f"alg_logs_data_run/GPS/data_generation_{cutest_wrapper.problems[problem_idx].name}_cluster{cluster_size}_k{k}_kassumed{k_assumed}_delta{delta:.2f}_tao{tao:.2f}_batchlim{n_batch_calls_limit}_{'opp' if opportunistic else 'com'}_gps.txt"
    if os.path.exists(log_file_path):
        return log_file_path

    GPS_expl_k_fail_wrapper = BB_wrapper.BB_k_fail_wrapper(cutest_wrapper.problem_functions[problem_idx], k_assumed*torch.ones((4096, 2), dtype=torch.int16), cluster_size)
    alg = k_fail_GPS.GPS_k_fail(torch.from_numpy(cutest_wrapper.problems[problem_idx].x0), GPS_expl_k_fail_wrapper, delta, tao, prediction_software=k_fail_prediction.constant_prediction_software(k), log_file_path=log_file_path, use_opportunistic_cpu_exploitation=opportunistic) # are true by default

    alg.log_current(message=f"k={k}")

    start_time = time.time()

    while alg.n_batch_calls < n_batch_calls_limit:
        if time.time() - start_time >= TIME_LIMIT_S:
            with open("alg_logs_data_run/GPS/timeout_list.txt", "a") as f:
                f.write(f"TIME LIMIT REACHED ON: {cutest_wrapper.problems[problem_idx].name} | {cutest_wrapper.problems[problem_idx]} | x_k = {alg.x} | f(x) = {alg.cur_f_val}\n")
            break
        alg.step_default(random_rotate=True)

    return log_file_path


# In[8]:


def mbtr_path_batch_constrained(problem_idx, cluster_size, k, k_assumed, delta, mu, eta, gamma, eps_stop, preferred_model_order, n_batch_calls_limit, opportunistic_cpu_exploitation_manual_point_limit, opportunistic=True, orthogonal=True):
    log_file_path = f"alg_logs_data_run/MBTR/data_generation_{cutest_wrapper.problems[problem_idx].name}_cluster{cluster_size}_k{k}_kassumed{k_assumed}_delta{delta:.2f}_mu{mu:.2f}_eta{eta:.2f}_gamma{gamma:.2f}_epsstop{eps_stop:.2f}_preford{preferred_model_order}_batchlim{n_batch_calls_limit}_oppcpulim{opportunistic_cpu_exploitation_manual_point_limit}_{'opp' if opportunistic else 'com'}_{'orth' if orthogonal else 'rand'}_mbtr.txt"
    if os.path.exists(log_file_path):
        return log_file_path

    MBTR_expl_k_fail_wrapper = BB_wrapper.BB_k_fail_wrapper(cutest_wrapper.problem_functions[problem_idx], k_assumed*torch.ones((1024, 2), dtype=torch.int16), cluster_size)

    alg = k_fail_MBTR.MBTR_k_fail(torch.from_numpy(cutest_wrapper.problems[problem_idx].x0), MBTR_expl_k_fail_wrapper, delta, mu, eta, gamma, eps_stop, prediction_software=k_fail_prediction.constant_prediction_software(k), log_file_path=log_file_path,
                                  preferred_model_order=preferred_model_order,
                                  opportunistic_cpu_exploitation_manual_point_limit=opportunistic_cpu_exploitation_manual_point_limit,
                                  use_opportunistic_cpu_exploitation=opportunistic,
                                  use_orthogonal=orthogonal)

    alg.log_current()

    start_time = time.time()

    while alg.n_batch_calls < n_batch_calls_limit:
        if time.time() - start_time >= TIME_LIMIT_S:
            with open("alg_logs_data_run/MBTR/timeout_list.txt", "a") as f:
                f.write(f"TIME LIMIT REACHED ON: {cutest_wrapper.problems[problem_idx].name} | {cutest_wrapper.problems[problem_idx]} | x_k = {alg.x} | f(x) = {alg.cur_f_val}\n")
            break
        alg.step_default()

    return log_file_path


# In[ ]:


def mbtr_gps_hybrid_path_batch_constrained(problem_idx, cluster_size, k, k_assumed, delta, mu, eta, gamma, eps_stop, preferred_model_order, n_batch_calls_limit, opportunistic_cpu_exploitation_manual_point_limit, opportunistic=True, orthogonal=True):
    log_file_path = f"alg_logs_data_run/MBTR_GPS_hybrid/data_generation_{cutest_wrapper.problems[problem_idx].name}_cluster{cluster_size}_k{k}_kassumed{k_assumed}_delta{delta:.2f}_mu{mu:.2f}_eta{eta:.2f}_gamma{gamma:.2f}_epsstop{eps_stop:.2f}_preford{preferred_model_order}_batchlim{n_batch_calls_limit}_oppcpulim{opportunistic_cpu_exploitation_manual_point_limit}_{'opp' if opportunistic else 'com'}_{'orth' if orthogonal else 'rand'}_hybrid.txt"
    if os.path.exists(log_file_path):
        return log_file_path

    MBTR_expl_k_fail_wrapper = BB_wrapper.BB_k_fail_wrapper(cutest_wrapper.problem_functions[problem_idx], k_assumed*torch.ones((1024, 2), dtype=torch.int16), cluster_size)

    alg = k_fail_GPS_MBTR_hybrid.MBTR_GPS_hybrid_k_fail(torch.from_numpy(cutest_wrapper.problems[problem_idx].x0), MBTR_expl_k_fail_wrapper, delta, mu, eta, gamma, eps_stop, prediction_software=k_fail_prediction.constant_prediction_software(k), log_file_path=log_file_path,
                                  preferred_model_order=preferred_model_order,
                                  opportunistic_cpu_exploitation_manual_point_limit=opportunistic_cpu_exploitation_manual_point_limit,
                                  use_opportunistic_cpu_exploitation=opportunistic,
                                  use_orthogonal=orthogonal)

    alg.log_current()

    start_time = time.time()

    while alg.n_batch_calls < n_batch_calls_limit:
        if time.time() - start_time >= TIME_LIMIT_S:
            with open("alg_logs_data_run/MBTR_GPS_hybrid/timeout_list.txt", "a") as f:
                f.write(f"TIME LIMIT REACHED ON: {cutest_wrapper.problems[problem_idx].name} | {cutest_wrapper.problems[problem_idx]} | x_k = {alg.x} | f(x) = {alg.cur_f_val}\n")
            break
        alg.step_default()

    return log_file_path


# # GPS and MBTR hyperparameters and cluster settings

# In[ ]:


dim_to_cluster_sizes = { # upper inclusive : [possible ranges]
    4: [4],
    16: [4, 16],
    100: [16, 64]
}

cluster_size_to_k_vals = { # cluster size: possible k values
    4: [0, 1],
    16: [0, 1, 2, 4],
    64: [0, 4, 8, 16]
}

k_assumed_functions = {
    "optimistic": lambda _cluster_size, _k_actual: 0,
    "realistic": lambda _cluster_size, _k_actual: _k_actual,
    "pessimistic": lambda _cluster_size, _k_actual: cluster_size_to_k_vals[_cluster_size][-1],
}


# GPS
gps_delta = 1.0
gps_tau = 0.3

# MBTR ORDER 1
mbtr_1_delta = 1.0
mbtr_1_mu = 2.0
mbtr_1_eta = 0.1
mbtr_1_gamma = 0.5
mbtr_1_eps_stop = 0.0

# MBTR ORDER 2
mbtr_2_delta = 1.0
mbtr_2_mu = 2.0
mbtr_2_eta = 0.1
mbtr_2_gamma = 0.5
mbtr_2_eps_stop = 0.0

# batch limit depended on dim
batch_limit_func = lambda n: 10 * n
# MBTR function limit
mbtr_model_function_lim = lambda n: 6*n


# # Run the algorithms

# In[11]:


def get_cluster_sizes(n):
    for k, v in dim_to_cluster_sizes.items():
        if n <= k:
            return v

    raise Exception("did not find cluster size range for supplied dim")


# In[12]:


from tqdm import tqdm
import math

log_paths = [[] for _ in range(NUM_ALGS)]


# In[13]:


current_path_idx = 0


# GPS

# In[14]:


gps_combination_count = 0 
for problem_idx in range(NUM_PROBLEMS):
    problem_n = cutest_wrapper.problems[problem_idx].n
    cluster_sizes = get_cluster_sizes(problem_n)
    for cluster_size in cluster_sizes:
        ks = cluster_size_to_k_vals[cluster_size]
        for k in ks:
            gps_combination_count += 1

def gps_all_combinations(path_idx, k_assumed_f, opportunistic):
    pbar = tqdm(total=gps_combination_count)

    for problem_idx in range(NUM_PROBLEMS):
        problem_n = cutest_wrapper.problems[problem_idx].n

        cluster_sizes = get_cluster_sizes(problem_n)

        for cluster_size in cluster_sizes:
            ks = cluster_size_to_k_vals[cluster_size]

            for k in ks:
                log_paths[path_idx].append(gps_path_batch_constrained(problem_idx, cluster_size, k, k_assumed_f(cluster_size, k), gps_delta, gps_tau, batch_limit_func(problem_n), opportunistic))

                
                pbar.update(1)

    pbar.close()


# In[15]:


for k_assumed_f in k_assumed_functions.values():
    # GPS - opportunistic
    gps_all_combinations(current_path_idx, k_assumed_f, True)
    # GPS - complete
    gps_all_combinations(current_path_idx+1, k_assumed_f, False)

    current_path_idx += 2


# MBTR

# In[ ]:


mbtr_combination_count = 0
for problem_idx in range(NUM_PROBLEMS):
    problem_n = cutest_wrapper.problems[problem_idx].n
    cluster_sizes = get_cluster_sizes(problem_n)
    for cluster_size in cluster_sizes:
        ks = cluster_size_to_k_vals[cluster_size]
        for k in ks:
            mbtr_combination_count += 1

def mbtr_all_combinations(path_idx, k_assumed_f, order, opportunistic, orthogonal):
    pbar = tqdm(total=mbtr_combination_count)

    for problem_idx in range(NUM_PROBLEMS):
        problem_n = cutest_wrapper.problems[problem_idx].n

        cluster_sizes = get_cluster_sizes(problem_n)

        for cluster_size in cluster_sizes:
            ks = cluster_size_to_k_vals[cluster_size]

            for k in ks:
                if order == 1:
                    log_paths[path_idx].append(mbtr_path_batch_constrained(problem_idx, cluster_size, k, k_assumed_f(cluster_size, k), mbtr_1_delta, mbtr_1_mu, mbtr_1_eta, mbtr_1_gamma, mbtr_1_eps_stop, 1, batch_limit_func(problem_n), mbtr_model_function_lim(problem_n), opportunistic, orthogonal))
                else:
                    log_paths[path_idx].append(mbtr_path_batch_constrained(problem_idx, cluster_size, k, k_assumed_f(cluster_size, k), mbtr_2_delta, mbtr_2_mu, mbtr_2_eta, mbtr_2_gamma, mbtr_2_eps_stop, 2, batch_limit_func(problem_n), mbtr_model_function_lim(problem_n), opportunistic, orthogonal))

                
                pbar.update(1)

    pbar.close()


# In[17]:


for k_assumed_f in k_assumed_functions.values():
    # MBTR - order 1 opportunistic orthogonal
    mbtr_all_combinations(current_path_idx, k_assumed_f, 1, True, True)
    # MBTR - order 1 opportunistic pure random
    mbtr_all_combinations(current_path_idx+1, k_assumed_f, 1, True, False)
    # MBTR - order 1 complete orthogonal
    mbtr_all_combinations(current_path_idx+2, k_assumed_f, 1, False, True)
    # MBTR - order 1 complete pure random
    mbtr_all_combinations(current_path_idx+3, k_assumed_f, 1, False, False)
    # MBTR - order 2 opportunistic orthogonal
    mbtr_all_combinations(current_path_idx+4, k_assumed_f, 2, True, True)
    # MBTR - order 2 opportunistic pure random
    mbtr_all_combinations(current_path_idx+5, k_assumed_f, 2, True, False)
    # MBTR - order 2 complete orthogonal
    mbtr_all_combinations(current_path_idx+6, k_assumed_f, 2, False, True)
    # MBTR - order 2 complete pure random
    mbtr_all_combinations(current_path_idx+7, k_assumed_f, 2, False, False)

    current_path_idx += 8


# Hybrid

# In[ ]:


hybrid_combination_count = 0
for problem_idx in range(NUM_PROBLEMS):
    problem_n = cutest_wrapper.problems[problem_idx].n
    cluster_sizes = get_cluster_sizes(problem_n)
    for cluster_size in cluster_sizes:
        ks = cluster_size_to_k_vals[cluster_size]
        for k in ks:
            hybrid_combination_count += 1

def hybrid_all_combinations(path_idx, k_assumed_f, order, opportunistic, orthogonal):
    pbar = tqdm(total=hybrid_combination_count)

    for problem_idx in range(NUM_PROBLEMS):
        problem_n = cutest_wrapper.problems[problem_idx].n

        cluster_sizes = get_cluster_sizes(problem_n)

        for cluster_size in cluster_sizes:
            ks = cluster_size_to_k_vals[cluster_size]

            for k in ks:
                if order == 1:
                    log_paths[path_idx].append(mbtr_gps_hybrid_path_batch_constrained(problem_idx, cluster_size, k, k_assumed_f(cluster_size, k), mbtr_1_delta, mbtr_1_mu, mbtr_1_eta, mbtr_1_gamma, mbtr_1_eps_stop, 1, batch_limit_func(problem_n), mbtr_model_function_lim(problem_n), opportunistic, orthogonal))
                else:
                    log_paths[path_idx].append(mbtr_gps_hybrid_path_batch_constrained(problem_idx, cluster_size, k, k_assumed_f(cluster_size, k), mbtr_2_delta, mbtr_2_mu, mbtr_2_eta, mbtr_2_gamma, mbtr_2_eps_stop, 2, batch_limit_func(problem_n), mbtr_model_function_lim(problem_n), opportunistic, orthogonal))

                
                pbar.update(1)

    pbar.close()


# In[19]:


for k_assumed_f in k_assumed_functions.values():
    # Hybrid - order 1 opportunistic orthogonal
    hybrid_all_combinations(current_path_idx, k_assumed_f, 1, True, True)
    # Hybrid - order 1 opportunistic pure random
    hybrid_all_combinations(current_path_idx+1, k_assumed_f, 1, True, False)
    # Hybrid - order 1 complete orthogonal
    hybrid_all_combinations(current_path_idx+2, k_assumed_f, 1, False, True)
    # Hybrid - order 1 complete pure random
    hybrid_all_combinations(current_path_idx+3, k_assumed_f, 1, False, False)
    # Hybrid - order 2 opportunistic orthogonal
    hybrid_all_combinations(current_path_idx+4, k_assumed_f, 2, True, True)
    # Hybrid - order 2 opportunistic pure random
    hybrid_all_combinations(current_path_idx+5, k_assumed_f, 2, True, False)
    # Hybrid - order 2 complete orthogonal
    hybrid_all_combinations(current_path_idx+6, k_assumed_f, 2, False, True)
    # Hybrid - order 2 complete pure random
    hybrid_all_combinations(current_path_idx+7, k_assumed_f, 2, False, False)

    current_path_idx += 8
