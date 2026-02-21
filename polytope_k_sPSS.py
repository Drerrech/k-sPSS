import numpy as np
from scipy.linalg import null_space
import torch

def default_k_sPSS(n, k):
    p = 2*k+n+1
    d = p-n

    V = np.zeros((d,p))

    T = np.linspace(1,p,num=p)

    for i in range(p):
        V[:,i] = [T[i]**j for j in range(1,d+1)]

    Z = null_space(V)
    
    return torch.from_numpy(Z).float()
