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
    
    D = torch.from_numpy(Z).float()

    # due to numerical instabilit will some times produce shape [X, >n]
    # not sure about proper solution yet, will just crop it
    D = D[:, :n]

    return D

def normalised_k_sPSS(n, k):
    p = 2*k+n+1
    d = p-n

    V = np.zeros((d,p))

    T = np.linspace(1,p,num=p)

    for i in range(p):
        V[:,i] = [T[i]**j for j in range(1,d+1)]

    Z = null_space(V)
    
    D = torch.from_numpy(Z).float()

    # due to numerical instabilit will some times produce shape [X, >n]
    # not sure about proper solution yet, will just crop it
    D = D[:, :n]

    # normalise each vector
    for i in range(D.shape[0]):
        D[i] /= max(torch.linalg.vector_norm(D[i]), 1e-12)
    
    return D
