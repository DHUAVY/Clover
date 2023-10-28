import torch
import numpy as np
import math

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def cos_norm(a, eps=1e-8):
    if a is None:
        return a
    a_n = a.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

# CKA
def linear_kernel(X, Y):
    return np.matmul(X, Y.T)

def rbf(X, Y, sigma=None):
    """
    Radial-Basis Function kernel for X and Y with bandwith chosen
    from median if not specified.
    """
    GX = np.dot(X, Y.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX

def HSIC(K, L):
    """
    Calculate Hilbert-Schmidt Independence Criterion on K and L.
    """
    n = K.shape[0]
    H = np.identity(n) - (1./n) * np.ones((n, n))

    KH = np.matmul(K, H)
    LH = np.matmul(L, H)
    return 1./((n-1)**2) * np.trace(np.matmul(KH, LH))

def CKA(X, Y, kernel=None):
    """
    Calculate Centered Kernel Alingment for X and Y. If no kernel
    is specified, the linear kernel will be used.
    """
    kernel = linear_kernel if kernel is None else kernel
    X = X.cuda().data.cpu().numpy()
    Y = Y.cuda().data.cpu().numpy()
    K = kernel(X, X)
    L = kernel(Y, Y)
        
    hsic = HSIC(K, L)
    varK = np.sqrt(HSIC(K, K))
    varL = np.sqrt(HSIC(L, L))
    return hsic / (varK * varL)