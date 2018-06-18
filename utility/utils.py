__author__ = 'kimangkhun'
import numpy as np
from scipy import linalg

def compute_flows(F, injections, idx_injections):
    verbose = 0
    N = len(F)

    TU = np.zeros((N, N), dtype=int)
    TD = np.zeros((N, N), dtype=int)

    for k in range(N):
        for j in range(N):
            # Up-stream connection of k is Q(k, 1)
            if j != k and (F[k, 0] == F[j, 1]): # same orientation
                TU[k, j] = 1
            elif j != k and (F[k, 0] == F[j, 0]): # opposite orientation
                TU[k, j] = -1

            # Down-stream connection of k is Q(k, 2)
            if j != k and (F[k, 1] == F[j, 0]): # same orientation
                TD[k, j] = 1
            elif j != k and (F[k, 1] == F[j, 1]): # opposite orientation
                TD[k, j] = -1

    for k in range(N):
        if np.count_nonzero(TU[k, :]) == 0:
            TU[k, k] = 1
        if np.count_nonzero(TD[k, :]) == 0:
            TD[k, k] = 1

    if verbose:
        print(TU)
        print(TD)

    # Combined topology matrix
    T = TD - TU

    # Transform the problem into an algebraic equation
    idx_quadripoles = np.setdiff1d(np.array(range(N)), idx_injections)
    Nq = len(idx_quadripoles)

    y = -np.dot(T[:, idx_injections], injections)
    tau = T[:, idx_quadripoles]
    tau_t = np.transpose(tau)
    epsi = 1e-8

    quadripoles = linalg.solve(tau_t.dot(tau) + epsi*np.eye(Nq), tau_t.dot(y))

    # Return result
    f = np.zeros((N, 1))
    f[idx_quadripoles, 0] = quadripoles[:,0]
    f[idx_injections, 0] = injections[:, 0]

    if verbose:
        for k in range(N):
            print("[f%d] S%s, S%s:\t %f\n" % (k, str(F[k, 0]), str(F[k, 1]), f[k, 0]))

    # Check
    converge = False
    t = max(abs(np.dot(T, f)))
    theta = max(abs(f))/1000
    #print("Verification T*f == 0? \n [max(abs(T*f)=%f] < [max(abs(f))/1000=%f]\n"%(t, theta))
    converge = (t < theta)
    return [f,converge]