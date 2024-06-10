import torch
seed = 0
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

import os
from math import ceil

def analyze(params, get_file_name_e01, get_file_name_e04, job_id):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Loading/computing statistics...")

    def is_cached_e01(file):
        return os.path.isfile(get_file_name_e01(L, n, Cw, Nw, seed, name=file))
    
    def is_cached_e04(file):
        return os.path.isfile(get_file_name_e04(L, n, Cw, Nw, seed, name=file))

    if not is_cached_e01("activations_aggregate"):
        exit(1)
    else:
        activations_aggregate = torch.load(get_file_name_e01(L, n, Cw, Nw, seed, name="activations_aggregate"))

    # Note: cfpc = connected four point correlator
    if is_cached_e04("cfpc_e04"):
        return
    
    reporting_interval = ceil(Nw / 20)
    
    # tpc
    tpc_sum = torch.zeros((L, n, n))
    for i in range(Nw):
        if i % reporting_interval == 0 or i == Nw-1:
            print(f"[job {job_id}] (analyze - tpc) Progress: {i+1}/{Nw}")
        for j in range(L):
            layer = activations_aggregate[i, j, :]
            repeated = layer.repeat(n, 1)
            covariance_matrix = repeated * repeated.T
            tpc_sum[j, :, :] += covariance_matrix

            del layer, repeated, covariance_matrix
    tpc_averaged = tpc_sum / Nw

    # fpc
    fpc_nonzero_elements_sum = torch.zeros((L, n, n))
    for i in range(Nw):
        if i % reporting_interval == 0 or i == Nw-1:
            print(f"[job {job_id}] (analyze - fpc) Progress: {i+1}/{Nw}")
        for j in range(L):
            layer = activations_aggregate[i, j, :] ** 2
            repeated = layer.repeat(n, 1)
            nonzero_fpc_elements_matrix = repeated * repeated.T
            fpc_nonzero_elements_sum[j, :, :] += nonzero_fpc_elements_matrix

            del layer, repeated, nonzero_fpc_elements_matrix

    fpc_nonzero_elements_averaged = fpc_nonzero_elements_sum / Nw

    # cfpc
    reporting_interval = ceil(Nw / 4)
    M_fpc = fpc_nonzero_elements_averaged
    M_wick = torch.zeros((L, n, n))
    for j in range(L):
        if i % reporting_interval == 0 or i == Nw-1:
            print(f"[job {job_id}] (analyze - cfpc) Progress: {i+1}/{Nw}")
        K = tpc_averaged[j, :, :]
        repeated = torch.diag(K).repeat(n, 1)
        alpha = repeated * repeated.T
        beta = K * K
        M_wick[j, :, :] = alpha + 2 * beta
    M_cfpc = M_fpc - M_wick

    # save results
    torch.save(tpc_averaged, get_file_name_e04(L, n, Cw, Nw, seed, name="tpc_e04"))
    torch.save(fpc_nonzero_elements_averaged, get_file_name_e04(L, n, Cw, Nw, seed, name="fpc_e04"))
    torch.save(M_cfpc, get_file_name_e04(L, n, Cw, Nw, seed, name="cfpc_e04"))
