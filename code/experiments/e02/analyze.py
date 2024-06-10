import torch
seed = 0
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

import os
from math import ceil


def analyze(params, get_file_name_e01, get_file_name_e02, job_id):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Loading/computing statistics...")

    input_file = get_file_name_e01(L, n, Cw, Nw, seed, name="activations_aggregate")
    if not os.path.isfile(input_file):
        exit(1)
    else:
        activations_aggregate = torch.load(input_file)

    # Note: tpc = two point correlator
    if not (os.path.isfile(get_file_name_e02(L, n, Cw, Nw, seed, name="fpc_factor3")) and \
        os.path.isfile(get_file_name_e02(L, n, Cw, Nw, seed, name="fpc_factor1"))):
        fpc_factor1_aggregate = torch.zeros((Nw, L))
        fpc_factor3_aggregate = torch.zeros((Nw, L))
        reporting_interval = ceil(Nw / 20)
        for i in range(Nw):
            if i % reporting_interval == 0 or i == Nw-1:
                print(f"[job {job_id}] (analyze) Progress: {i+1}/{Nw}")
            for j in range(L):
                layer = activations_aggregate[i, j, :] ** 2
                repeated = layer.repeat(n, 1)
                nonzero_fpc_elements_matrix = repeated * repeated.T
                fpc_factor3_aggregate[i, j] = torch.diag(nonzero_fpc_elements_matrix).mean()
                nonzero_fpc_elements_matrix.fill_diagonal_(0)
                fpc_factor1_aggregate[i, j] = nonzero_fpc_elements_matrix.sum() / (n * n - n)

                del layer, repeated, nonzero_fpc_elements_matrix
    
        fpc_factor3 = torch.mean(fpc_factor3_aggregate, dim=(0))
        fpc_factor1 = torch.mean(fpc_factor1_aggregate, dim=(0))
        torch.save(fpc_factor3, get_file_name_e02(L, n, Cw, Nw, seed, name="fpc_factor3"))
        torch.save(fpc_factor1, get_file_name_e02(L, n, Cw, Nw, seed, name="fpc_factor1"))
