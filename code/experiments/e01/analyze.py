import torch
seed = 0
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

import os
from math import ceil

def analyze(params, get_file_name, job_id):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Loading/computing statistics...")

    input_file = get_file_name(L, n, Cw, Nw, seed, name="activations_aggregate")
    if not os.path.isfile(input_file):
        exit(1)
    else:
        activations_aggregate = torch.load(input_file)

    # Note: tpc = two point correlator
    if not (os.path.isfile(get_file_name(L, n, Cw, Nw, seed, name="tpc_zeros")) and \
        os.path.isfile(get_file_name(L, n, Cw, Nw, seed, name="tpc_diagonal"))):
        tpc_zeros_aggregate = torch.zeros((Nw, L))
        tpc_diagonal_aggregate = torch.zeros((Nw, L))
        reporting_interval = ceil(Nw / 4)
        for i in range(Nw):
            if i % reporting_interval == 0 or i == Nw-1:
                print(f"[job {job_id}] (analyze) Progress: {i+1}/{Nw}")
            for j in range(L):
                layer = activations_aggregate[i, j, :]
                repeated = layer.repeat(n, 1)
                covariance_matrix = repeated * repeated.T
                tpc_diagonal_aggregate[i, j] = torch.diag(covariance_matrix).mean()
                covariance_matrix.fill_diagonal_(0)
                tpc_zeros_aggregate[i, j] = covariance_matrix.sum() / (n * n - n)

                del layer, repeated, covariance_matrix

        tpc_zeros = torch.mean(tpc_zeros_aggregate, dim=(0))
        tpc_diagonal = torch.mean(tpc_diagonal_aggregate, dim=(0))
        torch.save(tpc_zeros, get_file_name(L, n, Cw, Nw, seed, name="tpc_zeros"))
        torch.save(tpc_diagonal, get_file_name(L, n, Cw, Nw, seed, name="tpc_diagonal"))
