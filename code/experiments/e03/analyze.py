import torch
seed = 0
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

import os
from math import ceil
from .utils import flatten_without_diagonal

def analyze(params, get_file_name_e01, get_file_name_e03, job_id):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Loading/computing statistics...")

    def is_cached_e01(file):
        return os.path.isfile(get_file_name_e01(L, n, Cw, Nw, seed, name=file))
    
    def is_cached_e03(file):
        return os.path.isfile(get_file_name_e03(L, n, Cw, Nw, seed, name=file))

    if not is_cached_e01("activations_aggregate"):
        exit(1)
    else:
        activations_aggregate = torch.load(get_file_name_e01(L, n, Cw, Nw, seed, name="activations_aggregate"))

    # Note: tpc = two point correlator, fpc = four point correlator
    if not (
        is_cached_e03("tpc_zeros_e03") and \
        is_cached_e03("tpc_diagonal_e03") and \
        is_cached_e03("fpc_factor3_e03") and \
        is_cached_e03("fpc_factor1_e03")):

        # tpc
        tpc_diagonal_aggregate = torch.zeros((Nw, 2, n))
        tpc_zeros_sum = torch.zeros((2, n, n))
        reporting_interval = ceil(Nw / 20)
        for i in range(Nw):
            if i % reporting_interval == 0 or i == Nw-1:
                print(f"[job {job_id}] (analyze - tpc) Progress: {i+1}/{Nw}")
            for j in [0, 1]:
                layer_index = 0 if j == 0 else L-1
                layer = activations_aggregate[i, layer_index, :]
                repeated = layer.repeat(n, 1)
                covariance_matrix = repeated * repeated.T
                tpc_diagonal_aggregate[i, j, :] = torch.diag(covariance_matrix)
                covariance_matrix.fill_diagonal_(0)
                tpc_zeros_sum[j, :, :] += covariance_matrix

                del layer, repeated, covariance_matrix

        tpc_zeros = torch.empty((2, n * n - n))
        tpc_zeros[0, :] = flatten_without_diagonal(tpc_zeros_sum[0, :, :] / Nw)
        tpc_zeros[1, :] = flatten_without_diagonal(tpc_zeros_sum[1, :, :] / Nw)
        
        tpc_diagonal = torch.mean(tpc_diagonal_aggregate, dim=(0))
        torch.save(tpc_zeros, get_file_name_e03(L, n, Cw, Nw, seed, name="tpc_zeros_e03"))
        torch.save(tpc_diagonal, get_file_name_e03(L, n, Cw, Nw, seed, name="tpc_diagonal_e03"))

        # fpc
        fpc_factor1_sum = torch.zeros((2, n, n))
        fpc_factor3_aggregate = torch.zeros((Nw, 2, n))
        reporting_interval = ceil(Nw / 20)
        for i in range(Nw):
            if i % reporting_interval == 0 or i == Nw-1:
                print(f"[job {job_id}] (analyze - fpc) Progress: {i+1}/{Nw}")
            for j in [0, 1]:
                layer_index = 0 if j == 0 else L-1
                layer = activations_aggregate[i, layer_index, :] ** 2
                repeated = layer.repeat(n, 1)
                nonzero_fpc_elements_matrix = repeated * repeated.T
                fpc_factor3_aggregate[i, j, :] = torch.diag(nonzero_fpc_elements_matrix)
                nonzero_fpc_elements_matrix.fill_diagonal_(0)
                fpc_factor1_sum[j, :, :] += nonzero_fpc_elements_matrix

                del layer, repeated, nonzero_fpc_elements_matrix
    
        fpc_factor1 = torch.empty((2, n * n - n))
        fpc_factor1[0, :] = flatten_without_diagonal(fpc_factor1_sum[0, :, :] / Nw)
        fpc_factor1[1, :] = flatten_without_diagonal(fpc_factor1_sum[1, :, :] / Nw)

        fpc_factor3 = torch.mean(fpc_factor3_aggregate, dim=(0))
        torch.save(fpc_factor3, get_file_name_e03(L, n, Cw, Nw, seed, name="fpc_factor3_e03"))
        torch.save(fpc_factor1, get_file_name_e03(L, n, Cw, Nw, seed, name="fpc_factor1_e03"))

