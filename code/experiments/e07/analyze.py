from utils import free_memory, print_gpu_memory_info
import torch
# torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)
from logger import logV2 as log, error

import os

def analyze(params, get_cache_file_name_generate, get_cache_file_name_analyze, job_id):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    torch.manual_seed(seed)

    input_file = get_cache_file_name_generate(params, name="zs_aggr")
    output_file_D_mean = get_cache_file_name_analyze(params, name="D_mean")
    output_file_D_quantiles = get_cache_file_name_analyze(params, name="D_quantiles")
    output_file_zs_mean = get_cache_file_name_analyze(params, name="zs_mean")
    output_file_zs_quantiles = get_cache_file_name_analyze(params, name="zs_quantiles")

    if os.path.isfile(output_file_D_mean) \
        and os.path.isfile(output_file_D_quantiles) \
        and os.path.isfile(output_file_zs_mean) \
        and os.path.isfile(output_file_zs_quantiles):
        log(job_id, "Data cached: skipping analysis...")
        return

    if not os.path.isfile(input_file):
        error(job_id, "Required input not found!")
    
    print_gpu_memory_info("before 'analyze")
    log(job_id, "(analyze) Calculating preactivation norms and difference of the norms...")

    zs_aggr: torch.Tensor = torch.load(input_file) # dims: (Nw, 2, L, n)
    quantiles = torch.tensor([0.025, 0.125, 0.25, 0.75, 0.875, 0.975])
    if torch.cuda.is_available():
        zs_aggr = zs_aggr.to("cuda")
        quantiles = quantiles.to("cuda")

    zs_difference: torch.Tensor = zs_aggr[:, 1, :, :] - zs_aggr[:, 0, :, :] # dims: (Nw, L, n)
    # in-place computations to save GPU memory:
    zs_difference_squared = zs_difference.pow_(2) # dims: (Nw, L, n)
    D_not_averaged_over_runs = torch.sum(zs_difference_squared, dim=2) / n # dims: (Nw, L)
    D_mean = torch.mean(D_not_averaged_over_runs, dim=0) # dims: (L,)
    D_quantiles = torch.quantile(D_not_averaged_over_runs, quantiles, dim=0) # dims: (len(quantiles), L)
    log(job_id, f"(analyze) The dimensions of `D_mean` are: {D_mean.size()}")
    log(job_id, f"(analyze) The dimensions of `D_quantiles` are: {D_quantiles.size()}")
    torch.save(D_mean.cpu(), output_file_D_mean)
    torch.save(D_quantiles.cpu(), output_file_D_quantiles)

    # Note! Order is important here. We calculate the square of the preactivations in-place
    # to save GPU memory. Therefore, this step must happen last, because to calculate D
    # we need the square of the difference of the preactivations for the two inputs.
    zs_squared_aggr = zs_aggr.pow_(2) # dims: (Nw, 2, L, n)
    zs_squared = torch.sum(zs_squared_aggr, dim=3) # dims: (Nw, 2, L)
    zs_norms = torch.sqrt(zs_squared) # dims: (Nw, 2, L)
    zs_norms_means = torch.mean(zs_norms, dim=0) # dims: (2, L)
    zs_norms_quantiles = torch.quantile(zs_norms, quantiles, dim=0) # dims: (len(quantiles), 2, L)
    log(job_id, f"(analyze) The dimensions of `zs_norms_means` are: {zs_norms_means.size()}")
    log(job_id, f"(analyze) The dimensions of `zs_norms_quantiles` are: {zs_norms_quantiles.size()}")
    torch.save(zs_norms_means.cpu(), output_file_zs_mean)
    torch.save(zs_norms_quantiles.cpu(), output_file_zs_quantiles)

    to_delete = [zs_aggr, quantiles, zs_difference, zs_difference_squared, D_not_averaged_over_runs,
                 D_mean, D_quantiles, zs_squared_aggr, zs_squared, zs_norms, zs_norms_means, zs_norms_quantiles]
    free_memory(to_delete)
