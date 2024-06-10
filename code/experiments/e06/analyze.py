from utils import free_memory, print_gpu_memory_info
import torch
# torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)
from logger import logV2 as log, error

import os

def analyze(params, get_cache_file_name_generate, get_cache_file_name_analyze, job_id):
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    torch.manual_seed(seed)

    input_file = get_cache_file_name_generate(params, name="zs_aggr")
    output_file_R_mean = get_cache_file_name_analyze(params, name="R_mean")
    output_file_R_quantiles = get_cache_file_name_analyze(params, name="R_quantiles")
    output_file_zs_mean = get_cache_file_name_analyze(params, name="zs_mean")
    output_file_zs_quantiles = get_cache_file_name_analyze(params, name="zs_quantiles")

    if os.path.isfile(output_file_R_mean) \
        and os.path.isfile(output_file_R_quantiles) \
        and os.path.isfile(output_file_zs_mean) \
        and os.path.isfile(output_file_zs_quantiles):
        log(job_id, "Analysis results found in cache, continuing to next step...")
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

    # Note: in-place calculations necessary to stay within GPU memory limits:
    zs_squared = torch.sum(zs_aggr.pow_(2), dim=3) # dims: (Nw, 2, L)
    zs_norms = zs_squared.sqrt_()
    zs_norms_means = torch.mean(zs_norms, dim=0) # dims: (2, L)
    zs_norms_quantiles = torch.quantile(zs_norms, quantiles, dim=0) # dims: (len(quantiles), 2, L,)
    log(job_id, f"(analyze) The dimensions of `zs_norm_quantiles` are: {zs_norms_quantiles.size()}")
    
    torch.save(zs_norms_means.cpu(), output_file_zs_mean)
    torch.save(zs_norms_quantiles.cpu(), output_file_zs_quantiles)

    zs_squared_averaged_over_neurons = zs_squared / n # dims: (Nw, 2, L)
    zs_squared_averaged = torch.mean(zs_squared_averaged_over_neurons, dim=0) # dims: (2, L)
    R_mean = zs_squared_averaged[1, :] - zs_squared_averaged[0, :] # dims: (L,)  
    zs_difference_of_the_squares = \
        zs_squared_averaged_over_neurons[:, 1, :] \
        - zs_squared_averaged_over_neurons[:, 0, :] # dims: (Nw, L)
    R_quantiles = torch.quantile(zs_difference_of_the_squares, quantiles, dim=0) # dims: (len(quantiles), L)
    log(job_id, f"(analyze) The dimensions of `R_quantiles` are: {R_quantiles.size()}")
    
    torch.save(R_mean.cpu(), output_file_R_mean)
    torch.save(R_quantiles.cpu(), output_file_R_quantiles)

    to_delete = [zs_aggr, quantiles, zs_squared, zs_norms, zs_norms_means,
                 zs_norms_quantiles, zs_squared_averaged_over_neurons,
                 zs_squared_averaged, R_mean, zs_difference_of_the_squares,
                 R_quantiles,]
    free_memory(to_delete)
