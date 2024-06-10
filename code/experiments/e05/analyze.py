from utils import free_memory
import torch
# torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)
from logger import logV2 as log, error

import os


def analyze(params, get_cache_file_name_generate, get_cache_file_name_analyze, job_id):
    L, n, Cw, Cb, Nw, seed, activation, input_norm = params
    torch.manual_seed(seed)

    input_file = get_cache_file_name_generate(params, name="z_aggr")
    output_file_norms_mean = get_cache_file_name_analyze(params, name="z_norms_mean")
    output_file_norms_quantiles = get_cache_file_name_analyze(params, name="z_norms_quantiles")
    
    if os.path.isfile(output_file_norms_mean) and os.path.isfile(output_file_norms_quantiles):
        log(job_id, "Analysis results found in cache, continuing to next step...")
        return

    if not os.path.isfile(input_file):
        error(job_id, "Required input not found!")
    
    log(job_id, "(analyze) Computing preactivation norms...")
    z_aggr: torch.Tensor = torch.load(input_file) # dims: (Nw, L, n)
    quantiles = torch.tensor([0.025, 0.125, 0.25, 0.75, 0.875, 0.975])
    if torch.cuda.is_available():
        z_aggr = z_aggr.to("cuda")
        quantiles = quantiles.to("cuda")

    # Note: in-place calculations necessary to stay within GPU memory limits:
    z_norms = torch.sqrt(torch.sum(z_aggr.pow_(2), dim=2)) # dims: (Nw, L)
    z_norms_means = torch.mean(z_norms, dim=0) # dims: (L,)
    
    z_norms_quantiles = torch.quantile(z_norms, quantiles, dim=0) # (len(quantiles), L,)

    torch.save(z_norms_means.cpu(), output_file_norms_mean)
    torch.save(z_norms_quantiles.cpu(), output_file_norms_quantiles)

    to_delete = [z_aggr, quantiles, z_norms, z_norms_means, z_norms_quantiles]
    free_memory(to_delete)
    
