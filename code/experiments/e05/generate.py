import torch
# torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)
import math
from logger import logV2 as log

from .model import MLPs_at_initialization_batched
import os
from math import ceil
import seaborn as sns
sns.set_theme()


def generate(params, get_cache_file_name, job_id, batch_size_on_GPU = None):
    L, n, Cw, Cb, Nw, seed, activation, input_norm = params
    torch.manual_seed(seed)
    
    if os.path.isfile(get_cache_file_name(params, name="z_aggr")):
        log(job_id, "Data cached: skipping generation...")
        return

    assert Nw % batch_size_on_GPU == 0, "Check GPU batch size (must divide Nw)"

    log(job_id, "Generating data...")
    magnitude_of_single_element = math.sqrt((input_norm ** 2) / n)
    log(job_id, f"Generating data with input_norm={input_norm}"
            + ", which corresponds to the size S of a single element being:"
            + f" S={magnitude_of_single_element}")
    x = torch.full((n,), magnitude_of_single_element)
    x = torch.unsqueeze(x, 0)
    
    preactivations_aggregate = torch.zeros((Nw, L, n))
    gpu_accelerated_experiment = MLPs_at_initialization_batched(
        num_layers = L,
        layer_width = n,
        Cw = Cw,
        Cb = Cb,
        activation = activation,
        inputs = x,
        num_initializations_per_batch = batch_size_on_GPU,
    )
    reporting_interval = ceil(Nw // batch_size_on_GPU // 10)
    for i in range(Nw // batch_size_on_GPU):
        if i % reporting_interval == 0 or i == Nw-1:
            log(job_id, f"(generate) Progress: batch {i+1}/{Nw // batch_size_on_GPU}")
        batch_result = gpu_accelerated_experiment.execute_batch().cpu()

        # only one input, so squeeze the input dimension:
        batch_result = torch.squeeze(batch_result)

        start = i * batch_size_on_GPU
        end = start + batch_size_on_GPU
        preactivations_aggregate[start:end, :, :] = batch_result

    torch.save(preactivations_aggregate, get_cache_file_name(params, name="z_aggr"))

    gpu_accelerated_experiment.dispose()
