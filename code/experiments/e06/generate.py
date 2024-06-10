from utils import print_gpu_memory_info
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
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    torch.manual_seed(seed)
    
    if os.path.isfile(get_cache_file_name(params, name="zs_aggr")):
        log(job_id, "Data cached: skipping generation...")
        return
    
    assert Nw % batch_size_on_GPU == 0, "Check GPU batch size (must divide Nw)"
    print_gpu_memory_info("before 'generate'")

    log(job_id, "Generating data...")
    magnitude_of_single_element1 = math.sqrt(((midpoint_norm - midpoint_deviaton) ** 2) / n)
    magnitude_of_single_element2 = math.sqrt(((midpoint_norm + midpoint_deviaton) ** 2) / n)

    input1 = torch.full((n,), magnitude_of_single_element1)
    input2 = torch.full((n,), magnitude_of_single_element2)

    log(job_id, f"Input 1 norm: {torch.linalg.vector_norm(input1)}.")
    log(job_id, f"Input 2 norm: {torch.linalg.vector_norm(input2)}.")

    x = torch.cat((torch.unsqueeze(input1, dim=0), torch.unsqueeze(input2, dim=0)))
    log(job_id, f"The dimensions of the inputs tensor are: {x.size()}")

    preactivations_aggregate = torch.zeros((Nw, 2, L, n))
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

        start = i * batch_size_on_GPU
        end = start + batch_size_on_GPU
        preactivations_aggregate[start:end, :, :, :] = batch_result

    torch.save(preactivations_aggregate, get_cache_file_name(params, name="zs_aggr"))

    print_gpu_memory_info("After 'generate' but before clean up")
    gpu_accelerated_experiment.dispose()
    print_gpu_memory_info("After 'generate' and after clean up")
