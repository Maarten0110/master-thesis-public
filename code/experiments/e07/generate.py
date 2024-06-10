from utils import print_gpu_memory_info
import torch
# torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)
from logger import logV2 as log

from .model import MLPs_at_initialization_batched
import os
from math import ceil, sqrt, cos, sin, acos
import seaborn as sns
sns.set_theme()


def generate(params, get_cache_file_name, job_id, batch_size_on_GPU = None):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    torch.manual_seed(seed)
    
    if os.path.isfile(get_cache_file_name(params, name="zs_aggr")):
        log(job_id, "Data cached: skipping generation...")
        return

    assert Nw % batch_size_on_GPU == 0, "Check GPU batch size (must divide Nw)"
    print_gpu_memory_info("before 'generate'")

    log(job_id, "Generating data...")
    input1 = inputs_norm * torch.tensor([sqrt(2)/2, sqrt(2)/2])
    t = inputs_angle
    rotation_matrix = torch.tensor([[cos(t), -sin(t)], [sin(t), cos(t)]])
    input2 = rotation_matrix @ input1

    norm1 = torch.linalg.vector_norm(input1)
    norm2 = torch.linalg.vector_norm(input2)
    log(job_id, f"VERIFICATION: input 1 norm: {'{:02.2f}'.format(norm1)} (expected {'{:02.2f}'.format(inputs_norm)}).")
    log(job_id, f"VERIFICATION: input 2 norm: {'{:02.2f}'.format(norm2)} (expected {'{:02.2f}'.format(inputs_norm)}).")
    cos_of_angle = torch.dot(input1, input2) / (norm1 * norm2)
    if abs(cos_of_angle) > 1:
        log(job_id, f"WARNING: abs(cos(angle)) > 1, clamping! Actual value: {cos_of_angle}")
        cos_of_angle = torch.sign(cos_of_angle)
    angle = acos(cos_of_angle)
    log(job_id, f"VERIFICATION: angle between inputs: {'{:02.2f}'.format(angle)}. (expected {'{:02.2f}'.format(inputs_angle)})")

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
