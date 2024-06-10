import torch
seed = 0
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

from .model import LinearNetwork
import os
from math import ceil
import seaborn as sns
sns.set_theme()


def generate(params, get_file_name, job_id):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Loading/generating data...")
    if not os.path.isfile(get_file_name(L, n, Cw, Nw, seed, name="activations_aggregate")):
        activations_aggregate = torch.zeros((Nw, L, n))
        reporting_interval = ceil(Nw / 4)
        for i in range(Nw):
            if i % reporting_interval == 0 or i == Nw-1:
                print(f"[job {job_id}] (generate) Progress: {i+1}/{Nw}")
            model = LinearNetwork(L, n, Cw)
            x = torch.ones((n,))
            _ = model(x)
            activations_aggregate[i, :, :] = model.activations
            model.dispose()

        torch.save(activations_aggregate, get_file_name(L, n, Cw, Nw, seed, name="activations_aggregate"))
