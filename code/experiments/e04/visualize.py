import torch
seed = 0
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

def visualize(params, get_cache_file_name, get_out_file_name, job_id, show=False):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Generating plots...")

    input_file_cfpc = get_cache_file_name(L, n, Cw, Nw, seed, name="cfpc_e04")

    if not os.path.isfile(input_file_cfpc):
        exit(1)
    else:
        cfpc = torch.load(input_file_cfpc)

    # cfpc
    layer_indices = np.arange(1, L+1)
    G0 = 1
    cfpc_theoretical = (((1 + 2 / n) ** (layer_indices - 1)) - 1) * (((Cw ** layer_indices) * G0) ** 2)
    cfpc_emperical = torch.sum(cfpc, dim=(1, 2)) / (n * n + 2*n)
    
    fig1 = plt.figure()
    plot1 = sns.lineplot(x=layer_indices, y=cfpc_emperical, label="emperical")
    sns.lineplot(x=layer_indices, y=cfpc_theoretical, label="theoretical")
    plot1.set_title("Connected 4-point correlator analysis:\n $G^{(l)}_4 - (G^{(l)}_2)^2$ for each layer"
                        + f"\n$N_w={Nw}, L={L}, n={n}, C_w={Cw}, seed={seed}$")
    plot1.set_xlabel("layer")
    fig1.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="cfpc"), bbox_inches='tight')

    if show:
        plt.show()
