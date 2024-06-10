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


def visualize(params, get_cache_file_name, get_out_file_name, job_id, use_logarithmic_scale, show=False):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Generating plots...")

    fpc_factor3_file = get_cache_file_name(L, n, Cw, Nw, seed, name="fpc_factor3")
    fpc_factor1_file = get_cache_file_name(L, n, Cw, Nw, seed, name="fpc_factor1")
    if not os.path.isfile(fpc_factor3_file) or not os.path.isfile(fpc_factor1_file):
        exit(1)
    else:
        fpc_factor3 = torch.load(fpc_factor3_file)
        fpc_factor1 = torch.load(fpc_factor1_file)

    title_suffix = f"\n$N_w={Nw}, L={L}, n={n}, C_w={Cw}, seed={seed}$"

    fig = plt.figure()
    G0 = 1
    layer_indices = np.arange(1, L+1)
    predicted_fpc_factor1 = ((1 + 2 / n) ** (layer_indices - 1)) * (((Cw ** layer_indices) * G0) ** 2)
    predicted_fpc_factor3 = 3 * predicted_fpc_factor1

    plot = sns.lineplot(x=layer_indices, y=fpc_factor1, label="\"factor 1\" elements")
    sns.lineplot(x=layer_indices, y=fpc_factor3, label="\"factor 3\" elements")
    sns.lineplot(x=layer_indices, y=predicted_fpc_factor1, label="\"factor 1\" elements (theoretical value)")
    sns.lineplot(x=layer_indices, y=predicted_fpc_factor3, label="\"factor 3\" elements (theoretical value)")
    title = f"Four point correlator analysis{title_suffix}"
    if use_logarithmic_scale:
        title += "\n$\\bf{Note}$: logarithmic scale on y-axis!"
    plot.set_title(title)
    if use_logarithmic_scale:
        plot.set_yscale("log")

    fig.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="4point"), bbox_inches='tight')

    if show:
        plt.show()
