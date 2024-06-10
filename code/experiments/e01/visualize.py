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

    input_file_activations = get_cache_file_name(L, n, Cw, Nw, seed, name="activations_aggregate")
    input_file_tpc_zeros = get_cache_file_name(L, n, Cw, Nw, seed, name="tpc_zeros")
    input_file_tpc_diagonal = get_cache_file_name(L, n, Cw, Nw, seed, name="tpc_diagonal")
    if not os.path.isfile(input_file_activations) or \
        not os.path.isfile(input_file_tpc_zeros) or \
        not os.path.isfile(input_file_tpc_diagonal):
        exit(1)
    else:
        activations_aggregate = torch.load(input_file_activations)
        tpc_zeros = torch.load(input_file_tpc_zeros)
        tpc_diagonal = torch.load(input_file_tpc_diagonal)
    
    title_suffix = f"\n$N_w={Nw}, L={L}, n={n}, C_w={Cw}, seed={seed}$"

    fig1 = plt.figure()
    activations_averaged_over_draws = torch.mean(activations_aggregate, dim=0)
    plot1 = sns.histplot(activations_averaged_over_draws.flatten().numpy(), bins=100, stat="percent")
    plot1.set_title(f"Frequency of activations (100 bins){title_suffix}")
    plot1.set(xlabel="Activation", ylabel="Percentage")

    fig2 = plt.figure()
    G0 = 1
    layer_indices = np.arange(1, L+1)
    predicted_nonzero = G0 * (Cw ** layer_indices)
    plot2 = sns.lineplot(x=layer_indices, y=tpc_diagonal, label="(should be) nonzero elements")
    sns.lineplot(x=layer_indices, y=tpc_zeros, label="(should be) zero elements")
    sns.lineplot(x=layer_indices, y=predicted_nonzero, label="nonzero elements (theoretical value)")
    title = f"Two point correlator analysis{title_suffix}"
    if use_logarithmic_scale:
        title += "\n$\\bf{Note}$: logarithmic scale on y-axis!"
    plot2.set_title(title)
    if use_logarithmic_scale:
        plot2.set_yscale("log")

    fig1.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="1point"), bbox_inches='tight')
    fig2.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="2point"), bbox_inches='tight')

    if show:
        plt.show()

