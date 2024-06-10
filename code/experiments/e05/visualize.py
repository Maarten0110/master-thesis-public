import torch
# torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logger import logV2 as log, error
sns.set_theme()

import os

def visualize(params, get_cache_file_name, get_out_file_name, job_id):
    L, n, Cw, Cb, Nw, seed, activation, input_norm = params
    torch.manual_seed(seed)

    input_file_norms_mean = get_cache_file_name(params, name="z_norms_mean")
    input_file_norms_quantiles = get_cache_file_name(params, name="z_norms_quantiles")
    output_file = get_out_file_name(params, name="out") 

    if not os.path.isfile(input_file_norms_mean) or not os.path.isfile(input_file_norms_quantiles):
        error(job_id, "Required input not found!")

    z_norms_mean = torch.load(input_file_norms_mean)
    z_norms_quantiles = torch.load(input_file_norms_quantiles)

    ls = np.linspace(1, L, num=L)
    q_95_lower = z_norms_quantiles[0, :]
    q_95_upper = z_norms_quantiles[-1, :]
    q_75_lower = z_norms_quantiles[1, :]
    q_75_upper = z_norms_quantiles[-2, :]
    q_50_lower = z_norms_quantiles[2, :]
    q_50_upper = z_norms_quantiles[-3, :]

    fig = plt.figure()
    plot = sns.lineplot(x = ls, y = z_norms_mean, color=(1, 0, 0, 1))
    plt.fill_between(ls, q_95_lower, q_95_upper, color=(1, 0, 0, 0.1))
    plt.fill_between(ls, q_75_lower, q_75_upper, color=(1, 0, 0, 0.1))
    plt.fill_between(ls, q_50_lower, q_50_upper, color=(1, 0, 0, 0.1))
    plot.set_title(f"Average preactivation norm (+ 95, 75 and 50 percent intervals) vs. depth,\n for activation \"{activation['file_base']}\"")
    # plot.set_ylim([0, 1.8])

    fig.savefig(output_file, bbox_inches='tight')
