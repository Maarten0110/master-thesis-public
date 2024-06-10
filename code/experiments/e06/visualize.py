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
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    torch.manual_seed(seed)

    input_file_R_mean = get_cache_file_name(params, name="R_mean")
    input_file_R_quantiles = get_cache_file_name(params, name="R_quantiles")
    input_file_zs_mean = get_cache_file_name(params, name="zs_mean")
    input_file_zs_quantiles = get_cache_file_name(params, name="zs_quantiles")
    
    output_file_R = get_out_file_name(params, name="R")
    output_file_R_relative = get_out_file_name(params, name="R_rel")
    output_file_zs = get_out_file_name(params, name="zs")

    if not os.path.isfile(input_file_R_mean) \
        or not os.path.isfile(input_file_R_quantiles) \
        or not os.path.isfile(input_file_zs_mean) \
        or not os.path.isfile(input_file_zs_quantiles):
        error(job_id, "Required input not found!")

    R_mean = torch.load(input_file_R_mean)
    R_quantiles = torch.load(input_file_R_quantiles)
    R_q_95_lower = R_quantiles[0, :]
    R_q_95_upper = R_quantiles[-1, :]
    R_q_75_lower = R_quantiles[1, :]
    R_q_75_upper = R_quantiles[-2, :]
    R_q_50_lower = R_quantiles[2, :]
    R_q_50_upper = R_quantiles[-3, :]

    zs_norms_means = torch.load(input_file_zs_mean)
    zs_norms_quantiles = torch.load(input_file_zs_quantiles)

    ls = np.linspace(1, L, num=L)

    fig1 = plt.figure()
    plot = sns.lineplot(x = ls, y = R_mean, color=(1, 0, 0, 1))
    plt.fill_between(ls, R_q_95_lower, R_q_95_upper, color=(1, 0, 0, 0.1))
    plt.fill_between(ls, R_q_75_lower, R_q_75_upper, color=(1, 0, 0, 0.1))
    plt.fill_between(ls, R_q_50_lower, R_q_50_upper, color=(1, 0, 0, 0.1))
    plot.set_title(f"R (+ 95, 75 and 50 intervals) vs. depth,\n for activation \"{activation['file_base']}\"")
    fig1.savefig(output_file_R, bbox_inches='tight')

    fig2 = plt.figure()
    base = R_mean[0]
    plot = sns.lineplot(x = ls, y = R_mean / base, color=(1, 0, 0, 1))
    plt.fill_between(ls, R_q_95_lower / base, R_q_95_upper / base, color=(1, 0, 0, 0.1))
    plt.fill_between(ls, R_q_75_lower / base, R_q_75_upper / base, color=(1, 0, 0, 0.1))
    plt.fill_between(ls, R_q_50_lower / base, R_q_50_upper / base, color=(1, 0, 0, 0.1))
    plot.set_title(f"relative (to R at l=1) (95, 75 and 50 intervals)\nvs. depth, for activation \"{activation['file_base']}\"")
    fig2.savefig(output_file_R_relative, bbox_inches='tight')

    fig3 = plt.figure()
    plt.fill_between(ls, zs_norms_quantiles[0, 0, :], zs_norms_quantiles[-1, 0, :], color=(1, 0, 0, 0.2))
    plt.fill_between(ls, zs_norms_quantiles[0, 1, :], zs_norms_quantiles[-1, 1, :], color=(0, 0, 1, 0.2))
    
    plot = sns.lineplot(x = ls, y = zs_norms_means[0, :], color=(1, 0, 0, 1), label="input 1")
    plot = sns.lineplot(x = ls, y = zs_norms_means[1, :], color=(0, 0, 1, 1), label="input 2")
    plot.set_title(f"Average preactivation norms (+ 95 percent intervals) vs. depth,\n for activation \"{activation['file_base']}\"")
    fig3.savefig(output_file_zs, bbox_inches='tight')
    

