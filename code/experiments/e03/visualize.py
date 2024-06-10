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

def get_bins(tensor):
    _min = torch.min(tensor).item()
    _max = torch.max(tensor).item()
    return np.linspace(_min - 0.05 * abs(_min), _max + 0.05 * abs(_max), num=50)

def visualize(params, get_cache_file_name, get_out_file_name, job_id, show=False):
    L, n, Cw, Nw, seed = params
    print(f"[job {job_id}] Generating plots...")

    input_file_tpc_zeros = get_cache_file_name(L, n, Cw, Nw, seed, name="tpc_zeros_e03")
    input_file_tpc_diagonal = get_cache_file_name(L, n, Cw, Nw, seed, name="tpc_diagonal_e03")
    input_file_fpc_factor1 = get_cache_file_name(L, n, Cw, Nw, seed, name="fpc_factor1_e03")
    input_file_fpc_factor3 = get_cache_file_name(L, n, Cw, Nw, seed, name="fpc_factor3_e03")
    
    if not os.path.isfile(input_file_tpc_zeros) \
        or not os.path.isfile(input_file_tpc_diagonal) \
        or not os.path.isfile(input_file_fpc_factor1) \
        or not os.path.isfile(input_file_fpc_factor3):
        exit(1)
    else:
        tpc_zeros = torch.load(input_file_tpc_zeros)
        tpc_diagonal = torch.load(input_file_tpc_diagonal)
        fpc_factor1 = torch.load(input_file_fpc_factor1)
        fpc_factor3 = torch.load(input_file_fpc_factor3)
    
    title_suffix = f"\n$N_w={Nw}, L={L}, n={n}, C_w={Cw}, seed={seed}$"

    # tpc - diagonal
    fig1 = plt.figure()
    GL = Cw ** L
    plot1 = sns.histplot(tpc_diagonal[0, :], color="blue", alpha = 0.2, stat="percent", label="first layer", edgecolor="none", bins=get_bins(tpc_diagonal))
    sns.histplot(tpc_diagonal[1, :], color="red", alpha = 0.2, stat="percent", label=f"last layer (L={L})", edgecolor="none", bins=get_bins(tpc_diagonal))
    plt.axvline(x=GL, color="red")
    plt.text(GL+0.01, 1, f"Theoretical expected value at L={L}", color="red")

    plot1.legend(loc="upper right") 
    plot1.set_title("Histogram of average of the nonzero elements of the\n2-point correlator of a neuron in the first and last layer" + title_suffix) #$$z_{i\\alpha}z_{i\\alpha}$$ 
    fig1.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="2point"), bbox_inches='tight')

    # tpc - zeros
    fig2 = plt.figure()
    plot2 = sns.histplot(tpc_zeros[0, :], color="blue", alpha = 0.2, stat="percent", label="first layer", edgecolor="none", bins=get_bins(tpc_zeros))
    sns.histplot(tpc_zeros[1, :], color="red", alpha = 0.2, stat="percent", label=f"last layer (L={L})", edgecolor="none", bins=get_bins(tpc_zeros))

    plot2.legend(loc="upper right")
    plot2.set_title("Histogram of average of the zero elements of the\n2-point correlator of a neuron in the first and last layer" + title_suffix) #$$z_{i\\alpha}z_{i\\beta}$$
    fig2.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="2point_zeros"), bbox_inches='tight')

    # fpc - factor 3
    fig3 = plt.figure()
    G4L = ((1 + 2 / n) ** (L - 1)) *  ((Cw ** L) ** 2)
    plot3 = sns.histplot(fpc_factor3[0, :], color="blue", alpha = 0.2, stat="percent", label="first layer", edgecolor="none", bins=get_bins(fpc_factor3))
    sns.histplot(fpc_factor3[1, :], color="red", alpha = 0.2, stat="percent", label=f"last layer (L={L})", edgecolor="none", bins=get_bins(fpc_factor3))
    plt.axvline(x=3*G4L, color="red")
    plt.text(3*G4L+0.01, 1, f"Theoretical expected value at L={L}", color="red")

    plot3.legend(loc="upper right")
    plot3.set_title("Histogram of average of the factor 3 elements of the\n4-point correlator of a neuron in the first and last layer" + title_suffix) #$$z_{i\\alpha}^2z_{i\\alpha}^2$$
    fig3.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="4point_factor3"), bbox_inches='tight')

    # fpc - factor 1
    fig4 = plt.figure()
    plot4 = sns.histplot(fpc_factor1[0, :], color="blue", alpha = 0.2, stat="percent", label="first layer", edgecolor="none", bins=get_bins(fpc_factor1))
    sns.histplot(fpc_factor1[1, :], color="red", alpha = 0.2, stat="percent", label=f"last layer (L={L})", edgecolor="none", bins=get_bins(fpc_factor1))
    plt.axvline(x=G4L, color="red")
    plt.text(G4L+0.01, 1, f"Theoretical expected value at L={L}", color="red")

    plot4.legend(loc="upper right")
    plot4.set_title("Histogram of average of the factor 1 elements of the\n4-point correlator of a neuron in the first and last layer" + title_suffix) #$$z_{i\\alpha}^2z_{i\\beta}^2$$
    fig4.savefig(get_out_file_name(L, n, Cw, Nw, seed, name="4point_factor1"), bbox_inches='tight')

    if show:
        plt.show()
