from math import pi
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
import torch
import os
import numpy as np

base_folder = "/Users/maartenvt/Library/CloudStorage/GoogleDrive-maarten0110@gmail.com/My Drive/uni/master/mep/colab_cache_and_output"

def Cw_scale_invariant_as_string(a_plus, a_min):
    A2 = ((a_plus ** 2) + (a_min ** 2)) / 2
    return f"{1 / A2}"

if __name__ == "__main__":

    labels_1 = [
        "$N_W = 10,000$",
        "$N_W =  7,500$",
        "$N_W =  5,000$",
        "$N_W =  2,500$",
        "$N_W =  1,000$",
    ]
    Nws = [10000, 7500, 5000, 2500, 1000]
    labels_2 = [
        "$n = 1000$",
        "$n =  750$",
        "$n =  500$",
        "$n =  250$",
        "$n =  100$",
    ]
    ns = [1000, 750, 500, 250, 100]
    

    # [e05]
    cache_folder_e05 = base_folder + "/e05/cache"
    out_folder_e05 = "code/experiments/e05/out/aggregate/SI/decay"
    os.makedirs(out_folder_e05, exist_ok=True)

    norms = []
    quants = []

    for Nw in Nws:
        norms.append(torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=2.0_Cb=None_Nw={Nw}_seed=0_act=relu_norm=1.pt"))
        quants.append(torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=2.0_Cb=None_Nw={Nw}_seed=0_act=relu_norm=1.pt"))

    L = norms[0].size()[0]
    ls = np.linspace(1, L, num=L)
    fig, ax = plt.subplots()
    for i in range(len(norms)):
        color_grade = 0.2+0.8*(i+1)/len(norms)
        sns.lineplot(x=ls, y=norms[i], label=labels_1[i], color=(color_grade, 0.1, 1-color_grade))
    ax.set_ylim([0.5, 1.1])
    plt.title("The average preactivation norms for scale-invariant\nactivation functions, averaged over $N_W$ initalizations.")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/norms_Nw.png")

    norms = []
    quants = []

    for n in ns:
        norms.append(torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n={n}_Cw=2.0_Cb=None_Nw=10000_seed=0_act=relu_norm=1.pt"))
        quants.append(torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n={n}_Cw=2.0_Cb=None_Nw=10000_seed=0_act=relu_norm=1.pt"))


    L = norms[0].size()[0]
    ls = np.linspace(1, L, num=L)
    fig, ax = plt.subplots()
    for i in range(len(norms)):
        color_grade = 0.2+0.8*(i+1)/len(norms)
        sns.lineplot(x=ls, y=norms[i], label=labels_2[i], color=(color_grade, 0.1, 1-color_grade))
    ax.set_ylim([0.5, 1.1])
    plt.title("The average preactivation norms for scale-invariant\nactivation functions, for an MLP of width $n$.")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/norms_layer_width.png")
