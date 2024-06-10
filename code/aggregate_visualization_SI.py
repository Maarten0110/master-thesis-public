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

def plot_dots_and_confidence_intervals(xs, ys, lower, upper, linewidth=0.5, dots_color=None, interval_color='black'):
    for x, y, bottom, top in zip(xs, ys, lower, upper):
        plt.plot([x, x], [top, bottom], color=interval_color, linewidth=linewidth)
        plt.plot(x, top, '_', color=interval_color)
        plt.plot(x, bottom, '_', color=interval_color)
        plt.plot(x, y, 'o', color=dots_color)

def plot_e07_angle_plots(
        labels,
        out_folder_e07,
        angles,
        D_linear_list,
        D_abs_list,
        D_relu_list,
        D_leaky_list,
        D_leaky2_list,
        D_leaky3_list,
        D_leaky4_list,
        D_leaky5_list,
        D_leaky6_list,
        D_q_linear_list,
        D_q_abs_list,
        D_q_relu_list,
        D_q_leaky_list,
        D_q_leaky2_list,
        D_q_leaky3_list,
        D_q_leaky4_list,
        D_q_leaky5_list,
        D_q_leaky6_list,
        log_scale=False):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.0)
    angles_as_floats = [float(x) for x in angles]
    # FL = first layer, LL = last layer
    D_mean_FL_linear = [D[0].item() for D in D_linear_list]
    D_mean_LL_linear = [D[-1].item() for D in D_linear_list]
    D_mean_FL_abs = [D[0].item() for D in D_abs_list]
    D_mean_LL_abs = [D[-1].item() for D in D_abs_list]
    D_mean_FL_relu = [D[0].item() for D in D_relu_list]
    D_mean_LL_relu = [D[-1].item() for D in D_relu_list]
    D_mean_FL_leaky = [D[0].item() for D in D_leaky_list]
    D_mean_LL_leaky = [D[-1].item() for D in D_leaky_list]
    D_mean_FL_leaky2 = [D[0].item() for D in D_leaky2_list]
    D_mean_LL_leaky2 = [D[-1].item() for D in D_leaky2_list]
    D_mean_FL_leaky3 = [D[0].item() for D in D_leaky3_list]
    D_mean_LL_leaky3 = [D[-1].item() for D in D_leaky3_list]
    D_mean_FL_leaky4 = [D[0].item() for D in D_leaky4_list]
    D_mean_LL_leaky4 = [D[-1].item() for D in D_leaky4_list]
    D_mean_FL_leaky5 = [D[0].item() for D in D_leaky5_list]
    D_mean_LL_leaky5 = [D[-1].item() for D in D_leaky5_list]
    D_mean_FL_leaky6 = [D[0].item() for D in D_leaky6_list]
    D_mean_LL_leaky6 = [D[-1].item() for D in D_leaky6_list]

    D_q_2_5_LL_linear = [D[0, -1].item() for D in D_q_linear_list]
    D_q_2_5_LL_abs =    [D[0, -1].item() for D in D_q_abs_list]    
    D_q_2_5_LL_relu =   [D[0, -1].item() for D in D_q_relu_list]    
    D_q_2_5_LL_leaky =  [D[0, -1].item() for D in D_q_leaky_list]    
    D_q_2_5_LL_leaky2 = [D[0, -1].item() for D in D_q_leaky2_list]    
    D_q_2_5_LL_leaky3 = [D[0, -1].item() for D in D_q_leaky3_list]    
    D_q_2_5_LL_leaky4 = [D[0, -1].item() for D in D_q_leaky4_list]    
    D_q_2_5_LL_leaky5 = [D[0, -1].item() for D in D_q_leaky5_list]    
    D_q_2_5_LL_leaky6 = [D[0, -1].item() for D in D_q_leaky6_list]

    D_q_97_5_LL_linear = [D[-1, -1].item() for D in D_q_linear_list]
    D_q_97_5_LL_abs =    [D[-1, -1].item() for D in D_q_abs_list]    
    D_q_97_5_LL_relu =   [D[-1, -1].item() for D in D_q_relu_list]    
    D_q_97_5_LL_leaky =  [D[-1, -1].item() for D in D_q_leaky_list]    
    D_q_97_5_LL_leaky2 = [D[-1, -1].item() for D in D_q_leaky2_list]    
    D_q_97_5_LL_leaky3 = [D[-1, -1].item() for D in D_q_leaky3_list]    
    D_q_97_5_LL_leaky4 = [D[-1, -1].item() for D in D_q_leaky4_list]    
    D_q_97_5_LL_leaky5 = [D[-1, -1].item() for D in D_q_leaky5_list]    
    D_q_97_5_LL_leaky6 = [D[-1, -1].item() for D in D_q_leaky6_list]

    sns.lineplot(label="$D^{(L)}$, " + labels[0],  ls="-",  markers=True, color="red", x=angles_as_floats, y=D_mean_LL_linear)
    sns.lineplot(label="$D^{(L)}$, " + labels[1],  ls="-",  markers=True, color="blue", x=angles_as_floats, y=D_mean_LL_abs)
    sns.lineplot(label="$D^{(L)}$, " + labels[2],  ls="-",  markers=True, color="green", x=angles_as_floats, y=D_mean_LL_relu)
    sns.lineplot(label="$D^{(L)}$, " + labels[3],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_LL_leaky,  color=(0.05, 0, 0.05))
    sns.lineplot(label="$D^{(L)}$, " + labels[4],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_LL_leaky2, color=(1/6, 0, 1/6))
    sns.lineplot(label="$D^{(L)}$, " + labels[5],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_LL_leaky4, color=(2/6, 0, 2/6))
    sns.lineplot(label="$D^{(L)}$, " + labels[6],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_LL_leaky5, color=(3/6, 0, 3/6))
    sns.lineplot(label="$D^{(L)}$, " + labels[7],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_LL_leaky6, color=(4/6, 0, 4/6))
    sns.lineplot(label="$D^{(L)}$, " + labels[8],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_LL_leaky3, color=(5/6, 0, 5/6))
    sns.lineplot(label="$D^{(0)}$, same for all $\sigma(\cdot)$", ls=":",  markers=True, color="black", x=angles_as_floats, y=D_mean_FL_linear)

    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_linear, D_q_2_5_LL_linear, D_q_97_5_LL_linear, dots_color="red", interval_color="red")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_abs,    D_q_2_5_LL_abs,    D_q_97_5_LL_abs, dots_color="blue", interval_color="blue")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_relu,   D_q_2_5_LL_relu,   D_q_97_5_LL_relu, dots_color="green", interval_color="green")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_leaky,  D_q_2_5_LL_leaky , D_q_97_5_LL_leaky, dots_color=(0.05, 0, 0.05), interval_color=(0.05, 0, 0.05))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_leaky2, D_q_2_5_LL_leaky2, D_q_97_5_LL_leaky2, dots_color=(1/6, 0, 1/6), interval_color=(1/6, 0, 1/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_leaky4, D_q_2_5_LL_leaky4, D_q_97_5_LL_leaky4, dots_color=(2/6, 0, 2/6), interval_color=(2/6, 0, 2/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_leaky5, D_q_2_5_LL_leaky5, D_q_97_5_LL_leaky5, dots_color=(3/6, 0, 3/6), interval_color=(3/6, 0, 3/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_leaky6, D_q_2_5_LL_leaky6, D_q_97_5_LL_leaky6, dots_color=(4/6, 0, 4/6), interval_color=(4/6, 0, 4/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_leaky3, D_q_2_5_LL_leaky3, D_q_97_5_LL_leaky3, dots_color=(5/6, 0, 5/6), interval_color=(5/6, 0, 5/6))

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("angle between inputs")
    plt.title("$D^{(l)}$ for the first layer ($l=0$) and the last layer ($l=L$)\nvs. the angle between the inputs. Includes 2.5% and 97.5% quantiles.")
    if log_scale:
        fig.savefig(out_folder_e07 + f"/D_vs_angle_log.png")
    else: 
        fig.savefig(out_folder_e07 + f"/D_vs_angle.png")

    D_mean_ratio_linear = [y/x for (x,y) in zip(D_mean_FL_linear, D_mean_LL_linear)]
    D_mean_ratio_abs = [y/x for (x,y) in zip(D_mean_FL_abs, D_mean_LL_abs)]
    D_mean_ratio_relu = [y/x for (x,y) in zip(D_mean_FL_relu, D_mean_LL_relu)]
    D_mean_ratio_leaky = [y/x for (x,y) in zip(D_mean_FL_leaky, D_mean_LL_leaky)]
    D_mean_ratio_leaky2 = [y/x for (x,y) in zip(D_mean_FL_leaky2, D_mean_LL_leaky2)]
    D_mean_ratio_leaky3 = [y/x for (x,y) in zip(D_mean_FL_leaky3, D_mean_LL_leaky3)]
    D_mean_ratio_leaky4 = [y/x for (x,y) in zip(D_mean_FL_leaky4, D_mean_LL_leaky4)]
    D_mean_ratio_leaky5 = [y/x for (x,y) in zip(D_mean_FL_leaky5, D_mean_LL_leaky5)]
    D_mean_ratio_leaky6 = [y/x for (x,y) in zip(D_mean_FL_leaky6, D_mean_LL_leaky6)]  

    D_q_2_5_ratio_linear = [q/base for (q, base) in zip(D_q_2_5_LL_linear, D_mean_FL_linear)]
    D_q_2_5_ratio_abs =    [q/base for (q, base) in zip(D_q_2_5_LL_abs,    D_mean_FL_abs)]
    D_q_2_5_ratio_relu =   [q/base for (q, base) in zip(D_q_2_5_LL_relu,   D_mean_FL_relu)]
    D_q_2_5_ratio_leaky =  [q/base for (q, base) in zip(D_q_2_5_LL_leaky,  D_mean_FL_leaky)]
    D_q_2_5_ratio_leaky2 = [q/base for (q, base) in zip(D_q_2_5_LL_leaky2, D_mean_FL_leaky2)]
    D_q_2_5_ratio_leaky3 = [q/base for (q, base) in zip(D_q_2_5_LL_leaky3, D_mean_FL_leaky3)]
    D_q_2_5_ratio_leaky4 = [q/base for (q, base) in zip(D_q_2_5_LL_leaky4, D_mean_FL_leaky4)]
    D_q_2_5_ratio_leaky5 = [q/base for (q, base) in zip(D_q_2_5_LL_leaky5, D_mean_FL_leaky5)]
    D_q_2_5_ratio_leaky6 = [q/base for (q, base) in zip(D_q_2_5_LL_leaky6, D_mean_FL_leaky6)]
    
    D_q_97_5_ratio_linear = [q/base for (q, base) in zip(D_q_97_5_LL_linear, D_mean_FL_linear)]
    D_q_97_5_ratio_abs =    [q/base for (q, base) in zip(D_q_97_5_LL_abs,    D_mean_FL_abs)]
    D_q_97_5_ratio_relu =   [q/base for (q, base) in zip(D_q_97_5_LL_relu,   D_mean_FL_relu)]
    D_q_97_5_ratio_leaky =  [q/base for (q, base) in zip(D_q_97_5_LL_leaky,  D_mean_FL_leaky)]
    D_q_97_5_ratio_leaky2 = [q/base for (q, base) in zip(D_q_97_5_LL_leaky2, D_mean_FL_leaky2)]
    D_q_97_5_ratio_leaky3 = [q/base for (q, base) in zip(D_q_97_5_LL_leaky3, D_mean_FL_leaky3)]
    D_q_97_5_ratio_leaky4 = [q/base for (q, base) in zip(D_q_97_5_LL_leaky4, D_mean_FL_leaky4)]
    D_q_97_5_ratio_leaky5 = [q/base for (q, base) in zip(D_q_97_5_LL_leaky5, D_mean_FL_leaky5)]
    D_q_97_5_ratio_leaky6 = [q/base for (q, base) in zip(D_q_97_5_LL_leaky6, D_mean_FL_leaky6)]

    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.0)
    sns.lineplot(label=labels[0],  ls="-",  markers=True, color="red", x=angles_as_floats, y=D_mean_ratio_linear)
    sns.lineplot(label=labels[1],  ls="-",  markers=True, color="blue", x=angles_as_floats, y=D_mean_ratio_abs)
    sns.lineplot(label=labels[2],  ls="-",  markers=True, color="green", x=angles_as_floats, y=D_mean_ratio_relu)
    sns.lineplot(label=labels[3],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_ratio_leaky,  color=(0.05, 0, 0.05))
    sns.lineplot(label=labels[4],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_ratio_leaky2, color=(1/6, 0, 1/6))
    sns.lineplot(label=labels[5],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_ratio_leaky4, color=(2/6, 0, 2/6))
    sns.lineplot(label=labels[6],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_ratio_leaky5, color=(3/6, 0, 3/6))
    sns.lineplot(label=labels[7],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_ratio_leaky6, color=(4/6, 0, 4/6))
    sns.lineplot(label=labels[8],  ls="-",  markers=True, x=angles_as_floats, y=D_mean_ratio_leaky3, color=(5/6, 0, 5/6))

    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_linear , D_q_2_5_ratio_linear , D_q_97_5_ratio_linear, dots_color="red", interval_color="red")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_abs , D_q_2_5_ratio_abs , D_q_97_5_ratio_abs, dots_color="blue", interval_color="blue")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_relu , D_q_2_5_ratio_relu , D_q_97_5_ratio_relu, dots_color="green", interval_color="green")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_leaky , D_q_2_5_ratio_leaky , D_q_97_5_ratio_leaky, dots_color=(0.05, 0, 0.05), interval_color=(0.05, 0, 0.05))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_leaky2, D_q_2_5_ratio_leaky2, D_q_97_5_ratio_leaky2, dots_color=(1/6, 0, 1/6), interval_color=(1/6, 0, 1/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_leaky4, D_q_2_5_ratio_leaky4, D_q_97_5_ratio_leaky4, dots_color=(2/6, 0, 2/6), interval_color=(2/6, 0, 2/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_leaky5, D_q_2_5_ratio_leaky5, D_q_97_5_ratio_leaky5, dots_color=(3/6, 0, 3/6), interval_color=(3/6, 0, 3/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_leaky6, D_q_2_5_ratio_leaky6, D_q_97_5_ratio_leaky6, dots_color=(4/6, 0, 4/6), interval_color=(4/6, 0, 4/6))
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_leaky3, D_q_2_5_ratio_leaky3, D_q_97_5_ratio_leaky3, dots_color=(5/6, 0, 5/6), interval_color=(5/6, 0, 5/6))

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("angle between inputs")
    plt.title("ratio $D^{(L)}/D^{(0)}$ of the first layer ($l=0$) and the last layer ($l=L$)\nvs. the angle between the inputs. Includes 2.5% and 97.5% quantiles.")
    if log_scale:
        fig.savefig(out_folder_e07 + f"/D_vs_angle_relative_log.png")
    else:
        fig.savefig(out_folder_e07 + f"/D_vs_angle_relative.png")

if __name__ == "__main__":

    labels = [
        "linear",
        "absolute value",
        "ReLU",
        "leaky ReLU ($a_-=0.1, a_+=1$)",
        "leaky ReLU ($a_-=0.4, a_+=1$)",
        "leaky ReLU ($a_-=0.5, a_+=1$)",
        "leaky ReLU ($a_-=0.75, a_+=1$)",
        "leaky ReLU ($a_-=0.9, a_+=1$)",
        "leaky ReLU ($a_-=0.95, a_+=1$)",
    ]

    # [e05]
    cache_folder_e05 = base_folder + "/e05/cache"
    out_folder_e05 = "code/experiments/e05/out/aggregate/SI"
    os.makedirs(out_folder_e05, exist_ok=True)

    norms_linear = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1_Cb=None_Nw=10000_seed=0_act=linear_norm=1.pt")
    norms_relu = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=2_Cb=None_Nw=10000_seed=0_act=relu_norm=1.pt")
    norms_abs = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1_Cb=None_Nw=10000_seed=0_act=s-i-abs_norm=1.pt")
    norms_leaky = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1.9801980198019802_Cb=None_Nw=10000_seed=0_act=leaky-relu_norm=1.pt")
    norms_leaky2 = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1.7241379310344827_Cb=None_Nw=10000_seed=0_act=leaky-relu2_norm=1.pt")
    
    quant_linear = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1_Cb=None_Nw=10000_seed=0_act=linear_norm=1.pt")
    quant_relu = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=2_Cb=None_Nw=10000_seed=0_act=relu_norm=1.pt")
    quant_abs = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1_Cb=None_Nw=10000_seed=0_act=s-i-abs_norm=1.pt")
    quant_leaky = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1.9801980198019802_Cb=None_Nw=10000_seed=0_act=leaky-relu_norm=1.pt")
    quant_leaky2 = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1.7241379310344827_Cb=None_Nw=10000_seed=0_act=leaky-relu2_norm=1.pt")

    L = norms_linear.size()[0]
    ls = np.linspace(1, L, num=L)
    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=norms_linear, color="red")
    sns.lineplot(label=labels[1], x=ls, y=norms_abs, color="blue")
    sns.lineplot(label=labels[2], x=ls, y=norms_relu, color="green")
    sns.lineplot(label=labels[3], x=ls, y=norms_leaky, color="orange")
    sns.lineplot(label=labels[4], x=ls, y=norms_leaky2, color="purple")
    
    plt.title("The average preactivation norms for scale-invariant\nactivation functions (zoomed in)")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/norms.png")

    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=quant_linear[0, :], ls="-", color="red")
    sns.lineplot(label=labels[1], x=ls, y=quant_abs[0, :],    ls="-", color="blue")
    sns.lineplot(label=labels[2], x=ls, y=quant_relu[0, :],   ls="-", color="green")
    sns.lineplot(label=labels[3], x=ls, y=quant_leaky[0, :],  ls="-", color="orange")
    sns.lineplot(label=labels[4], x=ls, y=quant_leaky2[0, :], ls="-", color="purple")
    plt.title("2.5% quantile of the preactivations norms\nfor scale-invariant activation functions")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/quantiles_2.5.png")

    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=quant_linear[-1, :], ls="-", color="red")
    sns.lineplot(label=labels[1], x=ls, y=quant_abs[-1, :],    ls="-", color="blue")
    sns.lineplot(label=labels[2], x=ls, y=quant_relu[-1, :],   ls="-", color="green")
    sns.lineplot(label=labels[3], x=ls, y=quant_leaky[-1, :],  ls="-", color="orange")
    sns.lineplot(label=labels[4], x=ls, y=quant_leaky2[-1, :], ls="-", color="purple")
    plt.title("97.5% quantile of the preactivations norms\nfor the scale-invariant activation functions")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/quantiles_97.5.png")

    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=norms_linear, color="red")
    sns.lineplot(label=labels[1], x=ls, y=norms_abs, color="blue")
    sns.lineplot(label=labels[2], x=ls, y=norms_relu, color="green")
    sns.lineplot(label=labels[3], x=ls, y=norms_leaky, color="orange")
    sns.lineplot(label=labels[4], x=ls, y=norms_leaky2, color="purple")
    sns.lineplot(x=ls, y=quant_linear[0, :], ls="--", color="red")
    sns.lineplot(x=ls, y=quant_abs[0, :],    ls="--", color="blue")
    sns.lineplot(x=ls, y=quant_relu[0, :],   ls="--", color="green")
    sns.lineplot(x=ls, y=quant_leaky[0, :],  ls="--", color="orange")
    sns.lineplot(x=ls, y=quant_leaky2[0, :], ls="--", color="purple")
    sns.lineplot(x=ls, y=quant_linear[-1, :], ls="--", color="red")
    sns.lineplot(x=ls, y=quant_abs[-1, :],    ls="--", color="blue")
    sns.lineplot(x=ls, y=quant_relu[-1, :],   ls="--", color="green")
    sns.lineplot(x=ls, y=quant_leaky[-1, :],  ls="--", color="orange")
    sns.lineplot(x=ls, y=quant_leaky2[-1, :], ls="--", color="purple")
    plt.title("The average preactivation norms and quantiles\nfor scale-invariant activation functions")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/combined.png")

    # [e06]
    cache_folder_e06 = base_folder + "/e06/cache"
    out_folder_e06 = "code/experiments/e06/out/aggregate/SI"
    os.makedirs(out_folder_e06, exist_ok=True)

    def R_file_name_mean(base, Cw, dev):
        return cache_folder_e06 + f"/R_mean_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_Mnorm=1_Mdev={dev}.pt"
    
    def R_file_name_quantiles(base, Cw, dev):
        return cache_folder_e06 + f"/R_quantiles_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_Mnorm=1_Mdev={dev}.pt"
    
    R_linear_list = []
    R_relu_list   = []
    R_abs_list    = []
    R_leaky_list  = []
    R_leaky2_list = []

    R_q_linear_list = []
    R_q_relu_list   = []
    R_q_abs_list    = []
    R_q_leaky_list  = []
    R_q_leaky2_list = []

    deviations = ["0.001", "0.01", "0.05", "0.5"]

    for dev in deviations:
        R_linear_list.append(torch.load(R_file_name_mean("linear",      "1.0", dev))) 
        R_relu_list  .append(torch.load(R_file_name_mean("relu",        "2.0", dev)))   
        R_abs_list   .append(torch.load(R_file_name_mean("s-i-abs",     "1.0", dev)))    
        R_leaky_list .append(torch.load(R_file_name_mean("leaky-relu",  "1.9801980198019802", dev)))  
        R_leaky2_list.append(torch.load(R_file_name_mean("leaky-relu2", "1.7241379310344827", dev)))

        R_q_linear_list.append(torch.load(R_file_name_quantiles("linear",      "1.0", dev))) 
        R_q_relu_list  .append(torch.load(R_file_name_quantiles("relu",        "2.0", dev)))   
        R_q_abs_list   .append(torch.load(R_file_name_quantiles("s-i-abs",     "1.0", dev)))    
        R_q_leaky_list .append(torch.load(R_file_name_quantiles("leaky-relu",  "1.9801980198019802", dev)))  
        R_q_leaky2_list.append(torch.load(R_file_name_quantiles("leaky-relu2", "1.7241379310344827", dev))) 

    L = R_linear_list[0].size()[0]
    ls = np.linspace(1, L, num=L)

    for i in range(len(deviations)):
        fig, ax = plt.subplots()
        sns.lineplot(label=labels[0], x=ls, y=R_linear_list[i], ls="-", color="red")
        sns.lineplot(label=labels[1], x=ls, y=R_abs_list[i]   , ls="-", color="blue")
        sns.lineplot(label=labels[2], x=ls, y=R_relu_list[i]  , ls="-", color="green")
        sns.lineplot(label=labels[3], x=ls, y=R_leaky_list[i] , ls="-", color="orange")
        sns.lineplot(label=labels[4], x=ls, y=R_leaky2_list[i], ls="-", color="purple")
        plt.title("$R^{(l)}$ for scale-invariant activation functions. $\\varepsilon_R = " + deviations[i] + "$")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e06 + f"/R_mean_dev={deviations[i]}.png")

        fig, ax = plt.subplots()
        sns.lineplot(label=labels[0], x=ls, y=R_q_linear_list[i][0, :], ls="-", color="red")
        sns.lineplot(label=labels[1], x=ls, y=R_q_abs_list[i][0, :]   , ls="-", color="blue")
        sns.lineplot(label=labels[2], x=ls, y=R_q_relu_list[i][0, :]  , ls="-", color="green")
        sns.lineplot(label=labels[3], x=ls, y=R_q_leaky_list[i][0, :] , ls="-", color="orange")
        sns.lineplot(label=labels[4], x=ls, y=R_q_leaky2_list[i][0, :], ls="-", color="purple")
        plt.title("2.5% quantile for the metric underlying $R^{(l)}$,\nfor various scale-invariant activation functions")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e06 + f"/R_q=2.5_dev={deviations[i]}.png")

        fig, ax = plt.subplots()
        sns.lineplot(label=labels[0], x=ls, y=R_q_linear_list[i][-1, :], ls="-", color="red")
        sns.lineplot(label=labels[1], x=ls, y=R_q_abs_list[i][-1, :]   , ls="-", color="blue")
        sns.lineplot(label=labels[2], x=ls, y=R_q_relu_list[i][-1, :]  , ls="-", color="green")
        sns.lineplot(label=labels[3], x=ls, y=R_q_leaky_list[i][-1, :] , ls="-", color="orange")
        sns.lineplot(label=labels[4], x=ls, y=R_q_leaky2_list[i][-1, :], ls="-", color="purple")
        plt.title("97.5% quantile for the metric underlying $R^{(l)}$,\nfor various scale-invariant activation functions")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e06 + f"/R_q=97.5_dev={deviations[i]}.png")

    # [e07]
    cache_folder_e07 = base_folder + "/e07/cache"
    out_folder_e07 = "code/experiments/e07/out/aggregate/SI"
    os.makedirs(out_folder_e07, exist_ok=True)

    def D_file_name_mean(base, Cw, angle):
        return cache_folder_e07 + f"/D_mean_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_norm=1_angle={angle}.pt"
    
    def D_file_name_quantiles(base, Cw, angle):
        return cache_folder_e07 + f"/D_quantiles_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_norm=1_angle={angle}.pt"

    D_linear_list = []
    D_relu_list   = []
    D_abs_list    = []
    D_leaky_list  = []
    D_leaky2_list = []
    D_leaky3_list = []
    D_leaky4_list = []
    D_leaky5_list = []
    D_leaky6_list = []

    D_q_linear_list = []
    D_q_relu_list   = []
    D_q_abs_list    = []
    D_q_leaky_list  = []
    D_q_leaky2_list = []
    D_q_leaky3_list = []
    D_q_leaky4_list = []
    D_q_leaky5_list = []
    D_q_leaky6_list = []

    angles = [
        "0.01227184630308513",
        "0.19634954084936207",
        "0.7853981633974483",
        "1.5707963267948966",
        "3.141592653589793",
    ]

    for angle in angles:
        D_linear_list.append(torch.load(D_file_name_mean("linear",      "1.0", angle))) 
        D_relu_list  .append(torch.load(D_file_name_mean("relu",        "2.0", angle)))   
        D_abs_list   .append(torch.load(D_file_name_mean("s-i-abs",     "1.0", angle)))    
        D_leaky_list .append(torch.load(D_file_name_mean("leaky-relu",  "1.9801980198019802", angle)))  
        D_leaky2_list.append(torch.load(D_file_name_mean("leaky-relu2", "1.7241379310344827", angle)))
        D_leaky3_list.append(torch.load(D_file_name_mean("leaky-relu3", "1.0512483574244416", angle)))
        D_leaky4_list.append(torch.load(D_file_name_mean("leaky-relu4", Cw_scale_invariant_as_string(1, 0.5),  angle)))
        D_leaky5_list.append(torch.load(D_file_name_mean("leaky-relu5", Cw_scale_invariant_as_string(1, 0.75), angle)))
        D_leaky6_list.append(torch.load(D_file_name_mean("leaky-relu6", Cw_scale_invariant_as_string(1, 0.9),  angle)))

        D_q_linear_list.append(torch.load(D_file_name_quantiles("linear",      "1.0", angle))) 
        D_q_relu_list  .append(torch.load(D_file_name_quantiles("relu",        "2.0", angle)))   
        D_q_abs_list   .append(torch.load(D_file_name_quantiles("s-i-abs",     "1.0", angle)))    
        D_q_leaky_list .append(torch.load(D_file_name_quantiles("leaky-relu",  "1.9801980198019802", angle)))  
        D_q_leaky2_list.append(torch.load(D_file_name_quantiles("leaky-relu2", "1.7241379310344827", angle)))
        D_q_leaky3_list.append(torch.load(D_file_name_quantiles("leaky-relu3", "1.0512483574244416", angle)))
        D_q_leaky4_list.append(torch.load(D_file_name_quantiles("leaky-relu4", Cw_scale_invariant_as_string(1, 0.5),  angle)))
        D_q_leaky5_list.append(torch.load(D_file_name_quantiles("leaky-relu5", Cw_scale_invariant_as_string(1, 0.75), angle)))
        D_q_leaky6_list.append(torch.load(D_file_name_quantiles("leaky-relu6", Cw_scale_invariant_as_string(1, 0.9),  angle)))

    L = D_linear_list[0].size()[0]
    ls = np.linspace(1, L, num=L)

    for i in range(len(angles)):
        angle_divided_by_pi = float(angles[i]) / pi
        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=D_linear_list[i], ls="-", color="red")
        sns.lineplot(label=labels[1], x=ls, y=D_abs_list[i]   , ls="-", color="blue")
        sns.lineplot(label=labels[2], x=ls, y=D_relu_list[i]  , ls="-", color="green")
        sns.lineplot(label=labels[3], x=ls, y=D_leaky_list[i] , ls="-", color=(0.05, 0, 0.05))
        sns.lineplot(label=labels[4], x=ls, y=D_leaky2_list[i], ls="-", color=(1/6, 0, 1/6))
        sns.lineplot(label=labels[5], x=ls, y=D_leaky4_list[i], ls="-", color=(2/6, 0, 2/6))
        sns.lineplot(label=labels[6], x=ls, y=D_leaky5_list[i], ls="-", color=(3/6, 0, 3/6))
        sns.lineplot(label=labels[7], x=ls, y=D_leaky6_list[i], ls="-", color=(4/6, 0, 4/6))
        sns.lineplot(label=labels[8], x=ls, y=D_leaky3_list[i], ls="-", color=(5/6, 0, 5/6))
        plt.title("$D^{(l)}$ for various scale-invariant activation functions\nangle between inputs = " + '{:02.3f}'.format(angle_divided_by_pi) + " π")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e07 + f"/D_mean_angle={angles[i]}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=D_q_linear_list[i][0, :], ls="-", color="red")
        sns.lineplot(label=labels[1], x=ls, y=D_q_abs_list[i][0, :]   , ls="-", color="blue")
        sns.lineplot(label=labels[2], x=ls, y=D_q_relu_list[i][0, :]  , ls="-", color="green")
        sns.lineplot(label=labels[3], x=ls, y=D_q_leaky_list[i][0, :] , ls="-", color=(0.05, 0, 0.05))
        sns.lineplot(label=labels[4], x=ls, y=D_q_leaky2_list[i][0, :], ls="-", color=(1/6, 0, 1/6))
        sns.lineplot(label=labels[5], x=ls, y=D_q_leaky4_list[i][0, :], ls="-", color=(2/6, 0, 2/6))
        sns.lineplot(label=labels[6], x=ls, y=D_q_leaky5_list[i][0, :], ls="-", color=(3/6, 0, 3/6))
        sns.lineplot(label=labels[7], x=ls, y=D_q_leaky6_list[i][0, :], ls="-", color=(4/6, 0, 4/6))
        sns.lineplot(label=labels[8], x=ls, y=D_q_leaky3_list[i][0, :], ls="-", color=(5/6, 0, 5/6))
        plt.title("2.5% quantile for the metric underlying $D^{(l)}$,\nfor various scale-invariant activation functions\nangle between inputs = " + '{:02.3f}'.format(angle_divided_by_pi) + " π")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e07 + f"/D_q=2.5_angle={angles[i]}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=D_q_linear_list[i][-1, :], ls="-", color="red")
        sns.lineplot(label=labels[1], x=ls, y=D_q_abs_list[i][-1, :]   , ls="-", color="blue")
        sns.lineplot(label=labels[2], x=ls, y=D_q_relu_list[i][-1, :]  , ls="-", color="green")
        sns.lineplot(label=labels[3], x=ls, y=D_q_leaky_list[i][-1, :] , ls="-", color=(0.05, 0, 0.05))
        sns.lineplot(label=labels[4], x=ls, y=D_q_leaky2_list[i][-1, :], ls="-", color=(1/6, 0, 1/6))
        sns.lineplot(label=labels[5], x=ls, y=D_q_leaky4_list[i][-1, :], ls="-", color=(2/6, 0, 2/6))
        sns.lineplot(label=labels[6], x=ls, y=D_q_leaky5_list[i][-1, :], ls="-", color=(3/6, 0, 3/6))
        sns.lineplot(label=labels[7], x=ls, y=D_q_leaky6_list[i][-1, :], ls="-", color=(4/6, 0, 4/6))
        sns.lineplot(label=labels[8], x=ls, y=D_q_leaky3_list[i][-1, :], ls="-", color=(5/6, 0, 5/6))
        plt.title("97.5% quantile for the metric underlying $D^{(l)}$,\nfor various scale-invariant activation functions\nangle between inputs = " + '{:02.3f}'.format(angle_divided_by_pi) + " π")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e07 + f"/D_q=97.5_angle={angles[i]}.png")

    plot_e07_angle_plots(
        labels,
        out_folder_e07,
        angles,
        D_linear_list,
        D_abs_list,
        D_relu_list,
        D_leaky_list,
        D_leaky2_list,
        D_leaky3_list,
        D_leaky4_list,
        D_leaky5_list,
        D_leaky6_list,
        D_q_linear_list,
        D_q_abs_list,
        D_q_relu_list,
        D_q_leaky_list,
        D_q_leaky2_list,
        D_q_leaky3_list,
        D_q_leaky4_list,
        D_q_leaky5_list,
        D_q_leaky6_list,
        log_scale=False,
    )
    plot_e07_angle_plots(
        labels,
        out_folder_e07,
        angles,
        D_linear_list,
        D_abs_list,
        D_relu_list,
        D_leaky_list,
        D_leaky2_list,
        D_leaky3_list,
        D_leaky4_list,
        D_leaky5_list,
        D_leaky6_list,
        D_q_linear_list,
        D_q_abs_list,
        D_q_relu_list,
        D_q_leaky_list,
        D_q_leaky2_list,
        D_q_leaky3_list,
        D_q_leaky4_list,
        D_q_leaky5_list,
        D_q_leaky6_list,
        log_scale=True,
    )
