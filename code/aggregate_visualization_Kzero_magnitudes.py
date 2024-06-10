from math import pi, sqrt
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
import torch
import os
import numpy as np

base_folder = "/Users/maartenvt/Library/CloudStorage/GoogleDrive-maarten0110@gmail.com/My Drive/uni/master/mep/colab_cache_and_output"

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
        D_tanh_list,
        D_tanh_modified_list,
        D_sin_list,
        D_sin_modified_list,
        D_sigmoid_shifted_list,
        D_poly1_list,
        D_poly2_list,
        D_q_tanh_list,
        D_q_tanh_modified_list,
        D_q_sin_list,
        D_q_sin_modified_list,
        D_q_sigmoid_shifted_list,
        D_q_poly1_list,
        D_q_poly2_list,
        log_scale=False):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.0)
    angles_as_floats = [float(x) for x in angles]
    # FL = first layer, LL = last layer
    D_mean_FL_tanh = [D[0].item() for D in  D_tanh_list]
    D_mean_LL_tanh = [D[-1].item() for D in D_tanh_list]
    D_mean_FL_tanh_modified = [D[0].item() for D in  D_tanh_modified_list]  
    D_mean_LL_tanh_modified = [D[-1].item() for D in D_tanh_modified_list]
    D_mean_FL_sin = [D[0].item() for D in  D_sin_list]
    D_mean_LL_sin = [D[-1].item() for D in D_sin_list]
    D_mean_FL_sin_modified = [D[0].item() for D in  D_sin_modified_list]  
    D_mean_LL_sin_modified = [D[-1].item() for D in D_sin_modified_list]
    D_mean_FL_sigmoid_shifted = [D[0].item() for D in  D_sigmoid_shifted_list]
    D_mean_LL_sigmoid_shifted = [D[-1].item() for D in D_sigmoid_shifted_list]
    D_mean_FL_poly1 = [D[0].item() for D in  D_poly1_list]  
    D_mean_LL_poly1 = [D[-1].item() for D in D_poly1_list]
    D_mean_FL_poly2 = [D[0].item() for D in  D_poly2_list]  
    D_mean_LL_poly2 = [D[-1].item() for D in D_poly2_list]

    
    D_q_2_5_LL_tanh = [D[0, -1].item() for D in D_q_tanh_list]
    D_q_2_5_LL_tanh_modified = [D[0, -1].item() for D in D_q_tanh_modified_list]
    D_q_2_5_LL_sin = [D[0, -1].item() for D in D_q_sin_list]
    D_q_2_5_LL_sin_modified = [D[0, -1].item() for D in D_q_sin_modified_list]
    D_q_2_5_LL_sigmoid_shifted = [D[0, -1].item() for D in D_q_sigmoid_shifted_list]
    D_q_2_5_LL_poly1 = [D[0, -1].item() for D in D_q_poly1_list]
    D_q_2_5_LL_poly2 = [D[0, -1].item() for D in D_q_poly2_list]

    D_q_97_5_LL_tanh = [D[-1, -1].item() for D in D_q_tanh_list]
    D_q_97_5_LL_tanh_modified = [D[-1, -1].item() for D in D_q_tanh_modified_list]
    D_q_97_5_LL_sin = [D[-1, -1].item() for D in D_q_sin_list]
    D_q_97_5_LL_sin_modified = [D[-1, -1].item() for D in D_q_sin_modified_list]
    D_q_97_5_LL_sigmoid_shifted = [D[-1, -1].item() for D in D_q_sigmoid_shifted_list]
    D_q_97_5_LL_poly1 = [D[-1, -1].item() for D in D_q_poly1_list]
    D_q_97_5_LL_poly2 = [D[-1, -1].item() for D in D_q_poly2_list]

    sns.lineplot(label=labels[0], x=angles_as_floats, y=D_mean_LL_tanh,             ls="-",  color="red", linewidth=5)
    sns.lineplot(label=labels[1], x=angles_as_floats, y=D_mean_LL_tanh_modified,    ls=":",  color="red", linewidth=6)
    sns.lineplot(label=labels[2], x=angles_as_floats, y=D_mean_LL_sin,              ls="-",  color="blue")
    sns.lineplot(label=labels[3], x=angles_as_floats, y=D_mean_LL_sin_modified,     ls="--", color="blue", linewidth = 3)
    sns.lineplot(label=labels[4], x=angles_as_floats, y=D_mean_LL_sigmoid_shifted,  ls="-",  color="green")
    sns.lineplot(label=labels[5], x=angles_as_floats, y=D_mean_LL_poly1,            ls="-",  color="gray", linewidth=3)
    sns.lineplot(label=labels[6], x=angles_as_floats, y=D_mean_LL_poly2,            ls=":",  color="black", linewidth=2)

    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_tanh, D_q_2_5_LL_tanh, D_q_97_5_LL_tanh, dots_color="red", interval_color="red")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_tanh_modified, D_q_2_5_LL_tanh_modified, D_q_97_5_LL_tanh_modified, dots_color="red", interval_color="red")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_sin, D_q_2_5_LL_sin, D_q_97_5_LL_sin, dots_color="blue", interval_color="blue")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_sin_modified, D_q_2_5_LL_sin_modified, D_q_97_5_LL_sin_modified, dots_color="blue", interval_color="blue")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_sigmoid_shifted, D_q_2_5_LL_sigmoid_shifted, D_q_97_5_LL_sigmoid_shifted, dots_color="green", interval_color="green")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_poly1, D_q_2_5_LL_poly1, D_q_97_5_LL_poly1, dots_color="gray", interval_color="gray")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_LL_poly2, D_q_2_5_LL_poly2, D_q_97_5_LL_poly2, dots_color="black", interval_color="black")

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("angle between inputs")
    plt.title("$D^{(l)}$ for the first layer ($l=0$) and the last layer ($l=L$)\nvs. the angle between the inputs. Includes 2.5% and 97.5% quantiles.")
    if log_scale:
        fig.savefig(out_folder_e07 + f"/D_vs_angle_log.png")
    else: 
        fig.savefig(out_folder_e07 + f"/D_vs_angle.png")

    D_mean_ratio_tanh = [y/x for (x,y) in zip(D_mean_FL_tanh, D_mean_LL_tanh)]
    D_mean_ratio_tanh_modified = [y/x for (x,y) in zip(D_mean_FL_tanh_modified, D_mean_LL_tanh_modified)]
    D_mean_ratio_sin = [y/x for (x,y) in zip(D_mean_FL_sin, D_mean_LL_sin)]
    D_mean_ratio_sin_modified = [y/x for (x,y) in zip(D_mean_FL_sin_modified, D_mean_LL_sin_modified)]
    D_mean_ratio_sigmoid_shifted = [y/x for (x,y) in zip(D_mean_FL_sigmoid_shifted, D_mean_LL_sigmoid_shifted)]
    D_mean_ratio_poly1 = [y/x for (x,y) in zip(D_mean_FL_poly1, D_mean_LL_poly1)]
    D_mean_ratio_poly2 = [y/x for (x,y) in zip(D_mean_FL_poly2, D_mean_LL_poly2)]

    D_q_2_5_ratio_tanh = [q/base for (q, base) in zip(D_q_2_5_LL_tanh, D_mean_FL_tanh)]
    D_q_2_5_ratio_tanh_modified = [q/base for (q, base) in zip(D_q_2_5_LL_tanh_modified, D_mean_FL_tanh_modified)]
    D_q_2_5_ratio_sin = [q/base for (q, base) in zip(D_q_2_5_LL_sin, D_mean_FL_sin)]
    D_q_2_5_ratio_sin_modified = [q/base for (q, base) in zip(D_q_2_5_LL_sin_modified, D_mean_FL_sin_modified)]
    D_q_2_5_ratio_sigmoid_shifted = [q/base for (q, base) in zip(D_q_2_5_LL_sigmoid_shifted, D_mean_FL_sigmoid_shifted)]
    D_q_2_5_ratio_poly1 = [q/base for (q, base) in zip(D_q_2_5_LL_poly1, D_mean_FL_poly1)]
    D_q_2_5_ratio_poly2 = [q/base for (q, base) in zip(D_q_2_5_LL_poly2, D_mean_FL_poly2)]

    D_q_97_5_ratio_tanh = [q/base for (q, base) in zip(D_q_97_5_LL_tanh, D_mean_FL_tanh)]
    D_q_97_5_ratio_tanh_modified = [q/base for (q, base) in zip(D_q_97_5_LL_tanh_modified, D_mean_FL_tanh_modified)]
    D_q_97_5_ratio_sin = [q/base for (q, base) in zip(D_q_97_5_LL_sin, D_mean_FL_sin)]
    D_q_97_5_ratio_sin_modified = [q/base for (q, base) in zip(D_q_97_5_LL_sin_modified, D_mean_FL_sin_modified)]
    D_q_97_5_ratio_sigmoid_shifted = [q/base for (q, base) in zip(D_q_97_5_LL_sigmoid_shifted, D_mean_FL_sigmoid_shifted)]
    D_q_97_5_ratio_poly1 = [q/base for (q, base) in zip(D_q_97_5_LL_poly1, D_mean_FL_poly1)]
    D_q_97_5_ratio_poly2 = [q/base for (q, base) in zip(D_q_97_5_LL_poly2, D_mean_FL_poly2)]

    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.0)

    sns.lineplot(label=labels[0], x=angles_as_floats, y=D_mean_ratio_tanh,             ls="-",  color="red", linewidth=5)
    sns.lineplot(label=labels[1], x=angles_as_floats, y=D_mean_ratio_tanh_modified,    ls=":",  color="red", linewidth=6)
    sns.lineplot(label=labels[2], x=angles_as_floats, y=D_mean_ratio_sin,              ls="-",  color="blue")
    sns.lineplot(label=labels[3], x=angles_as_floats, y=D_mean_ratio_sin_modified,     ls="--", color="blue", linewidth = 3)
    sns.lineplot(label=labels[4], x=angles_as_floats, y=D_mean_ratio_sigmoid_shifted,  ls="-",  color="green")
    sns.lineplot(label=labels[5], x=angles_as_floats, y=D_mean_ratio_poly1,            ls="-",  color="gray", linewidth=3)
    sns.lineplot(label=labels[6], x=angles_as_floats, y=D_mean_ratio_poly2,            ls=":",  color="black", linewidth=2)

    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_tanh, D_q_2_5_ratio_tanh, D_q_97_5_ratio_tanh, dots_color="red", interval_color="red")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_tanh_modified, D_q_2_5_ratio_tanh_modified, D_q_97_5_ratio_tanh_modified, dots_color="red", interval_color="red")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_sin, D_q_2_5_ratio_sin, D_q_97_5_ratio_sin, dots_color="blue", interval_color="blue")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_sin_modified, D_q_2_5_ratio_sin_modified, D_q_97_5_ratio_sin_modified, dots_color="blue", interval_color="blue")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_sigmoid_shifted, D_q_2_5_ratio_sigmoid_shifted, D_q_97_5_ratio_sigmoid_shifted, dots_color="green", interval_color="green")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_poly1, D_q_2_5_ratio_poly1, D_q_97_5_ratio_poly1, dots_color="gray", interval_color="gray")
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_poly2, D_q_2_5_ratio_poly2, D_q_97_5_ratio_poly2, dots_color="black", interval_color="black")

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("angle between inputs")
    plt.title("ratio $D^{(L)}/D^{(0)}$ of the first layer ($l=0$) and the last layer ($l=L$)\nvs. the angle between the inputs. Includes 2.5% and 97.5% quantiles.")
    if log_scale:
        fig.savefig(out_folder_e07 + f"/D_vs_angle_relative_log.png")
    else: 
        fig.savefig(out_folder_e07 + f"/D_vs_angle_relative.png")

def K_to_input_norm(K, n):
    return sqrt(n*K)

if __name__ == "__main__":
    labels = [
        "tanh",
        "modified tanh, $\\beta=0.05$",
        "sin",
        "modified sin, $\\beta=0.05$",
        "shifted sigmoid",
        "custom polynomial 1",
        "custom polynomial 2",
    ]
    Cws = [
        "1.0006666666666666",
        "400.2666666666666",
        "1.0006666666666666",
        "400.2666666666666",
        "16.010666666666665",
        "1.0006666666666666",
        "1.0006666666666666",
    ]
    file_bases = [
        "tanh",
        "tanh_b=0.05",
        "sin",
        "sin_b=0.05",
        "sigmoid-shifted",
        "poly_eps=0.01",
        "poly2_eps1=0.01_eps2=0.01",
    ]
    Ks = [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
    ]
    n = 1000 # layer width

    # [e05]
    cache_folder_e05 = base_folder + "/e05/cache"
    out_folder_e05 = "code/experiments/e05/out/aggregate/Kzero_magnitudes"
    os.makedirs(out_folder_e05, exist_ok=True)

    colors_255 = [
        (154, 0, 0),
        (243, 153, 153),
        (17, 85, 204),
        (164, 194, 244),
        (56, 118, 29),
        (241, 195, 51),
        (142, 125, 195),
    ]

    colors = [
        (x[0]/255, x[1]/255, x[2]/255) for x in colors_255
    ]

    for K in Ks:

        norms_tanh              = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=tanh_norm={K_to_input_norm(K, n)}.pt")
        norms_tanh_modified     = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=tanh_b=0.05_norm={K_to_input_norm(K, n)}.pt")
        norms_sin               = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=sin_norm={K_to_input_norm(K, n)}.pt")
        norms_sin_modified      = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=sin_b=0.05_norm={K_to_input_norm(K, n)}.pt")
        # norms_sigmoid_shifted   = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=16.010666666666665_Cb=None_Nw=10000_seed=0_act=sigmoid-shifted_norm={K_to_input_norm(K, n)}.pt")
        norms_poly1             = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly_eps=0.01_norm={K_to_input_norm(K, n)}.pt")
        norms_poly2             = torch.load(cache_folder_e05 + f"/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly2_eps1=0.01_eps2=0.01_norm={K_to_input_norm(K, n)}.pt")

        quant_tanh              = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=tanh_norm={K_to_input_norm(K, n)}.pt")
        quant_tanh_modified     = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=tanh_b=0.05_norm={K_to_input_norm(K, n)}.pt")
        quant_sin               = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=sin_norm={K_to_input_norm(K, n)}.pt")
        quant_sin_modified      = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=sin_b=0.05_norm={K_to_input_norm(K, n)}.pt")
        # quant_sigmoid_shifted   = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=16.010666666666665_Cb=None_Nw=10000_seed=0_act=sigmoid-shifted_norm={K_to_input_norm(K, n)}.pt")
        quant_poly1             = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly_eps=0.01_norm={K_to_input_norm(K, n)}.pt")
        quant_poly2             = torch.load(cache_folder_e05 + f"/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly2_eps1=0.01_eps2=0.01_norm={K_to_input_norm(K, n)}.pt")

        L = norms_tanh.size()[0]
        ls = np.linspace(1, L, num=L)
        fig, ax = plt.subplots()
        sns.lineplot(label=labels[0], x=ls, y=norms_tanh,             color=colors[0])
        sns.lineplot(label=labels[1], x=ls, y=norms_tanh_modified,    color=colors[1])
        sns.lineplot(label=labels[2], x=ls, y=norms_sin,              color=colors[2])
        sns.lineplot(label=labels[3], x=ls, y=norms_sin_modified,     color=colors[3])
        # sns.lineplot(label=labels[4], x=ls, y=norms_sigmoid_shifted,  color=colors[4])            
        sns.lineplot(label=labels[5], x=ls, y=norms_poly1,            color=colors[5])
        sns.lineplot(label=labels[6], x=ls, y=norms_poly2,            color=colors[6])
        ax.set_ylim([0, 1.5*K_to_input_norm(K, n)])
        plt.title("The average preactivation norms for activation functions in\nthe $K^*_{00}=0$ universality class. " + "$K_{00}^{(l)} = " + str(K) + "$")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e05 + f"/norms_K={K}.png", dpi=300)

        # fig, ax = plt.subplots()
        # sns.lineplot(label=labels[0], x=ls, y=quant_tanh[0, :],                 ls="-",  color="red", linewidth=5)
        # sns.lineplot(label=labels[1], x=ls, y=quant_tanh_modified[0, :],        ls=":",  color="red", linewidth=6)
        # sns.lineplot(label=labels[2], x=ls, y=quant_sin[0, :],                  ls="-",  color="blue")
        # sns.lineplot(label=labels[3], x=ls, y=quant_sin_modified[0, :],         ls="--", color="blue", linewidth = 3)
        # # sns.lineplot(label=labels[4], x=ls, y=quant_sigmoid_shifted[0, :],      ls="-",  color="green")            
        # sns.lineplot(label=labels[5], x=ls, y=quant_poly1[0, :],                ls="-",  color="gray", linewidth=3)
        # sns.lineplot(label=labels[6], x=ls, y=quant_poly2[0, :],                ls=":",  color="black", linewidth=2)
        # ax.set_ylim([0, 1.5*K_to_input_norm(K, n)])
        # plt.title("2.5% quantile of the preactivations norms for activation functions in\nthe $K^*_{00}=0$ universality class. " + "$K_{00}^{(l)} = " + str(K) + "$")
        # ax.set_xlabel("layer")
        # fig.savefig(out_folder_e05 + f"/quantiles_2.5_K={K}.png")

        # fig, ax = plt.subplots()
        # sns.lineplot(label=labels[0], x=ls, y=quant_tanh[-1, :],                ls="-",  color="red", linewidth=5)
        # sns.lineplot(label=labels[1], x=ls, y=quant_tanh_modified[-1, :],       ls=":",  color="red", linewidth=6)
        # sns.lineplot(label=labels[2], x=ls, y=quant_sin[-1, :],                 ls="-",  color="blue")
        # sns.lineplot(label=labels[3], x=ls, y=quant_sin_modified[-1, :],        ls="--", color="blue", linewidth = 3)
        # # sns.lineplot(label=labels[4], x=ls, y=quant_sigmoid_shifted[-1, :],     ls="-",  color="green")
        # sns.lineplot(label=labels[5], x=ls, y=quant_poly1[-1, :],               ls="-",  color="gray", linewidth=3)
        # sns.lineplot(label=labels[6], x=ls, y=quant_poly2[-1, :],               ls=":",  color="black", linewidth=2)
        # ax.set_ylim([0, 1.5*K_to_input_norm(K, n)])
        # plt.title("97.5% quantile of the preactivations norms for activation functions in\nthe $K^*_{00}=0$ universality class. " + "$K_{00}^{(l)} = " + str(K) + "$")
        # ax.set_xlabel("layer")
        # fig.savefig(out_folder_e05 + f"/quantiles_97.5_K={K}.png")

        # fig, ax = plt.subplots()
        # sns.lineplot(label=labels[0], x=ls, y=quant_tanh[-1, :] - quant_tanh[0, :],                          ls="-",  color="red", linewidth=5)
        # sns.lineplot(label=labels[1], x=ls, y=quant_tanh_modified[-1, :] - quant_tanh_modified[0, :],        ls=":",  color="red", linewidth=6)
        # sns.lineplot(label=labels[2], x=ls, y=quant_sin[-1, :] - quant_sin[0, :],                            ls="-",  color="blue")
        # sns.lineplot(label=labels[3], x=ls, y=quant_sin_modified[-1, :] - quant_sin_modified[0, :],          ls="--", color="blue", linewidth = 3)
        # # sns.lineplot(label=labels[4], x=ls, y=quant_sigmoid_shifted[-1, :] - quant_sigmoid_shifted[0, :],    ls="-",  color="green")            
        # sns.lineplot(label=labels[5], x=ls, y=quant_poly1[-1, :] - quant_poly1[0, :],                        ls="-",  color="gray", linewidth=3)
        # sns.lineplot(label=labels[6], x=ls, y=quant_poly2[-1, :] - quant_poly2[0, :],                        ls=":",  color="black", linewidth=2)
        # ax.set_ylim([0, 1.5*K_to_input_norm(K, n)])
        # plt.title("Difference between the 97.5% and 2.5% quantile of the preactivations norms for activation\nfunctions in the $K^*_{00}=0$ universality class. " + "$K_{00}^{(l)} = " + str(K) + "$")
        # ax.set_xlabel("layer")
        # fig.savefig(out_folder_e05 + f"/quantiles_diff_95_K={K}.png")

