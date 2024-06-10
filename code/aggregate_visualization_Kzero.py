from math import pi
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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
        colors,
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
    sns.lineplot(label=labels[1], x=angles_as_floats, y=D_mean_LL_tanh_modified,    ls="-",  color="red", linewidth=6)
    sns.lineplot(label=labels[2], x=angles_as_floats, y=D_mean_LL_sin,              ls="-",  color="blue")
    sns.lineplot(label=labels[3], x=angles_as_floats, y=D_mean_LL_sin_modified,     ls="-",  color="blue", linewidth = 3)
    sns.lineplot(label=labels[4], x=angles_as_floats, y=D_mean_LL_sigmoid_shifted,  ls="-",  color="green")
    sns.lineplot(label=labels[5], x=angles_as_floats, y=D_mean_LL_poly1,            ls="-",  color="gray", linewidth=3)
    sns.lineplot(label=labels[6], x=angles_as_floats, y=D_mean_LL_poly2,            ls="-",  color="black", linewidth=2)

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

    sns.lineplot(label=labels[0], x=angles_as_floats, y=D_mean_ratio_tanh,             ls="-",  color=colors[0])
    sns.lineplot(label=labels[1], x=angles_as_floats, y=D_mean_ratio_tanh_modified,    ls="-",  color=colors[1])
    sns.lineplot(label=labels[2], x=angles_as_floats, y=D_mean_ratio_sin,              ls="-",  color=colors[2])
    sns.lineplot(label=labels[3], x=angles_as_floats, y=D_mean_ratio_sin_modified,     ls="-",  color=colors[3])
    sns.lineplot(label=labels[4], x=angles_as_floats, y=D_mean_ratio_sigmoid_shifted,  ls="-",  color=colors[4])
    sns.lineplot(label=labels[5], x=angles_as_floats, y=D_mean_ratio_poly1,            ls="-",  color=colors[5])
    sns.lineplot(label=labels[6], x=angles_as_floats, y=D_mean_ratio_poly2,            ls="-",  color=colors[6])

    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_tanh, D_q_2_5_ratio_tanh, D_q_97_5_ratio_tanh,                                    dots_color=colors[0], interval_color=colors[0])
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_tanh_modified, D_q_2_5_ratio_tanh_modified, D_q_97_5_ratio_tanh_modified,         dots_color=colors[1], interval_color=colors[1])
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_sin, D_q_2_5_ratio_sin, D_q_97_5_ratio_sin,                                       dots_color=colors[2], interval_color=colors[2])
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_sin_modified, D_q_2_5_ratio_sin_modified, D_q_97_5_ratio_sin_modified,            dots_color=colors[3], interval_color=colors[3])
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_sigmoid_shifted, D_q_2_5_ratio_sigmoid_shifted, D_q_97_5_ratio_sigmoid_shifted,   dots_color=colors[4], interval_color=colors[4])
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_poly1, D_q_2_5_ratio_poly1, D_q_97_5_ratio_poly1,                                 dots_color=colors[5], interval_color=colors[5])
    plot_dots_and_confidence_intervals(angles_as_floats, D_mean_ratio_poly2, D_q_2_5_ratio_poly2, D_q_97_5_ratio_poly2,                                 dots_color=colors[6], interval_color=colors[6])

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("angle between inputs")
    plt.title("ratio $D^{(L)}/D^{(0)}$ of the first layer ($l=0$) and the last layer ($l=L$)\nvs. the angle between the inputs. Includes 2.5% and 97.5% quantiles.")
    if log_scale:
        fig.savefig(out_folder_e07 + f"/D_vs_angle_relative_log.png", dpi=300)
    else: 
        fig.savefig(out_folder_e07 + f"/D_vs_angle_relative.png", dpi=300)

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

    # [e05]
    cache_folder_e05 = base_folder + "/e05/cache"
    out_folder_e05 = "code/experiments/e05/out/aggregate/Kzero"
    os.makedirs(out_folder_e05, exist_ok=True)

    norms_tanh              = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=tanh_norm=1.pt")
    norms_tanh_modified     = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=tanh_b=0.05_norm=1.pt")
    norms_sin               = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=sin_norm=1.pt")
    norms_sin_modified      = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=sin_norm=1.pt")
    norms_sigmoid_shifted   = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=16.010666666666665_Cb=None_Nw=10000_seed=0_act=sigmoid-shifted_norm=1.pt")
    norms_poly1             = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly_eps=0.01_norm=1.pt")
    norms_poly2             = torch.load(cache_folder_e05 + "/z_norms_mean_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly2_eps1=0.01_eps2=0.01_norm=1.pt")

    quant_tanh              = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=tanh_norm=1.pt")
    quant_tanh_modified     = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=tanh_b=0.05_norm=1.pt")
    quant_sin               = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=sin_norm=1.pt")
    quant_sin_modified      = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=400.2666666666666_Cb=None_Nw=10000_seed=0_act=sin_norm=1.pt")
    quant_sigmoid_shifted   = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=16.010666666666665_Cb=None_Nw=10000_seed=0_act=sigmoid-shifted_norm=1.pt")
    quant_poly1             = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly_eps=0.01_norm=1.pt")
    quant_poly2             = torch.load(cache_folder_e05 + "/z_norms_quantiles_L=100_n=1000_Cw=1.0006666666666666_Cb=None_Nw=10000_seed=0_act=poly2_eps1=0.01_eps2=0.01_norm=1.pt")


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

    L = norms_tanh.size()[0]
    ls = np.linspace(1, L, num=L)
    fig, ax = plt.subplots()

    # Define a list of markers for each line
    # markers = ["o", "s", "D", "^", "v", "<", ">"]
    markers = ["o", "o", "o", "o", "o", "o", "o"]

    # Define the interval for showing markers
    markevery = 10

    # Plot the lines without markers
    sns.lineplot(x=ls, y=norms_tanh,               ls="-",  color=colors[0], label='_nolegend_')
    sns.lineplot(x=ls, y=norms_tanh_modified,      ls="-",  color=colors[1], label='_nolegend_')
    sns.lineplot(x=ls, y=norms_sin,                ls="-",  color=colors[2], label='_nolegend_')
    sns.lineplot(x=ls, y=norms_sin_modified,       ls="-",  color=colors[3], label='_nolegend_')
    sns.lineplot(x=ls, y=norms_sigmoid_shifted,    ls="-",  color=colors[4], label='_nolegend_')
    sns.lineplot(x=ls, y=norms_poly1,              ls="-",  color=colors[5], label='_nolegend_')
    sns.lineplot(x=ls, y=norms_poly2,              ls="-",  color=colors[6], label='_nolegend_')

    # Add markers separately
    ax.plot(ls, norms_tanh, marker=markers[0],              markevery=(0, markevery), linestyle='-', color=colors[0])
    ax.plot(ls, norms_tanh_modified, marker=markers[1],     markevery=(0, markevery), linestyle='-', color=colors[1])
    ax.plot(ls, norms_sin, marker=markers[2],               markevery=(0, markevery), linestyle='-', color=colors[2])
    ax.plot(ls, norms_sin_modified, marker=markers[3],      markevery=(2, markevery), linestyle='-', color=colors[3])
    ax.plot(ls, norms_sigmoid_shifted, marker=markers[4],   markevery=(0, markevery), linestyle='-', color=colors[4])
    ax.plot(ls, norms_poly1, marker=markers[5],             markevery=(4, markevery), linestyle='-', color=colors[5])
    ax.plot(ls, norms_poly2, marker=markers[6],             markevery=(6, markevery), linestyle='-', color=colors[6])

    # Create custom legend
    legend_elements = [Line2D([0], [0], color=colors[i], marker=markers[i], label=labels[i], markersize=8, linestyle='-') for i in range(len(labels))]
    ax.legend(handles=legend_elements)

    # plt.title("The average preactivation norms and quantiles for activation\nfunctions in the $K^*_{00}=0$ universality class (zoomed in)")
    plt.title("The average preactivation norms and quantiles for activation\nfunctions in the $K^*_{00}=0$ universality class. " + "$K_{00}^{(l)} = 0.0316$")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/zoomed_in.png", dpi=300)

    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0],x=ls, y=norms_tanh,               ls="-",  color=colors[0])
    sns.lineplot(label=labels[1],x=ls, y=norms_tanh_modified,      ls="-",  color=colors[1])
    sns.lineplot(label=labels[2],x=ls, y=norms_sin,                ls="-",  color=colors[2])
    sns.lineplot(label=labels[3],x=ls, y=norms_sin_modified,       ls="-",  color=colors[3])
    sns.lineplot(label=labels[4],x=ls, y=norms_sigmoid_shifted,    ls="-",  color=colors[4])
    sns.lineplot(label=labels[5],x=ls, y=norms_poly1,              ls="-",  color=colors[5])
    sns.lineplot(label=labels[6],x=ls, y=norms_poly2,              ls="-",  color=colors[6])
    
    sns.lineplot(x=ls, y=quant_tanh[0, :],               ls="--", color=colors[0])
    sns.lineplot(x=ls, y=quant_tanh_modified[0, :],      ls="--", color=colors[1])
    sns.lineplot(x=ls, y=quant_sin[0, :],                ls="--", color=colors[2])
    sns.lineplot(x=ls, y=quant_sin_modified[0, :],       ls="--", color=colors[3])
    sns.lineplot(x=ls, y=quant_sigmoid_shifted[0, :],    ls="--", color=colors[4])            
    sns.lineplot(x=ls, y=quant_poly1[0, :],              ls="--", color=colors[5])
    sns.lineplot(x=ls, y=quant_poly2[0, :],              ls="--", color=colors[6])
    
    sns.lineplot(x=ls, y=quant_tanh[-1, :],               ls="--", color=colors[0])
    sns.lineplot(x=ls, y=quant_tanh_modified[-1, :],      ls="--", color=colors[1])
    sns.lineplot(x=ls, y=quant_sin[-1, :],                ls="--", color=colors[2])
    sns.lineplot(x=ls, y=quant_sin_modified[-1, :],       ls="--", color=colors[3])
    sns.lineplot(x=ls, y=quant_sigmoid_shifted[-1, :],    ls="--", color=colors[4])            
    sns.lineplot(x=ls, y=quant_poly1[-1, :],              ls="--", color=colors[5])
    sns.lineplot(x=ls, y=quant_poly2[-1, :],              ls="--", color=colors[6])

    plt.title("The average preactivation norms and quantiles for activation\nfunctions in the $K^*_{00}=0$ universality class")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/combined.png", dpi=300)


    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=quant_tanh[0, :],                 ls="-",  color="red", linewidth=5)
    sns.lineplot(label=labels[1], x=ls, y=quant_tanh_modified[0, :],        ls=":",  color="red", linewidth=6)
    sns.lineplot(label=labels[2], x=ls, y=quant_sin[0, :],                  ls="-",  color="blue")
    sns.lineplot(label=labels[3], x=ls, y=quant_sin_modified[0, :],         ls="--", color="blue", linewidth = 3)
    sns.lineplot(label=labels[4], x=ls, y=quant_sigmoid_shifted[0, :],      ls="-",  color="green")            
    sns.lineplot(label=labels[5], x=ls, y=quant_poly1[0, :],                ls="-",  color="gray", linewidth=3)
    sns.lineplot(label=labels[6], x=ls, y=quant_poly2[0, :],                ls=":",  color="black", linewidth=2)

    plt.title("2.5% quantile of the preactivations norms for activation\nfunctions in the $K^*_{00}=0$ universality class")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/quantiles_2.5.png")

    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=quant_tanh[-1, :],                ls="-",  color="red", linewidth=5)
    sns.lineplot(label=labels[1], x=ls, y=quant_tanh_modified[-1, :],       ls=":",  color="red", linewidth=6)
    sns.lineplot(label=labels[2], x=ls, y=quant_sin[-1, :],                 ls="-",  color="blue")
    sns.lineplot(label=labels[3], x=ls, y=quant_sin_modified[-1, :],        ls="--", color="blue", linewidth = 3)
    sns.lineplot(label=labels[4], x=ls, y=quant_sigmoid_shifted[-1, :],     ls="-",  color="green")
    sns.lineplot(label=labels[5], x=ls, y=quant_poly1[-1, :],               ls="-",  color="gray", linewidth=3)
    sns.lineplot(label=labels[6], x=ls, y=quant_poly2[-1, :],               ls=":",  color="black", linewidth=2)

    plt.title("97.5% quantile of the preactivations norms for activation\nfunctions in the $K^*_{00}=0$ universality class")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/quantiles_97.5.png")

    fig, ax = plt.subplots()
    sns.lineplot(label=labels[0], x=ls, y=quant_tanh[-1, :] - quant_tanh[0, :],                          ls="-",  color="red", linewidth=5)
    sns.lineplot(label=labels[1], x=ls, y=quant_tanh_modified[-1, :] - quant_tanh_modified[0, :],        ls=":",  color="red", linewidth=6)
    sns.lineplot(label=labels[2], x=ls, y=quant_sin[-1, :] - quant_sin[0, :],                            ls="-",  color="blue")
    sns.lineplot(label=labels[3], x=ls, y=quant_sin_modified[-1, :] - quant_sin_modified[0, :],          ls="--", color="blue", linewidth = 3)
    sns.lineplot(label=labels[4], x=ls, y=quant_sigmoid_shifted[-1, :] - quant_sigmoid_shifted[0, :],    ls="-",  color="green")            
    sns.lineplot(label=labels[5], x=ls, y=quant_poly1[-1, :] - quant_poly1[0, :],                        ls="-",  color="gray", linewidth=3)
    sns.lineplot(label=labels[6], x=ls, y=quant_poly2[-1, :] - quant_poly2[0, :],                        ls=":",  color="black", linewidth=2)

    plt.title("Difference between the 97.5% and 2.5% quantile of the preactivations norms for\nactivation functions in the $K^*_{00}=0$ universality class")
    ax.set_xlabel("layer")
    fig.savefig(out_folder_e05 + "/quantiles_diff_95.png")

    # [e06]
    cache_folder_e06 = base_folder + "/e06/cache"
    out_folder_e06 = "code/experiments/e06/out/aggregate/Kzero"
    os.makedirs(out_folder_e06, exist_ok=True)

    def R_file_name_mean(base, Cw, dev):
        return cache_folder_e06 + f"/R_mean_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_Mnorm=1_Mdev={dev}.pt"
    
    def R_file_name_quantiles(base, Cw, dev):
        return cache_folder_e06 + f"/R_quantiles_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_Mnorm=1_Mdev={dev}.pt"
    
    deviations = ["0.001", "0.01", "0.1", "0.5", "0.75"]

    R_tanh_list = []
    R_tanh_modified_list = []
    R_sin_list = []
    R_sin_modified_list = []
    R_signmoid_shifted_list = []
    R_poly1_list = []
    R_poly2_list = []
    R_tanh_list = []
    R_list_of_lists = [R_tanh_list, R_tanh_modified_list, R_sin_list, R_sin_modified_list,
                       R_signmoid_shifted_list, R_poly1_list, R_poly2_list]

    R_q_tanh_list = []
    R_q_tanh_modified_list = []
    R_q_sin_list = []
    R_q_sin_modified_list = []
    R_q_signmoid_shifted_list = []
    R_q_poly1_list = []
    R_q_poly2_list = []
    R_q_list_of_lists = [R_q_tanh_list, R_q_tanh_modified_list, R_q_sin_list, R_q_sin_modified_list,
                         R_q_signmoid_shifted_list, R_q_poly1_list, R_q_poly2_list]
                         
    for dev in deviations:
        for R_list, R_q_list, file_base, Cw in zip(R_list_of_lists, R_q_list_of_lists, file_bases, Cws):
            R_list.append(torch.load(R_file_name_mean(file_base, Cw, dev)))
            R_q_list.append(torch.load(R_file_name_quantiles(file_base, Cw, dev)))

    L = R_tanh_list[0].size()[0]
    ls = np.linspace(1, L, num=L)

    for i in range(len(deviations)):
        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        plt.title("$R^{(l)}$ for various activation functions in the $K^*_{00}=0$ universality class.\n$\\varepsilon_R = " + deviations[i] + "$")
        sns.lineplot(label=labels[0], x=ls, y=R_tanh_list[i],               ls="-",  color="red", linewidth=5)
        sns.lineplot(label=labels[1], x=ls, y=R_tanh_modified_list[i],      ls=":",  color="red", linewidth=6)
        # sns.lineplot(label=labels[2], x=ls, y=R_sin_list[i],                ls="-",  color="blue")
        sns.lineplot(label=labels[3], x=ls, y=R_sin_modified_list[i],       ls="--", color="blue", linewidth = 3)
        sns.lineplot(label=labels[4], x=ls, y=R_signmoid_shifted_list[i],   ls="-",  color="green")
        sns.lineplot(label=labels[5], x=ls, y=R_poly1_list[i],              ls="-",  color="gray", linewidth=3)
        sns.lineplot(label=labels[6], x=ls, y=R_poly2_list[i],              ls=":",  color="black", linewidth=2)
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e06 + f"/R_mean_dev={deviations[i]}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=R_q_tanh_list[i][0, :],               ls="-",  color="red", linewidth=5)
        sns.lineplot(label=labels[1], x=ls, y=R_q_tanh_modified_list[i][0, :],      ls=":",  color="red", linewidth=6)
        sns.lineplot(label=labels[2], x=ls, y=R_q_sin_list[i][0, :],                ls="-",  color="blue")
        sns.lineplot(label=labels[3], x=ls, y=R_q_sin_modified_list[i][0, :],       ls="--", color="blue", linewidth = 3)
        sns.lineplot(label=labels[4], x=ls, y=R_q_signmoid_shifted_list[i][0, :],   ls="-",  color="green")
        sns.lineplot(label=labels[5], x=ls, y=R_q_poly1_list[i][0, :],              ls="-",  color="gray", linewidth=3)
        sns.lineplot(label=labels[6], x=ls, y=R_q_poly2_list[i][0, :],              ls=":",  color="black", linewidth=2)

        plt.title("2.5% quantile for the metric underlying $R^{(l)}$,\nfor various activation functions in the $K^*_{00}=0$ universality class")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e06 + f"/R_q=2.5_dev={deviations[i]}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=R_q_tanh_list[i][-1, :],                  ls="-",  color="red", linewidth=5)
        sns.lineplot(label=labels[1], x=ls, y=R_q_tanh_modified_list[i][-1, :],         ls=":",  color="red", linewidth=6)
        sns.lineplot(label=labels[2], x=ls, y=R_q_sin_list[i][-1, :],                   ls="-",  color="blue")
        sns.lineplot(label=labels[3], x=ls, y=R_q_sin_modified_list[i][-1, :],          ls="--", color="blue", linewidth = 3)
        sns.lineplot(label=labels[4], x=ls, y=R_q_signmoid_shifted_list[i][-1, :],      ls="-",  color="green")
        sns.lineplot(label=labels[5], x=ls, y=R_q_poly1_list[i][-1, :],                 ls="-",  color="gray", linewidth=3)
        sns.lineplot(label=labels[6], x=ls, y=R_q_poly2_list[i][-1, :],                 ls=":",  color="black", linewidth=2)
        plt.title("97.5% quantile for the metric underlying $R^{(l)}$,\nfor various activation functions in the $K^*_{00}=0$ universality class")
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e06 + f"/R_q=97.5_dev={deviations[i]}.png")

    # [e07]
    cache_folder_e07 = base_folder + "/e07/cache"
    out_folder_e07 = "code/experiments/e07/out/aggregate/Kzero"
    os.makedirs(out_folder_e07, exist_ok=True)

    def D_file_name_mean(base, Cw, angle):
        if base == "tanh":
            return cache_folder_e07 + f"/D_mean_L=100_n=1000_Cw={Cw}_Nw=10000_seed=0_act={base}_norm=1_angle={angle}.pt"
        return cache_folder_e07 + f"/D_mean_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_norm=1_angle={angle}.pt"
    
    def D_file_name_quantiles(base, Cw, angle):
        if base == "tanh":
            return cache_folder_e07 + f"/D_quantiles_L=100_n=1000_Cw={Cw}_Nw=10000_seed=0_act={base}_norm=1_angle={angle}.pt"
        return cache_folder_e07 + f"/D_quantiles_L=100_n=1000_Cw={Cw}_Cb=None_Nw=10000_seed=0_act={base}_norm=1_angle={angle}.pt"

    D_tanh_list = []
    D_tanh_modified_list = []
    D_sin_list = []
    D_sin_modified_list = []
    D_sigmoid_shifted_list = []
    D_poly1_list = []
    D_poly2_list = []
    D_list_of_lists = [D_tanh_list, D_tanh_modified_list, D_sin_list, D_sin_modified_list,
                       D_sigmoid_shifted_list, D_poly1_list, D_poly2_list]

    D_q_tanh_list = []
    D_q_tanh_modified_list = []
    D_q_sin_list = []
    D_q_sin_modified_list = []
    D_q_sigmoid_shifted_list = []
    D_q_poly1_list = []
    D_q_poly2_list = []
    D_q_list_of_lists = [D_q_tanh_list, D_q_tanh_modified_list, D_q_sin_list, D_q_sin_modified_list,
                         D_q_sigmoid_shifted_list, D_q_poly1_list, D_q_poly2_list]

    angles = [
        "0.01227184630308513",
        "0.19634954084936207",
        "0.7853981633974483",
        "1.5707963267948966",
        "3.141592653589793",
    ]

    for angle in angles:
        for D_list, D_q_list, file_base, Cw in zip(D_list_of_lists, D_q_list_of_lists, file_bases, Cws):
            D_list.append(torch.load(D_file_name_mean(file_base, Cw, angle)))
            D_q_list.append(torch.load(D_file_name_quantiles(file_base, Cw, angle)))

    L = D_tanh_list[0].size()[0]
    ls = np.linspace(1, L, num=L)

    for i in range(len(angles)):
        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        plt.title("$D^{(l)}$ for various activation functions in the $K^*_{00}=0$ universality class\nangle = " + str(angles[i]))
        sns.lineplot(label=labels[0], x=ls, y=D_tanh_list[i],               ls="-",  color="red", linewidth=5)
        sns.lineplot(label=labels[1], x=ls, y=D_tanh_modified_list[i],      ls=":",  color="red", linewidth=6)
        sns.lineplot(label=labels[2], x=ls, y=D_sin_list[i],                ls="-",  color="blue")
        sns.lineplot(label=labels[3], x=ls, y=D_sin_modified_list[i],       ls="--", color="blue", linewidth = 3)
        sns.lineplot(label=labels[4], x=ls, y=D_sigmoid_shifted_list[i],   ls="-",  color="green")
        sns.lineplot(label=labels[5], x=ls, y=D_poly1_list[i],              ls="-",  color="gray", linewidth=3)
        sns.lineplot(label=labels[6], x=ls, y=D_poly2_list[i],              ls=":",  color="black", linewidth=2)
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e07 + f"/D_mean_angle={angles[i]}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=D_q_tanh_list[i][0, :],               ls="-",  color="red", linewidth=5)
        sns.lineplot(label=labels[1], x=ls, y=D_q_tanh_modified_list[i][0, :],      ls=":",  color="red", linewidth=6)
        sns.lineplot(label=labels[2], x=ls, y=D_q_sin_list[i][0, :],                ls="-",  color="blue")
        sns.lineplot(label=labels[3], x=ls, y=D_q_sin_modified_list[i][0, :],       ls="--", color="blue", linewidth = 3)
        sns.lineplot(label=labels[4], x=ls, y=D_q_sigmoid_shifted_list[i][0, :],    ls="-",  color="green")
        sns.lineplot(label=labels[5], x=ls, y=D_q_poly1_list[i][0, :],              ls="-",  color="gray", linewidth=3)
        sns.lineplot(label=labels[6], x=ls, y=D_q_poly2_list[i][0, :],              ls=":",  color="black", linewidth=2)

        plt.title("2.5% quantile for the metric underlying $D^{(l)}$,\nfor various activation functions in the $K^*_{00}=0$ universality class. angle = " + str(angles[i]))
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e07 + f"/D_q=2.5_angle={angles[i]}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 5.0)
        sns.lineplot(label=labels[0], x=ls, y=D_q_tanh_list[i][-1, :],                  ls="-",  color="red", linewidth=5)
        sns.lineplot(label=labels[1], x=ls, y=D_q_tanh_modified_list[i][-1, :],         ls=":",  color="red", linewidth=6)
        sns.lineplot(label=labels[2], x=ls, y=D_q_sin_list[i][-1, :],                   ls="-",  color="blue")
        sns.lineplot(label=labels[3], x=ls, y=D_q_sin_modified_list[i][-1, :],          ls="--", color="blue", linewidth = 3)
        sns.lineplot(label=labels[4], x=ls, y=D_q_sigmoid_shifted_list[i][-1, :],      ls="-",  color="green")
        sns.lineplot(label=labels[5], x=ls, y=D_q_poly1_list[i][-1, :],                 ls="-",  color="gray", linewidth=3)
        sns.lineplot(label=labels[6], x=ls, y=D_q_poly2_list[i][-1, :],                 ls=":",  color="black", linewidth=2)
        plt.title("97.5% quantile for the metric underlying $D^{(l)}$,\nfor various activation functions in the $K^*_{00}=0$ universality class. angle = " + str(angles[i]))
        ax.set_xlabel("layer")
        fig.savefig(out_folder_e07 + f"/D_q=97.5_angle={angles[i]}.png")

        plot_e07_angle_plots(
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
            colors,
            log_scale=False,
        )

        plot_e07_angle_plots(
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
            colors,
            log_scale=True,
        )