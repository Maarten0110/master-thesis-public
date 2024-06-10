from scipy import integrate
from math import pi, sqrt, exp, factorial as fac
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

def integrand_tanh(z, K):
    return 1 / sqrt(2*pi*K) * exp(-(z**2) / 2 / K) * (np.tanh(z)) ** 2

def integrand_tanh_b(z, K):
    return 1 / sqrt(2*pi*K) * exp(-(z**2) / 2 / K) * (np.tanh(z * 0.05)) ** 2

def integrand_sin(z, K):
    return 1 / sqrt(2*pi*K) * exp(-(z**2) / 2 / K) * (np.sin(z)) ** 2

def integrand_sin_b(z, K):
    return 1 / sqrt(2*pi*K) * exp(-(z**2) / 2 / K) * (np.sin(z * 0.05)) ** 2

def integrand_poly1(z, K):
    eps = 0.01
    s = z + 0.5*(z**2) - 3*(z**3)/(fac(3)*4) - (1/fac(4)) * (3/8 + 8/5 * eps) * (z ** 4)
    return 1 / sqrt(2*pi*K) * exp(-(z**2) / 2 / K) * (s ** 2)

def integrand_poly2(z, K):
    eps_a = eps_b = 0.01
    s = z + 0.5*sqrt(4*eps_a)*(z**2) - 4*eps_a*(z**3)/(fac(3)) - (eps_b + 12 * eps_a**2)/(fac(4) * sqrt(4*eps_a)) * (z ** 4)
    return 1 / sqrt(2*pi*K) * exp(-(z**2) / 2 / K) * (s ** 2)

def r(K, Cb, Cw, integrand):
    if K == 0:
        return 1, 0
    
    I = integrate.quad(lambda z: integrand(z, K), -np.inf, np.inf)
    value = (Cb + Cw * I[0]) / K
    error_estimate = Cw * I[1] / K

    return value, error_estimate
    
def Cw_K_star_zero(sigma_1, layer_width):
    Cw0 = 1 / (sigma_1 ** 2)
    return Cw0 + 1 / layer_width * (2/3) * Cw0

if __name__ == "__main__":
    
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

    layer_width = 1000
    Ks = np.linspace(0, 15, num=250)
    activations = [
        {"integrand": integrand_tanh,   "Cw": Cw_K_star_zero(1,     layer_width), "Cb": 0, "ls": "-",  "color": colors[0], "label": "tanh"},
        {"integrand": integrand_tanh_b, "Cw": Cw_K_star_zero(0.05,  layer_width), "Cb": 0, "ls": "-",  "color": colors[1], "label": "modified tanh, $\\beta=0.05$"},
        {"integrand": integrand_sin,    "Cw": Cw_K_star_zero(1,     layer_width), "Cb": 0, "ls": "-",  "color": colors[2], "label": "sin"},
        {"integrand": integrand_sin_b,  "Cw": Cw_K_star_zero(0.05,  layer_width), "Cb": 0, "ls": "-",  "color": colors[3], "label": "modified sin, $\\beta=0.05$"},
        {"integrand": integrand_poly1,  "Cw": Cw_K_star_zero(1,     layer_width), "Cb": 0, "ls": "-",  "color": colors[5], "label": "custom polynomial 1"},
        {"integrand": integrand_poly2,  "Cw": Cw_K_star_zero(1,     layer_width), "Cb": 0, "ls": "-",  "color": colors[6], "label": "custom polynomial 2"},
    ]
    results_value = np.zeros((len(activations), len(Ks)))
    results_error = np.zeros((len(activations), len(Ks)))
    for i, act in enumerate(activations):
        for j, K in enumerate(Ks):
            result = r(K, act["Cb"], act["Cw"], act["integrand"])
            results_value[i, j] = result[0]
            results_error[i, j] = result[1]

    fig, ax = plt.subplots()
    # plt.fill_between(Ks, results_value[5, :] - results_error[5, :], results_value[5, :] + results_error[5, :], color=(1, 0, 0, 0.1))
    for i, activation in enumerate(activations):
        sns.lineplot(x=Ks, y=results_value[i, :], label=activation["label"], ls=activation["ls"], color=activation["color"])
    # ax.set_yscale("log")
    ax.set_ylim([0, 2])
    ax.set_xlabel("k")
    ax.set_ylabel("r(k)")
    plt.title("The $r(k)$ metric for the $K^*_{00}=0$ activation functions.")
    fig.savefig("code/experiments/images/r.png", dpi=300)
