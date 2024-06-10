from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

def calculate_and_print_factor(a_min, a_plus):
    A2 = (a_plus ** 2 + a_min ** 2)/2
    A4 = (a_plus ** 4 + a_min ** 4)/2
    factor = 3 * A4 / (A2 ** 2) - 1
    print(f"if a_plus = {a_plus} and a_minus = {a_min} ...")
    print(f"    ... then the factor is equal to {factor}\n")

if __name__ == "__main__":
    print()
    calculate_and_print_factor(0, 1)
    calculate_and_print_factor(1, 1)
    calculate_and_print_factor(-1, 1)
    calculate_and_print_factor(-0.0001, 0.0001)
    calculate_and_print_factor(0.1, 1)

    bounds = 10
    step_size = 0.01
    num_steps = int(2 * bounds / step_size + 1)
    steps = np.linspace(-bounds, bounds, num_steps)
    a_plus = np.tile(steps, (num_steps, 1))
    a_min = a_plus.T

    A2 = (a_plus ** 2 + a_min ** 2)/2
    A4 = (a_plus ** 4 + a_min ** 4)/2
    middle_index = int(bounds / step_size)
    A2[middle_index, middle_index] = np.nan

    factor = 3 * A4 / (A2 ** 2) - 1

    fig = plt.figure()
    ax = plt.imshow(factor, extent=[-bounds, bounds, -bounds, bounds])
    fig.colorbar(ax)
    plt.xlabel("$a_+$")
    plt.ylabel("$a_-$")
    plt.title("O(1) factor in the magnitude of the four-point\nvertex for the scale-invariant activation class")
    fig.savefig("report/images/A4A2_heatmap.png")

    r = np.linspace(-10, 10, 1001)
    f = 6 * (1 + r ** 4) / ((1 + r ** 2)**2) - 1
    fig = plt.figure()
    sns.lineplot(x=r, y=f)
    plt.ylabel("$f\,(r)$")
    plt.xlabel("$r$")
    fig.savefig("report/images/A4A2_ratio_1.png")

    r = np.linspace(-2, 2, 1001)
    f = 6 * (1 + r ** 4) / ((1 + r ** 2)**2) - 1
    fig = plt.figure()
    sns.lineplot(x=r, y=f)
    plt.ylabel("$f\,(r)$")
    plt.xlabel("$r$")
    fig.savefig("report/images/A4A2_ratio_2.png")

