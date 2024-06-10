import os
from experiments.e01.generate import *
from experiments.e01.analyze import *
from experiments.e01.visualize import *
from torch.multiprocessing import Pool
from logger import log

num_workes = 8
cache_folder = "code/experiments/e01/cache"
out_folder = "code/experiments/e01/out"

def cache_file_name(L, n, Cw, Nw, seed, name):
    return f"{cache_folder}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.pt"

def out_file_name(L, n, Cw, Nw, seed, name):
    return f"{out_folder}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.png"

def run_job(params, cache_file_name, out_file_name, job_id, use_logarithmic_scale):
    log(f"[job {job_id}] Starting...")
    generate(params, cache_file_name, job_id)
    analyze(params, cache_file_name, job_id)
    visualize(params, cache_file_name, out_file_name, job_id, use_logarithmic_scale)
    log(f"[job {job_id}] Done!")

if __name__ == "__main__":

    if not os.path.isdir(cache_folder):
        os.makedirs(cache_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    cache_folder = "code/experiments/e01/cache"
    out_folder = "code/experiments/e01/out"

    def paramaters(L, n, Cw, Nw, seed):
        return L, n, Cw, Nw, seed

    jobs = [
        # starting point
        {"params": paramaters(L= 100, n=1000, Cw=1, Nw=1000, seed=0), "use_logarithmic_scale": False},
        
        # vary n
        {"params": paramaters(L= 100, n=2000, Cw=1, Nw=500,     seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=500,  Cw=1, Nw=2000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=100,  Cw=1, Nw=10000,   seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=50,   Cw=1, Nw=20000,   seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=10,   Cw=1, Nw=100000,  seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=5,    Cw=1, Nw=200000,  seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=2,    Cw=1, Nw=500000, seed=0), "use_logarithmic_scale": False},
        
        # vary Cw
        {"params": paramaters(L= 100, n=1000, Cw=2,    Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=1.5,  Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=1.25, Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=1.05, Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=1.01, Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.99, Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.95, Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.75, Nw=1000,    seed=0), "use_logarithmic_scale": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.5,  Nw=1000,    seed=0), "use_logarithmic_scale": False},
    ]

    with Pool(num_workes) as p:
        p.starmap(run_job, [(job["params"], cache_file_name, out_file_name, i, job["use_logarithmic_scale"]) for i, job in enumerate(jobs)])
    # for i, params in enumerate(jobs):
    #     run_job(params, cache_file_name, out_file_name, i)
    # run_job(jobs[7], cache_file_name, out_file_name, 7)
