from experiments.e01.generate import generate as e01_generate
from experiments.e04.analyze import analyze as e04_analyze
from experiments.e04.visualize import visualize as e04_visualize
from logger import log
import os
from torch.multiprocessing import Pool

num_workes = 4
cache_folder_e01 = "code/experiments/e01/cache"
cache_folder_e04 = "code/experiments/e04/cache"
out_folder_e04 = "code/experiments/e04/out"

if not os.path.isdir(cache_folder_e01):
    os.makedirs(cache_folder_e01)
if not os.path.isdir(cache_folder_e04):
    os.makedirs(cache_folder_e04)
if not os.path.isdir(out_folder_e04):
    os.makedirs(out_folder_e04)

def run_job(params, cache_file_name_e01, cache_file_name_e04, out_file_name_e04, job_id):
    log(f"[job {job_id}] Starting...")
    e01_generate(params, cache_file_name_e01, job_id=job_id)
    e04_analyze(params, cache_file_name_e01, cache_file_name_e04, job_id=job_id)
    e04_visualize(params, cache_file_name_e04, out_file_name_e04, job_id=job_id)

def paramaters(L, n, Cw, Nw, seed):
    return L, n, Cw, Nw, seed

def cache_file_name_e01(L, n, Cw, Nw, seed, name):
    return f"{cache_folder_e01}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.pt"

def cache_file_name_e04(L, n, Cw, Nw, seed, name):
    return f"{cache_folder_e04}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.pt"

def out_file_name_04(L, n, Cw, Nw, seed, name):
    return f"{out_folder_e04}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.png"

jobs = [
    # starting point
    {"params": paramaters(L= 30, n=3000, Cw=1, Nw=10000, seed=0)},
    
    # vary n
    {"params": paramaters(L= 30, n=2500, Cw=1, Nw=10000,  seed=0)},
    {"params": paramaters(L= 30, n=2000, Cw=1, Nw=10000,  seed=0)},
    {"params": paramaters(L= 30, n=1500, Cw=1, Nw=10000,  seed=0)},
    {"params": paramaters(L= 30, n=1000, Cw=1, Nw=10000,  seed=0)},
    {"params": paramaters(L= 30, n=750,  Cw=1, Nw=10000,  seed=0)},
    {"params": paramaters(L= 30, n=250,  Cw=1, Nw=10000,  seed=0)},
    {"params": paramaters(L= 30, n=500,  Cw=1, Nw=10000,  seed=0)},
    
    # vary Cw
    {"params": paramaters(L= 30, n=3000, Cw=1.05, Nw=10000,    seed=0)},
    {"params": paramaters(L= 30, n=3000, Cw=1.01, Nw=10000,    seed=0)},
    {"params": paramaters(L= 30, n=3000, Cw=0.99, Nw=10000,    seed=0)},
    {"params": paramaters(L= 30, n=3000, Cw=0.95, Nw=10000,    seed=0)},
]


if __name__ == "__main__":
    with Pool(num_workes) as p:
        p.starmap(run_job, [
            (job["params"], cache_file_name_e01, cache_file_name_e04, out_file_name_04, i)
            for i, job in enumerate(jobs)]
    )

    # for job in jobs:
    #     run_job(job["params"], cache_file_name_e01, cache_file_name_e04, out_file_name_04, 0)