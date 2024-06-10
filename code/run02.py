import os
from experiments.e01.generate import generate as e01_generate
from experiments.e01.analyze import analyze as e01_analyze
from experiments.e01.visualize import visualize as e01_visualize
from experiments.e02.analyze import analyze as e02_analyze
from experiments.e02.visualize import visualize as e02_visualize
from torch.multiprocessing import Pool
from logger import log

num_workes = 8
cache_folder_e01 = "code/experiments/e01/cache"
cache_folder_e02 = "code/experiments/e02/cache"
out_folder_e01 = "code/experiments/e01/out"
out_folder_e02 = "code/experiments/e02/out"

if not os.path.isdir(cache_folder_e01):
    os.makedirs(cache_folder_e01)
if not os.path.isdir(out_folder_e02):
    os.makedirs(out_folder_e02)

def run_job(params, cache_file_name_e01, cache_file_name_e02, out_file_name_e01, out_file_name_e02, job_id, use_log_scale_e01, use_log_scale_e02):
    log(f"[job {job_id}] Starting...")
    e01_generate(params, cache_file_name_e01, job_id)
    e01_analyze(params, cache_file_name_e01, job_id)
    e01_visualize(params, cache_file_name_e01, out_file_name_e01, job_id, use_log_scale_e01)
    e02_analyze(params, cache_file_name_e01, cache_file_name_e02, job_id)
    e02_visualize(params, cache_file_name_e02, out_file_name_e02, job_id, use_log_scale_e02)
    log(f"[job {job_id}] Done!")

def cache_file_name_e01(L, n, Cw, Nw, seed, name):
    return f"{cache_folder_e01}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.pt"

def cache_file_name_e02(L, n, Cw, Nw, seed, name):
    return f"{cache_folder_e02}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.pt"

def out_file_name_01(L, n, Cw, Nw, seed, name):
    return f"{out_folder_e01}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.png"

def out_file_name_02(L, n, Cw, Nw, seed, name):
    return f"{out_folder_e02}/{name}_L={L}_n={n}_Cw={Cw}_Nw={Nw}_seed={seed}.png"

def paramaters(L, n, Cw, Nw, seed):
    return L, n, Cw, Nw, seed

if __name__ == "__main__":
    if not os.path.isdir(cache_folder_e01):
        os.makedirs(cache_folder_e01)
    if not os.path.isdir(cache_folder_e02):
        os.makedirs(cache_folder_e02)
    if not os.path.isdir(out_folder_e01):
        os.makedirs(out_folder_e01)
    if not os.path.isdir(out_folder_e02):
        os.makedirs(out_folder_e02)

    jobs = [
        # starting point
        {"params": paramaters(L= 100, n=1000, Cw=1, Nw=1000, seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        
        # vary n
        {"params": paramaters(L= 100, n=2000, Cw=1, Nw=500,     seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=500,  Cw=1, Nw=2000,    seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=100,  Cw=1, Nw=10000,   seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=50,   Cw=1, Nw=20000,   seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=10,   Cw=1, Nw=100000,  seed=0), "use_log_scale_e01": False, "use_log_scale_e02": True},
        {"params": paramaters(L= 100, n=5,    Cw=1, Nw=200000,  seed=0), "use_log_scale_e01": False, "use_log_scale_e02": True},
        {"params": paramaters(L= 100, n=2,    Cw=1, Nw=500000,  seed=0), "use_log_scale_e01": False, "use_log_scale_e02": True},
        
        # vary Cw
        {"params": paramaters(L= 100, n=1000, Cw=2,    Nw=1000,    seed=0), "use_log_scale_e01": True, "use_log_scale_e02": True},
        {"params": paramaters(L= 100, n=1000, Cw=1.5,  Nw=1000,    seed=0), "use_log_scale_e01": True, "use_log_scale_e02": True},
        {"params": paramaters(L= 100, n=1000, Cw=1.25, Nw=1000,    seed=0), "use_log_scale_e01": True, "use_log_scale_e02": True},
        {"params": paramaters(L= 100, n=1000, Cw=1.05, Nw=1000,    seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=1000, Cw=1.01, Nw=1000,    seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.99, Nw=1000,    seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.95, Nw=1000,    seed=0), "use_log_scale_e01": False, "use_log_scale_e02": False},
        {"params": paramaters(L= 100, n=1000, Cw=0.75, Nw=1000,    seed=0), "use_log_scale_e01": True, "use_log_scale_e02": True},
        {"params": paramaters(L= 100, n=1000, Cw=0.5,  Nw=1000,    seed=0), "use_log_scale_e01": True, "use_log_scale_e02": True},
    ]

    with Pool(num_workes) as p:
        p.starmap(run_job, [
            (job["params"], cache_file_name_e01, cache_file_name_e02, out_file_name_01, out_file_name_02, i, job["use_log_scale_e01"], job["use_log_scale_e02"])
            for i, job in enumerate(jobs)]
        )

    # for i, params in enumerate(jobs):
    #     run_job(job["params"], cache_file_name_e01, cache_file_name_e02, out_file_name_01, out_file_name_02, i, job["use_log_scale_e01"], job["use_log_scale_e02"])
