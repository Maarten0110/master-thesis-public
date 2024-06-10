from utils import is_colab, colab_drive_folder
from experiments.e06.generate import generate as e06_generate
from experiments.e06.analyze import analyze as e06_analyze
from experiments.e06.visualize import visualize as e06_visualize
from logger import log, logV2 as log_job
import os

prefix = colab_drive_folder if is_colab() else "code/experiments/"

# cache folder locations for "generate" step
cache_folder_generate_e06_colab_drive_mount = colab_drive_folder + "e06/cache"
cache_folder_generate_e06_colab_on_vm = "cache/e06"
cache_folder_generate_e06_local = "code/experiments/e06/cache"

# cache folder locations for "analyze" and "visualize" step
cache_folder_analyze_e06 = prefix + "e06/cache"
out_folder_visualize_e06 = prefix + "e06/out"

def run_job(
        params,
        get_cache_file_name_generate_e06,
        get_cache_file_name_analyze_e06,
        get_out_file_name_e06,
        job_id,
        batch_size_on_GPU = None,
        remove_raw_data_after_analaysis_step = False,
    ):

    log_job(job_id, "Starting...")
    e06_generate(params, get_cache_file_name_generate_e06, job_id, batch_size_on_GPU = batch_size_on_GPU)
    e06_analyze(params, get_cache_file_name_generate_e06, get_cache_file_name_analyze_e06, job_id)
    if remove_raw_data_after_analaysis_step:
        remove_raw_data(params, get_cache_file_name_generate_e06)
    e06_visualize(params, get_cache_file_name_analyze_e06, get_out_file_name_e06, job_id)
    log_job(job_id, "Done!")

def remove_raw_data(params, get_cache_file_name_generate_e06):
    os.remove(get_cache_file_name_generate_e06(params, name="zs_aggr"))

def paramaters(L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton):
    return L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton

def cache_file_name_generate_e06_colab_drive_mount(params, name):
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    return f"{cache_folder_generate_e06_colab_drive_mount}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_Mnorm={midpoint_norm}_Mdev={midpoint_deviaton}.pt"

def cache_file_name_generate_e06_colab_on_vm(params, name):
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    return f"{cache_folder_generate_e06_colab_on_vm}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_Mnorm={midpoint_norm}_Mdev={midpoint_deviaton}.pt"

def cache_file_name_generate_e06_local(params, name):
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    return f"{cache_folder_generate_e06_local}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_Mnorm={midpoint_norm}_Mdev={midpoint_deviaton}.pt"

def cache_file_name_analyze_e06(params, name):
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    return f"{cache_folder_analyze_e06}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_Mnorm={midpoint_norm}_Mdev={midpoint_deviaton}.pt"

def out_file_name_e06(params, name):
    L, n, Cw, Cb, Nw, seed, activation, midpoint_norm, midpoint_deviaton = params
    return f"{out_folder_visualize_e06}/{name}_act={activation['file_base']}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_Mnorm={midpoint_norm}_Mdev={midpoint_deviaton}.png"

def run_experiment(jobs, batch_size_on_GPU = None, save_raw_data_to_drive = False, remove_raw_data_after_analaysis_step = False):
    cache_file_name_generate_e06 = None
    
    if is_colab() and save_raw_data_to_drive:
        os.makedirs(cache_folder_generate_e06_colab_drive_mount, exist_ok=True)
        cache_file_name_generate_e06 = cache_file_name_generate_e06_colab_drive_mount
        log("WARNING: Saving the raw data from the \"generate\" step to Google Drive! This can be many gigabytes!")
    
    if is_colab() and not save_raw_data_to_drive:
        os.makedirs(cache_folder_generate_e06_colab_on_vm, exist_ok=True)
        cache_file_name_generate_e06 = cache_file_name_generate_e06_colab_on_vm
        log("INFO: Saving the raw data from the \"generate\" step to the VM (it is not saved to Google Drive.)")
    
    if not is_colab():
        os.makedirs(cache_folder_generate_e06_local, exist_ok=True)
        cache_file_name_generate_e06 = cache_file_name_generate_e06_local
        log("INFO: Experiment assumed to be running locally (not on Colab).")

    os.makedirs(cache_folder_analyze_e06, exist_ok=True)
    os.makedirs(out_folder_visualize_e06, exist_ok=True)

    for i, job in enumerate(jobs):
        run_job(
            job["params"],
            cache_file_name_generate_e06,
            cache_file_name_analyze_e06,
            out_file_name_e06,
            i,
            batch_size_on_GPU = batch_size_on_GPU,
            remove_raw_data_after_analaysis_step = remove_raw_data_after_analaysis_step,
        )
