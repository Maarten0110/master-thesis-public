from utils import is_colab, colab_drive_folder
from experiments.e07.generate import generate as e07_generate
from experiments.e07.analyze import analyze as e07_analyze
from experiments.e07.visualize import visualize as e07_visualize
from logger import log, logV2 as log_job
import os

prefix = colab_drive_folder if is_colab() else "code/experiments/"

# cache folder locations for "generate" step
cache_folder_generate_e07_colab_drive_mount = colab_drive_folder + "e07/cache"
cache_folder_generate_e07_colab_on_vm = "cache/e07"
cache_folder_generate_e07_local = "code/experiments/e07/cache"

# cache folder locations for "analyze" and "visualize" step
cache_folder_analyze_e07 = prefix + "e07/cache"
out_folder_visualize_e07 = prefix + "e07/out"

def run_job(
        params,
        get_cache_file_name_generate_e07,
        get_cache_file_name_analyze_e07,
        get_out_file_name_e07,
        job_id,
        batch_size_on_GPU = None,
        remove_raw_data_after_analaysis_step = False,
    ):

    log_job(job_id, "Starting...")
    e07_generate(params, get_cache_file_name_generate_e07, job_id, batch_size_on_GPU = batch_size_on_GPU)
    e07_analyze(params, get_cache_file_name_generate_e07, get_cache_file_name_analyze_e07, job_id)
    if remove_raw_data_after_analaysis_step:
        remove_raw_data(params, get_cache_file_name_generate_e07)
    e07_visualize(params, get_cache_file_name_analyze_e07, get_out_file_name_e07, job_id)
    log_job(job_id, "Done!")

def remove_raw_data(params, get_cache_file_name_generate_e07):
    os.remove(get_cache_file_name_generate_e07(params, name="zs_aggr"))

def paramaters(L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle):
    return L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle

def cache_file_name_generate_e07_colab_drive_mount(params, name):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    return f"{cache_folder_generate_e07_colab_drive_mount}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_norm={inputs_norm}_angle={inputs_angle}.pt"

def cache_file_name_generate_e07_colab_on_vm(params, name):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    return f"{cache_folder_generate_e07_colab_on_vm}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_norm={inputs_norm}_angle={inputs_angle}.pt"

def cache_file_name_generate_e07_local(params, name):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    return f"{cache_folder_generate_e07_local}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_norm={inputs_norm}_angle={inputs_angle}.pt"

def cache_file_name_analyze_e07(params, name):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    return f"{cache_folder_analyze_e07}/{name}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_act={activation['file_base']}_norm={inputs_norm}_angle={inputs_angle}.pt"

def out_file_name_e07(params, name):
    L, n, Cw, Cb, Nw, seed, activation, inputs_norm, inputs_angle = params
    return f"{out_folder_visualize_e07}/{name}_act={activation['file_base']}_L={L}_n={n}_Cw={Cw}_Cb={Cb}_Nw={Nw}_seed={seed}_norm={inputs_norm}_angle={inputs_angle}.png"

def run_experiment(jobs, batch_size_on_GPU = 5, save_raw_data_to_drive = False, remove_raw_data_after_analaysis_step = False):
    cache_file_name_generate_e07 = None

    if is_colab() and save_raw_data_to_drive:
        os.makedirs(cache_folder_generate_e07_colab_drive_mount, exist_ok=True)
        cache_file_name_generate_e07 = cache_file_name_generate_e07_colab_drive_mount
        log("WARNING: Saving the raw data from the \"generate\" step to Google Drive! This can be many gigabytes!")
    
    if is_colab() and not save_raw_data_to_drive:
        os.makedirs(cache_folder_generate_e07_colab_on_vm, exist_ok=True)
        cache_file_name_generate_e07 = cache_file_name_generate_e07_colab_on_vm
        log("INFO: Saving the raw data from the \"generate\" step to the VM (it is not saved to Google Drive.)")
    
    if not is_colab():
        os.makedirs(cache_folder_generate_e07_local, exist_ok=True)
        cache_file_name_generate_e07 = cache_file_name_generate_e07_local
        log("INFO: Experiment assumed to be running locally (not on Colab).")

    os.makedirs(cache_folder_analyze_e07, exist_ok=True)
    os.makedirs(out_folder_visualize_e07, exist_ok=True)

    for i, job in enumerate(jobs):
        run_job(
            job["params"],
            cache_file_name_generate_e07,
            cache_file_name_analyze_e07,
            out_file_name_e07,
            i,
            batch_size_on_GPU = batch_size_on_GPU,
            remove_raw_data_after_analaysis_step = remove_raw_data_after_analaysis_step,
        )

