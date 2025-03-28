import os.path
import gc
import multiprocessing as mp
import random
from copy import deepcopy
import numpy as np
import config.config
from core import GPSolver
from config.config import get_config, n_cores, fundamental_units
from Utils import Utils_io, Utils_data, Utils_ground_truth, Utils_misc
import pickle
from config import config
from Targets.Feynman_with_units import cleaning_feynman_dataset


def reset_environment(dataset_name):
    """
    Deletes the preprocessed data and ground truth files required for the GP Solver.
    This is safer to use as changing eg. use_norms will not be taken into account if the files are not deleted.
    """
    Utils_io.delete_preprocessed_data(dataset_name)
    Utils_io.delete_dictionnaries(dataset_name)

def initialize_environment(dataset_name, cfg):
    """
    Sets up directories and ground truth files required for the GP Solver.
    """
    reset_environment(dataset_name)
    Utils_io.create_directory(dataset_name)
    Utils_io.check_metadata_downloads(dataset_name)

    # cleaning the dataset if not already done
    if dataset_name == 'Feynman_with_units':
        cleaning_feynman_dataset.delete_cleaned_files()
        cleaning_feynman_dataset.clean_variable_names_feynman_dataset()
        cleaning_feynman_dataset.clean_equation_list_feynman_dataset()
        cleaning_feynman_dataset.clean_equation_list_bonus_equations()

    # Build the unit dictionary from the source files
    if not os.path.exists(f'Targets/{dataset_name}/unit_dict.pkl'):
        Utils_ground_truth.Build_units_dict(dataset_name)

    # Build the formula dictionary from the source files
    if not os.path.exists(f'Targets/{dataset_name}/formula_dict_with_units'):
        Utils_ground_truth.Build_Feynmann_Formula_Dict(dataset_name, fundamental_units, cfg)

def main(task: tuple) -> tuple:
    """
    Main function to execute the GP solver for a given task.
    """
    try:
        dataset_name, cfg, equation_label, multi_processing, n_cores, run_number, recompute = task
        # reproducibility
        # WARNING : on using parallel tree generation, the results are not reproducible, because the seed is different on each subprocess
        # to turn off parallel tree generation, set use_parallel_tree_generation = False in config/config.py
        if config.use_seed:
            random.seed(run_number)
            np.random.seed(run_number)

        ground_truth = Utils_io.load_ground_truth(dataset_name, equation_label)
        Utils_data.preprocess_data(equation_label, dataset_name, ground_truth, cfg['use_noise'], cfg['noise_level'],
                                   cfg['use_denoising'], cfg['k_neighbors'], recompute = recompute)

        # two steps solving
        solver = GPSolver(
            dataset_name=dataset_name,
            equation_label=equation_label,
            run_number=run_number,
            multi_processing=multi_processing,
            cfg=cfg,
            ground_truth=ground_truth
        )
        final_answer, is_trivial, solution_found, exact_recovery, iteration, elapsed_time, all_eq_seen, QD_object = solver.solve()
        del solver
        gc.collect()  # Force garbage collection to free up memory

        if solution_found:
            return final_answer, is_trivial, solution_found, iteration, elapsed_time, equation_label, len(all_eq_seen), exact_recovery

        elif config.two_staged_run:
            print('Trying again on a larger grid and with a greater maximal length')
            # default run : first hour on size 30 with medium grid, then 1 hour on size 40 with large grid
            prev_iteration = iteration
            prev_elapsed_time = elapsed_time
            cfg2 = deepcopy(cfg)
            upgrade_rule = {'small': 'medium', 'medium': 'large', 'large': 'very_large'}
            cfg2['maximal_length'] += 10 # 35 -> 45
            cfg2['timeout'] = 3600*1 # 1 hour -> 5 hours #will be stopped anyway if 1e6 eq seen hit
            cfg2['qd_grid_size'] = upgrade_rule[cfg['qd_grid_size']]
            previous_qd_grid = QD_object.qd_grid
            solver = GPSolver(
                dataset_name=dataset_name,
                equation_label=equation_label,
                run_number=run_number,
                multi_processing=multi_processing,
                cfg=cfg2,
                ground_truth=ground_truth
            )
            final_answer, is_trivial, solution_found, exact_recovery, iteration, elapsed_time, all_eq_seen, QD_object = solver.solve(previous_qd_grid, all_eq_seen)
            del solver
            gc.collect()  # Force garbage collection to free up memory

            return final_answer, is_trivial, solution_found, iteration + prev_iteration, elapsed_time + prev_elapsed_time, equation_label, len(all_eq_seen), exact_recovery
        else:
            return final_answer, is_trivial, solution_found, iteration, elapsed_time, equation_label, len(all_eq_seen), False
    except Exception as e:
        print(f"Error solving {equation_label}: {e}")
        raise e

def execute_tasks(dataset_name, cfg, eqs, run_all_parallel, run_number, path, recompute):
    """
    Execute tasks either in parallel or sequentially.
    """
    if run_all_parallel:
        tasks = [[dataset_name, cfg, eq, False, 1, run_number] for eq in eqs]
        with mp.Pool(n_cores) as mp_pool:
            result = mp_pool.map(main, tasks)
            log_results(*result, path)

    else:
        for equation_label in eqs:
            task = [dataset_name, cfg, equation_label, True, n_cores, run_number, recompute]
            result = main(task)
            log_results(*result, path)

def log_results(final_answer, is_trivial, solution_found, iteration, elapsed_time, equation_label, eqseen, recovery, path):
    """
    Log the results of the GP solver.
    """
    if not os.path.exists(path):
        all_results = {}
    else:
        with open(path, 'rb') as f:
            all_results = pickle.load(f)

    if equation_label not in all_results:
        all_results[equation_label] = {'found': [solution_found], 'time': [elapsed_time], 'iter': [iteration],
                                       'final_answer': [final_answer],
                                       'eqseen': [eqseen], 'recovery': [recovery]}
    else:
        all_results[equation_label]['found'].append(solution_found)
        all_results[equation_label]['time'].append(elapsed_time)
        all_results[equation_label]['iter'].append(iteration)
        all_results[equation_label]['final_answer'].append(final_answer)
        all_results[equation_label]['eqseen'].append(eqseen)
        all_results[equation_label]['recovery'].append(recovery)

    with open(path, 'wb') as f:
        pickle.dump(all_results, f)

def load_equation_list():
    dimensionally_trivial, very_easy, easy_table, medium_table, hard_table = Utils_misc.load_equation_list()
    return very_easy + easy_table + medium_table + hard_table

if __name__ == '__main__':
    recompute = True # Set to True to recompute the data from scratch from actual formula
    dataset_name = 'Feynman_with_units'  # can be also 'Feynman_without_units'
    search_intensity = 'custom'  # Set search intensity as per requirement
    cfg = get_config(search_intensity)
    initialize_environment(dataset_name, cfg)
    eqs = load_equation_list()
    output_path = 'results_custom_run.pkl'

    print('running on', len(eqs), 'equations')
    for equation_label in eqs:
        Utils_io.check_data_downloads(dataset_name, equation_label)

    for run_number in range(10):
        run_targets_in_parallel = 0
        if len(eqs) == 1:
            run_targets_in_parallel = False

        # Execute tasks
        execute_tasks(dataset_name, cfg, eqs, run_targets_in_parallel, run_number, output_path, recompute)
        print("End of processing.")
