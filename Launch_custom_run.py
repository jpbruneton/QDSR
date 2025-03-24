from config import config
from Utils import Utils_io, Utils_data, Utils_ground_truth, Utils_misc
import numpy as np
from core import GPSolver
import gc
from config.config import get_config
import pandas as pd

# Example file where you want to run the solver on other targets than Feynmann AI.

###################  CASE I : give yourself a formula and run the solver on it

############   see Launch_custom_run2.py for case II where data is provided, and formula may not be known

## Say you want to discover f(x) = 3.01*x*y*sin(0.52*x/y), where x and y have dimension of length

## Step 1. Give a name to the dataset, say 'my_dataset',
# also provide n_var, var_names, var dimensions
# and the formula, ranges to generate the data, and number of points
# **** Functions must be written in usual format (eg. exp(), not np.exp()) ****

dataset_name = 'my_dataset'
true_name_variables = ['x', 'y', 'z']
n_var = len(true_name_variables)
unit_dict = {'x': {'m': 0, 's': 0, 'kg': 0, 'T': 0, 'V': 0}, 'y': {'m': 0, 's': 0, 'kg': 0, 'T': 0, 'V': 0}, 'z': {'m': 0, 's': 0, 'kg': 0, 'T': 0, 'V': 0}}
formula = '3.01*sqrt(x**2 + y**2) -1.21*x*x/y'
formula = '0.51*x*y*sin(0.52*x/y)'
formula = '1 + x + x**2 + x**3 + x**4 + x**5'
formula = 'x*y*z'

equation_label = 'my_formula_1'
var_range = {'x': (0, 2), 'y': (0,2), 'z' : (1,5) } #positive here to ensure log is well-defined
n_points = 50000

must_be_true = ('A' not in formula) and all(['A' not in x for x in true_name_variables])
assert must_be_true, '"A" is a reserved keyword, please use another name for variable names in formula'
assert len(unit_dict) == n_var, 'unit_dict must have the same number of variables as n_var'
assert len(true_name_variables) == n_var, 'true_name_variables must have the same number of variables as n_var'

# then run the solver on it
def solve_my_formula(dataset_name, n_var, true_name_variables, unit_dict, formula, equation_label, var_range, n_points):

    ## Step 2. Create a folder in the Targets directory with the name of the dataset
    Utils_io.create_directory(dataset_name)

    ## Step 3: create the ground truth dictionary ; NB target dimension (here L^2) will be computed automatically
    search_intensity = 'custom'
    cfg = get_config(search_intensity)
    ground_truth = Utils_ground_truth.create_ground_truth_dict_for_equation(
                            equation_label, formula, None, n_var, true_name_variables,
                            unit_dict, config.fundamental_units, cfg, verbose = False)


    # FYI : ground_truth is a dictionary that will be printed at the beginning of the run
    for k, v in ground_truth.items():
        print(f'{k}: {v}')
    ## Step 4 : save it in the Targets directory
    Utils_io.save_ground_truth(ground_truth, dataset_name)

    # Step 5 : generate the data using the Utils_data.generate_data() function
    generated_data = Utils_data.generate_data(ground_truth, var_range, n_points, noise_level=0)
    # print(generated_data.shape) ----> (10000, 3) ; 2 variables + 1 target

    # Step 6 : check the data has non NaN values or other weird stuff
    is_valid = not (np.isnan(generated_data).any() or np.isinf(generated_data).any() or np.iscomplex(generated_data).any())
    assert is_valid, 'Generated data contains NaN, inf or complex values, check your ranges and your formula'

    # save it as a csv file in the Targets directory to mimic the Feynman AI dataset
    pd.DataFrame(generated_data).to_csv(f'Targets/{dataset_name}/raw_data/{equation_label}', index=False, sep=' ', header=False)

    # Step 8 : preprocess the data to precompute eg. adimensional variables x/y and y/x and add them to the data
    # at this stage you can also add noise to the target


    Utils_data.preprocess_data(equation_label, dataset_name, ground_truth, cfg['use_noise'], cfg['noise_level'],
                               cfg['use_denoising'],
                               cfg['k_neighbors'],
                               recompute=True)
    # Step 9. Run the solver
    multi_processing = True
    run_number = 0

    solver = GPSolver(dataset_name,
        equation_label,
        run_number,
        multi_processing,
        cfg,
        ground_truth
    )
    final_answer, is_trivial, solution_found, iteration, elapsed_time, eq_seen, QD_object = solver.solve()
    del solver
    gc.collect()
    print(f'Final answer: {final_answer}, is trivial: {is_trivial}, solution found: {solution_found}, iteration: {iteration}, elapsed time: {elapsed_time}')

if __name__ == '__main__':
    solve_my_formula(dataset_name, n_var, true_name_variables, unit_dict, formula, equation_label, var_range, n_points)
