import psutil

# _________ common values to any run ____________

# Meta mode for data transformation ('linlin', 'loglin', 'loglog') ; meaning : look for y = f(x) or log(y) = f(x) or log(y) = log(f(x))
meta_mode = 'linlin'  # other possibility is 'loglin' or 'loglog' #other setup that linlin is not really a good idea

fundamental_units = ['m', 's', 'kg', 'T', 'V']  # Length, Time, Mass, Temperature, Voltage  "as in FeynamnAI dataset
num_fundamental_units = len(fundamental_units)

MAX_NESTED_FUNCTIONS = 2  # Maximum number of nested functions to use in the equations sin(cos(x) is ok but sin(exp(sqrt(x))) is not
MAX_NESTED_POWERS = 2  # Maximum number of scalars to use in the equations

metric = 'R2'  # other choice is 'R2'
max_bad_loss = 1000 if metric == 'NRMSE' else -1e3  # loss threshold to avoid updating the pool with high-loss individuals
failure_loss = 1e6 if metric == 'NRMSE' else -1e6  # default failure loss when evaluation fails (nans, inf, complex values, zero division, etc)
termination_loss = 1e-14 if metric == 'NRMSE' else 1 - 1e-14  # close to numerical precision

optimizer = 'least_squares'  # other choice is 'CMA-ES'
with_bounds = False # Flag to indicate whether to use bounds in the least squares optimization
bounds = (-5, 5) # Bounds for the least squares optimization

# _________ Unary symbols ____________
# functions to be used in the equations ;
# note that sqrt( must be set to True or False below in get_config() and MUST NOT enter these lists

no_function_list = []
small_function_list = ['np.exp(', 'np.log(', 'np.sin(']
medium_function_list = ['np.sin(', 'np.cos(', 'np.log(', 'np.exp(', 'np.tan(', 'np.arcsin(',
                        'np.arccos(']
large_function_list = ['np.sin(', 'np.cos(', 'np.log(', 'np.exp(', 'np.tan(', 'np.arcsin(',
                       'np.arccos(', 'np.arctan(', 'np.arcsinh(', 'np.arccosh(', 'np.arctanh(', 'np.sinh(',
                       'np.cosh(', 'np.tanh(']  # default

#for feature counting
trig_functions = {'np.sin(', 'np.cos(', 'np.tan(', 'np.arcsin(', 'np.arccos(', 'np.arctan('}
exp_log_functions = {'np.log(', 'np.exp(', 'np.arcsinh(', 'np.arccosh(', 'np.arctanh(', 'np.sinh(', 'np.cosh(',
                     'np.tanh('}

# used elsewhere to eval a given formula given variables ; if your formula contains other functions like special functions,
# you need to add them here ; see Utils.parsing.convert_to_internal_names_and_numpy_representation
all_function_list = ['sqrt'] + [x[3:-1] for x in large_function_list] + ['ln'] #last one is because of I.44.4 written with ln instead of log

use_favor = True

# misc options
enforce_monocore = 0
use_parallel_tree_generation = True
if enforce_monocore:
    use_parallel_tree_generation = False

use_seed = False
skip_tree_validation = True # Faster, but not recommended if you start changing the code : it is a good idea to check the trees each time they are generated or modified
two_staged_run = True # if True, will run a second time on a larger grid and with a greater maximal length if no solution is found, see paper

# Number of CPU cores to use
def default_n_cores(proportion) -> int:
    """
    Get the default number of CPU cores to use.
    Returns:
    int: Number of CPU cores.
    """
    return 40
    available_cores = psutil.cpu_count(logical=True)
    return max(int(proportion * available_cores), 1)

n_cores = default_n_cores(0.9)  # Number of CPU cores to use

def get_config(search_intensity: str) -> dict:
    """
    Get the configuration based on search intensity.

    Parameters:
    search_intensity (Union[int, str]): Search intensity level (0, 1, 2, 3, or 'custom').

    Returns:
    dict: Configuration dictionary.
    """
    assert search_intensity in ['custom'], 'add other search intensities if you need'

    if search_intensity == 'custom':
        config = {

            # Hyper parameters
            'apply_dimensional_analysis' : True,  # Flag to indicate if we want to apply dimensional analysis in the process of creating/evolving equations

            # our method can generate and precompute extra variables that are adimensional combinations of original variables, and also norm-like variables
            'add_adim_variables' : 1,  # if you have two lengths, you can have a ratio that is adimensional y0 = x0/x1
            'add_inverse' : 1,  # if True, you would also add y1 = x1/x0
            'add_products' : 1,  # if True, when you have y0, y1, ..., you also add y2 = y0*y1, etc1
            'add_norms' : 1,  # if True, if you have two lengths, you automatically form scalar products like sqrt(x0^^2 + x1^^2) ; sqrt(x0^^2 - x1^^2)

             # vocabulary related
            'use_sqrt': True,
            'use_power': True,
            'max_power': 10, # maximum power to use in the equations ie x^11 is not allowed
            'function_list': large_function_list,

            # target related
            'use_noise': True,  # add noise to the target values
            'noise_level': 0.1, #only allowed 0.0, 0.001, 0.01, 0.1 as in literature
            'use_denoising': False,  # use denoising in the preprocessing step using KD Tree algorithm
            'k_neighbors': 2,  # number of neighbors to use in the KD Tree algorithm

            # Genetic algorithm related
            'genetic_iterations': 1000000,  # Number of max iterations for the genetic algorithm
            'timeout': 3600,  # Timeout in seconds for the genetic algorithm, total true timeout is twice this value since we have two runs in one
             # default is 1 hour (+ 1 hour on a larger grid, longer individuals)
            'max_equations_seen': 1000000,  # Maximum number of equations to try in the genetic algorithm
             # (algo will stop when any of the three conditions is met: max_equations_seen, genetic_iterations, timeout)
             # (for these runs, we only use the timeout, since max_iterations and max_equations_seen are way too large to be reached)

            'maximal_length': 35,  # Maximum length of the equations (== number of nodes or symbols)
            'qd_grid_size': 'medium', # qd_grid size to use in the genetic algorithm, can be 'small', 'medium', 'large', 'very_large' see Utils/Utils_run.py
            'initial_pool_size': 10000,  # Number of random equations to start with
            'extend_pool_factor': 1,  # by how much we extend the pool at each iteration
            'parent_selection_method': 'best_loss', # other available option is 'random' : best loss selects parents with some decresing probability based on their ranked loss
            'p_mutate': 0.4,  # Mutation probability
            'p_cross': 0.8,  # Crossover probability
            'proba_double_mutation': 0.33,  # Probability of double mutation (if mutation is selected)

            # Evaluation related
            'use_simplification': True, # simple hand-designed simplification rules are applied on the fly to the equations (not sympy based)
            'target_size' : 250, # number of points to use in the evaluation of the equations; we don't need the 1e6 points : would be useless and very slow

            # Misc ; can be changed at will ; low impact on performance
            'verbose': 3, # 0, 1, 2, 3 are the possible values ; 3 is the most verbose ;
            # details (level 3) are printed in blue : info (level2) in green : better to read in dark mode !
        }

    else:
        raise NotImplementedError

    if not config['add_adim_variables']:
        config['add_inverse'] = False
        config['add_products'] = False
    if not config['apply_dimensional_analysis']:
        config['add_adim_variables'] = False
        config['add_inverse'] = False
        config['add_products'] = False
        config['add_norms'] = False

    if metric == 'NRMSE':
        assert failure_loss > max_bad_loss, 'failure loss must be greater than max_bad_loss with NRMSE metric'
    elif metric == 'R2':
        assert failure_loss < max_bad_loss, 'failure loss must be less than max_bad_loss with R2 metric'

    assert config['use_sqrt'] in [True, False], 'use_sqrt must be set to True or False'
    assert config['use_power'] in [True, False], 'use_power must be set to True or False'
    assert 'np.sqrt(' not in config[
        'function_list'], 'np.sqrt( cannot be in function list ; fill the use_sqrt flag instead'

    if not config['use_noise']:
        config['use_denoising'] = False
        config['noise_level'] = 0
    assert config['noise_level'] != 0 if config['use_noise'] else True, 'noise level must be non zero if noise is used'
    assert config['noise_level'] in [0.0, 0.001, 0.01, 0.1], 'noise level must be 0.0, 0.001, 0.01, 0.1'

    assert config['k_neighbors'] > 0 if config[
        'use_denoising'] else True, 'k_neighbors must be greater than 0 if denoising is used'

    assert config['genetic_iterations'] > 0, 'genetic_iteration must be greater than 0'

    assert config['maximal_length'] > 0, 'maximal_length must be greater than 0'
    assert config['initial_pool_size'] > 0, 'initial_pool_size must be greater than 0'
    assert config['extend_pool_factor'] > 0, 'extend_pool_factor must be greater than 0'
    assert config['extend_pool_factor'] <= 5, 'extend_pool_factor can be set to a large number but will take much longer time ; remove assert error if you want this'
    assert config['qd_grid_size'] in ['small', 'medium', 'large'], 'qd_grid_size must be set to small, medium, large, or very_large'
    assert config['parent_selection_method'] in ['random',
                                                 'best_loss'], 'selection_method must be set to random or best_loss'
    assert optimizer in ['least_squares','CMA-ES'], 'optimizer must be set to least_squares or CMA-ES or implemented in Evaluate_fit.py'
    assert config['verbose'] in [0, 1, 2, 3], 'verbose must be set to 0, 1, 2 or 3'

    return config
