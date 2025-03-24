"""
GP_Solver Module

This module provides the main functionalities for solving equations using genetic programming.
It includes methods for initializing data and vocabulary, performing preliminary dimensional analysis.

Main functionalities:
- Initialize data and vocabulary.
- Execute the genetic programming algorithm to find the target equation.
- Evaluate the pool of equations and analyze the results, save them in a file.
- Log information about the whole procedure.
- Handle early stopping based on the best loss.
"""
from config import config
import time
import logging
import colorlog
import os
import concurrent.futures
from typing import List, Tuple, Any, Dict
from concurrent.futures import ProcessPoolExecutor
from Utils import Utils_ground_truth
from core import (QD, GP, Target, Data_Loader, Vocabulary, Evaluate_Formula, simplify_pool, to_sympy, to_sympy_without_simplification,
                  identify_float, convert_back)

def eval_individual(task : Tuple[int, str, Target, bool, dict]) -> [int, list, float]:
    """
    Evaluate an individual equation.
    Parameters:
    task (Tuple[int, str, Target, dict]): A tuple containing the following elements:
        - tree_number: The index of the tree in the pool.
        - formula: The formula to be evaluated.
        - target: The target data for evaluation.
        - use_noise: A boolean indicating whether to use noise in the evaluation.
        - ground_truth: The ground truth dictionary containing the formula information.

    Returns:
    Tuple[int, list, list, float, float]: A tuple containing the following elements:
    - tree_number: The index of the tree in the pool.
    - scalar_values_no_noise: The best fitted scalar values of the equation for noiseless data.
    - scalar_values_noise: The best fitted scalar values of the equation for noisy data.
    - loss_no_noise: The loss of the equation for noiseless data.
    - loss_noise: The loss of the equation for noisy data.
    If use_noise is False, loss_noise is equal to loss_no_noise and scalar_values_noise is equal to scalar_values_no_noise.

    """
    tree_number, formula, target, use_noise, ground_truth = task
    if not use_noise:
        ef = Evaluate_Formula(formula, target, config.optimizer, use_noise, ground_truth)
        scalar_values_no_noise, loss_no_noise = ef.evaluate()
        loss_noise = loss_no_noise # since no noise
        scalar_values_noise = scalar_values_no_noise
    else: # use noise
        ef = Evaluate_Formula(formula, target, config.optimizer, False, ground_truth)
        scalar_values_no_noise, loss_no_noise = ef.evaluate()
        ef = Evaluate_Formula(formula, target, config.optimizer, True, ground_truth)
        scalar_values_noise, loss_noise = ef.evaluate()
    return tree_number, scalar_values_no_noise, scalar_values_noise, loss_no_noise, loss_noise


def reformat_result(results: List, pool: List) -> List[Dict[str, Any]]:
    """ Reformat the results from the evaluation of the pool of equations.

    """
    format_results = []
    for result in results:
        tree_number, scalar_values_no_noise, scalar_values_noise, loss_no_noise, loss_noise = result
        tree = pool[tree_number]
        (num_scalars, equation_length, num_variables, num_functions,
         num_powers, num_trig_functions, num_exp_log_functions,
         num_plus_operator, num_minus_operator, num_times_operator,
         num_divide_operator, num_nested_functions) = tree.get_features()

        format_results.append({
            'loss_noise': loss_noise,
            'loss_no_noise': loss_no_noise,
            'tree': tree,
            'num_scalars': num_scalars,
            'scalar_values_no_noise': scalar_values_no_noise,
            'scalar_values_noise': scalar_values_noise,
            'equation_length': equation_length,
            'num_variables': num_variables,
            'num_functions': num_functions,
            'num_powers': num_powers,
            'num_trig_functions': num_trig_functions,
            'num_exp_log_functions': num_exp_log_functions,
            'num_plus_operator': num_plus_operator,
            'num_minus_operator': num_minus_operator,
            'num_times_operator': num_times_operator,
            'num_divide_operator': num_divide_operator,
            'num_nested_functions': num_nested_functions
        })
    return format_results

def parallel_eval_individual(tasks: List[Tuple[int, str, Target, bool, dict]], n_cores: int) -> List[Dict[str, Any]]:
    """
    Wrapper function to handle parallel execution of eval_individual.

    Parameters:
    tasks (List[Tuple[Target, Vocabulary, State, str, int, bool]]): List of tasks to be evaluated.
    n_cores (int): Number of cores to use for parallel execution.

    Returns:
    List[Dict[str, Any]]: List of results from eval_individual.
    """
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(eval_individual, task) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results


class GPSolver:
    def __init__(self, dataset_name,
                 equation_label: str,
                 run_number: int,
                 multi_processing: bool = False,
                 cfg: Dict[str, Any] = None,
                 ground_truth: Dict[str, Any] = None):

        """
        Parameters:
        equation_label (str): The label of the equation, in the sense of FeynmanAI dataset.
        run_number (int): The run number.
        cfg (Dict[str, Any]): Other critical configuration parameters not directly used here; see config.py
        ground_truth (dict): Ground truth dictionary containing optional and mandatory description of true targets.

        # example of ground_truth:
        {'formula': '1/2*k_spring*x**2',                                                                # may be None if not known
        'n_variables': 2,                                                                               #required
        'variables_true_name': ['k_spring', 'x'],                                                       #required
         'variables_dict': {'x0': {'true_name': 'k_spring', 'units_as_a_list': [0, -2, 1, 0, 0]},       #required ; can be [0, 0, 0, 0, 0] if no units
                            'x1': {'true_name': 'x', 'units_as_a_list': [1, 0, 0, 0, 0]}},              #required
         'internal_name_to_true_name_dict': {'x0': 'k_spring', 'x1': 'x'},                              #required
         'true_name_to_internal_name_dict': {'k_spring': 'x0', 'x': 'x1'},                              #required
         'target_dimension': [2, -2, 1, 0, 0],                                                          #required
         'is_target_everywhere_positive': True,                                                         #required
         'is_target_everywhere_negative': False}                                                        #required
        """

        self.dataset_name = dataset_name
        self.equation_label = equation_label
        self.ground_truth = ground_truth

        if not cfg['apply_dimensional_analysis']:
            self.ground_truth = Utils_ground_truth.remove_all_dimensional_info_in_groud_truth(ground_truth)

        self.run_number = run_number
        self.multi_processing = multi_processing
        self.metric = config.metric

        # unpack config
        self.cfg = cfg
        self.use_noise = cfg['use_noise']
        self.noise_level = cfg['noise_level']
        self.use_denoising = cfg['use_denoising']
        self.use_simplification = cfg['use_simplification']
        self.genetic_iterations = cfg['genetic_iterations']
        self.maximal_length = cfg['maximal_length']
        self.apply_global_minus_one = self.ground_truth['apply_global_minus_one']

        # Initialize the paths
        self.get_paths()  # create directories, define filepath and output_folder

        # Create a logger
        self.define_logger(verbosity=cfg['verbose'])

        # Initialize the vocabulary
        vocabulary = Vocabulary()
        vocabulary.build_vocabulary(self.cfg, self.ground_truth)
        self.vocabulary = vocabulary

        # Initialize the data:
        self.is_valid_data = self.initialize_data()

        # log of the run parameters
        self.log_init()
        if self.multi_processing:
            self.log_run_parameters()

    ############## I/O ################
    def get_paths(self):
        """
        Create directories and define file paths and output folder.
        """
        self.main_directory = f'Targets/{self.dataset_name}/'

    def set_logging_level(self):
        if self.verbosity == 0:
            level = logging.CRITICAL  # Critical only
        elif self.verbosity == 1:
            level = logging.IMPORTANT  # Important messages
        elif self.verbosity == 2:
            level = logging.INFO  # Informational messages
        elif self.verbosity == 3:
            level = logging.DETAILS  # All details
        else:
            raise ValueError("Verbosity level must be between 0 and 3")

        logging.basicConfig(level=level)  # Set the global logging level

    ############## Logger ################
    def log_blank_line(self):
        self.logger.info("")  # This logs an empty message, which appears as a blank line

    def define_logger(self, verbosity):
        """
        Define the logger for the GPSolver.

        Configures a logger with custom levels, handlers for both console and file outputs,
        and color-coded formatting for console logs.

        Parameters:
            verbosity (int): Verbosity level (0: CRITICAL, 1: ERROR, 2: INFO, 3: DEBUG).
        """
        # Create or retrieve the logger
        self.logger = logging.getLogger('my_logger')

        # Ensure the logger is configured only once
        if not self.logger.hasHandlers():
            # Define custom log levels
            IMPORTANT = 45  # Custom level higher than ERROR
            DETAILS = 15    # Custom level lower than INFO
            logging.addLevelName(IMPORTANT, "IMPORTANT")
            logging.addLevelName(DETAILS, "DETAILS")

            # Add methods for custom log levels
            def important(self, message, *args, **kws):
                if self.isEnabledFor(IMPORTANT):
                    self._log(IMPORTANT, message, args, **kws)

            def details(self, message, *args, **kws):
                if self.isEnabledFor(DETAILS):
                    self._log(DETAILS, message, args, **kws)

            logging.Logger.important = important
            logging.Logger.details = details

            # Determine logging level based on verbosity
            if verbosity == 0:
                level = logging.CRITICAL
            elif verbosity == 1:
                level = logging.ERROR
            elif verbosity == 2:
                level = logging.INFO
            elif verbosity == 3:
                level = logging.DEBUG
            else:
                raise ValueError("Verbosity level must be between 0 and 3")

            self.logger.setLevel(level)

            # Set up log file path
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_path = os.path.join(root_dir, 'logs', 'log_file.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # Create handlers
            console_handler = logging.StreamHandler()  # Logs to console
            file_handler = logging.FileHandler(log_path)  # Logs to a file

            # Configure formatters
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                log_colors={
                    'DEBUG': 'white',
                    'INFO': 'green',
                    'IMPORTANT': 'red',
                    'DETAILS': 'blue',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            console_handler.setFormatter(console_formatter)
            file_handler.setFormatter(file_formatter)

            # Add handlers to the logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def log_init(self):
        formula = self.ground_truth['formula'] if self.ground_truth['formula'] is not None else 'unknown formula'
        self.log_blank_line() # not working properly ; not a big issue
        self.logger.critical('Start solving ' + self.equation_label + ' ' + formula + ' with variable names: ' + ' '.join(self.ground_truth['true_name_to_internal_name_dict'].keys()))
        for var, dim in self.ground_truth['dimension_of_variables'].items():
            self.logger.info(f'With var name {self.ground_truth['internal_name_to_true_name_dict'][var]} and dimensions: {dim}')
        self.logger.info(f'And target dimension: {self.ground_truth['target_dimension']}')

    def log_run_parameters(self):
        """
        Log the run parameters.
        """
        self.logger.details('RUN PARAMETERS -- modify at will the custom config')
        self.logger.details(
            f"optimizer: {config.optimizer}, apply dim analysis: {self.cfg['apply_dimensional_analysis']}, run_number: {self.run_number}")
        self.logger.details(
            f"functions: {self.cfg['function_list']}")
        self.logger.details(
            f"use_noise: {self.use_noise}, noise_level: {self.noise_level}, use_simplif: {self.use_simplification}")
        self.logger.details(
            f"maximal_length: {self.maximal_length})")
        self.logger.details(
            f"initial_pool_size: {self.cfg['initial_pool_size']}, extend_pool_factor: {self.cfg['extend_pool_factor']}, multi_processing: {self.multi_processing}")
        self.logger.details(
            f"n_cores: {config.n_cores}, verbose: {self.cfg['verbose']}, meta_mode: {config.meta_mode}")

        self.logger.details(f'Ground_truth items:')
        for k, v in self.ground_truth.items():
            self.logger.details(f'{k}: {v}')
        self.logger.info(f'Using vocabulary: {self.vocabulary.all_symbols}')

    ############## Data initialization ################
    def initialize_data(self) -> bool:
        try:
            self.dataloader = Data_Loader(self.apply_global_minus_one, self.logger,
                                     self.use_noise, self.noise_level, self.use_denoising,
                                     self.equation_label, self.main_directory,
                                     self.cfg['k_neighbors'])
            success = True
        except Exception as e:
            print(f'Error initializing data: {e}')
            success = False
        return success

    ############## Read Dimensional Analysis Results ################

    def find_trivial_solution(self, eigenvector: List[int]) -> str:
        """
        Find the trivial solution of a dimensionally trivial equation.
        Equation must be of the form A* (x0)**p_0 * (x1)**p_0 ... where:
            - A is the scalar to find,
            - x0, x1, ... are the variables in internal names,
            - p_0, p_1, ... are the powers listed in the eigenvector. (see utils_ground_truth - dimensional analysis for more details)
        Parameters:
        eigenvector (List[int]): The eigenvector associated to the equation.
        """
        self.subsample_trained_target = self.dataloader.get_subsample_target(self.cfg['target_size'])
        expr = 'A*('
        eigenvector = -1*eigenvector[1:]
        for i in range(len(eigenvector)):
            if eigenvector[i] == 0:
                continue
            elif eigenvector[i] == 1:
                expr += f'x{i} * '
            else:
                expr += f'x{i}**({eigenvector[i]}) * '
        expr = expr[:-3] + ')'
        readable_expr = expr
        for k, v in self.ground_truth['internal_name_to_true_name_dict'].items():
            readable_expr = readable_expr.replace(k, v)
        self.logger.critical(f'And must be of the following form: {readable_expr}')

        # send analysis to find best A #should be replaced by using only one data point
        ef = Evaluate_Formula(expr, self.subsample_trained_target, config.optimizer, self.cfg['use_noise'], self.ground_truth)
        scalar_values, loss = ef.evaluate()
        if scalar_values[0] == 1:
            final_answer = readable_expr[3:-1]
        else:
            # try id the float
            float_identity = identify_float(scalar_values[0])
            if float_identity is not None:
                final_answer = readable_expr.replace('A',  str(convert_back(*float_identity)))
            else:
                final_answer = readable_expr.replace('A', str(scalar_values[0]))
            final_answer = final_answer.replace('^', '**')
        if self.apply_global_minus_one:
            final_answer = f'-({final_answer})'

        # replace back ksi_vec and ksi to A_vec and A, cf Targets/cleaning_dataset
        final_answer = final_answer.replace('ksi_vec', 'A_vec').replace('ksi', 'A')
        self.logger.critical(f'Best fit is {final_answer}')
        return final_answer

    def initial_dimensional_analysis(self):
        """
        Dimensional analysis has already been performed when we built the ground_truth dictionary.
        If the equation is dimensionally trivial, we can find the trivial solution. (eg. E = 3.21 * m * g *z)
        all we need is to find the scalar value of the trivial solution.
        Returns:
        bool: Whether the equation is found or not.
        str: The final form of the equation when found, else None

        """
        is_equation_possible = self.ground_truth['is_equation_possible']
        is_equation_trivial = self.ground_truth['is_equation_trivial']
        final_form = 'None'
        found = False

        if not is_equation_possible:
            self.logger.critical(f'Equation {self.equation_label} is not dimensionnaly possible, aborting')
            final_form = 'impossible equation'
            found = True

        if is_equation_trivial:
            self.logger.critical(f'Equation {self.equation_label} is dimensionnaly trivial')
            final_form = self.find_trivial_solution(self.ground_truth['eigenvectors_with_target'][0])
            found = True

        return found, final_form

    ############## Evaluation ################
    def evaluate_pool(self, pool: List[Any], target: Target) -> List[Dict[str, Any]]:
        """
        Evaluate a pool of equations.

        Parameters:
        pool (List[State]): A list of equations to be evaluated.
        target (Target): The target data for evaluation.

        Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the evaluation results and extracted features.
        """
        monocore = not self.multi_processing
        if config.enforce_monocore:
            monocore = True
        #print('INFO monocore', monocore)
        if not monocore:  # parallel call to eval_individual
            tasks = []
            for i, tree in enumerate(pool):
                tasks.append((i, tree.infix_formula, target, self.use_noise, self.ground_truth))
            results = parallel_eval_individual(tasks, config.n_cores)
        else:
            results = []
            for i, tree in enumerate(pool):
                task = (i, tree.infix_formula, target, self.use_noise, self.ground_truth)
                results.append(eval_individual(task))
        return reformat_result(results, pool)

    ############## Analysis ################
    def save_all_equations_seen(self, results: List[Dict[str, Any]], all_equations_seen: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Updates the dictionary of all equations seen during the training.
        Parameters:
        results (list): A list of dictionaries containing evaluation results.
        all_equations_seen (dict): A dictionary to store all unique equations seen so far.
        Returns:
        dict:
        """
        for result in results:
            tree = result['tree']
            if str(tree.postfix_formula) not in all_equations_seen:
                all_equations_seen.update({str(tree.postfix_formula): 1})
        return all_equations_seen

    def rank_results(self, qd_grid_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank the results based on the loss.
        Parameters:
        qd_grid_elements (list): A list of dictionaries containing the evaluation results.
        Returns:
        list: A list of dictionaries containing the ranked evaluation results. First element is the best.
        """
        revert = True if self.metric == 'R2' else False
        sorting_key = 'loss_no_noise' if not self.use_noise else 'loss_noise'
        return sorted(qd_grid_elements, key=lambda x: x[sorting_key], reverse=revert)


    def logger_format_individual(self, individual: Dict[str, Any]) -> str:
        """
        Format an individual equation.
        Parameters:
        individual (dict): A dictionary containing the evaluation results of an equation.
        Returns:
        str: A formatted string of the individual equation.
        """
        return f'Loss noise: {individual["loss_noise"]}, Loss no noise: {individual["loss_no_noise"]}, Equation: {individual["tree"].infix_formula}, Scalars noise: {individual["scalar_values_noise"]}, Scalars no noise: {individual["scalar_values_no_noise"]}'

        if not self.use_noise:
            return f'Loss: {individual["loss_no_noise"]}, Equation: {individual["tree"].infix_formula}, Scalars: {individual["scalar_values_no_noise"]}'
        else:
            return f'Loss: {individual["loss_noise"]}, Equation: {individual["tree"].infix_formula}, Scalars: {individual["scalar_values_noise"]}'

    ############## Main ################
    def initialize_qd_gp(self, initial_qd_grid):
        QD_object = QD(self.maximal_length, self.cfg['qd_grid_size'], self.use_noise, self.logger, initial_qd_grid)
        GP_object = GP(self.vocabulary, self.use_simplification, self.cfg, self.ground_truth, self.run_number, self.logger)
        pool_of_trees = QD_object.return_all_trees() if initial_qd_grid else []
        return QD_object, GP_object, pool_of_trees

    def extend_pool(self, GP_object, QD_object, pool_of_trees, iteration):
        if iteration == 0:
            num_trees = self.cfg['initial_pool_size']
            pool_of_trees += GP_object.create_pool(num_trees, self.maximal_length)
            for tree in pool_of_trees:
                tree.check_tree(self.ground_truth['target_dimension'])

        else:
            pool_of_trees = GP_object.extend_pool(QD_object, self.maximal_length)
            for tree in pool_of_trees:
                tree.check_tree(self.ground_truth['target_dimension'])

        if self.use_simplification:
            pool_of_trees = simplify_pool(pool_of_trees, self.vocabulary)
        return pool_of_trees

    def analyze_results(self, QD_object):
        qd_grid_elements = QD_object.return_all_results()
        if qd_grid_elements:
            ranked_individuals = self.rank_results(qd_grid_elements)
            best_loss = ranked_individuals[0]['loss_no_noise'] if not self.use_noise else ranked_individuals[0]['loss_noise']
        else:
            ranked_individuals = []
            best_loss = config.failure_loss
        return ranked_individuals, best_loss

    def log_iteration_details(self, num_eq_seen, QD_object, best_loss, ranked_individuals):
        total_time = time.time() - self.start_time
        self.logger.details(f'Eval time: {self.eval_time}, Total time: {total_time},'
                         f' Proportion of time spent in eval: {self.eval_time / total_time}')
        self.logger.info(f'All Equations Seen: {num_eq_seen}')
        self.logger.info(f'QD Pool Size is now: {len(QD_object.qd_grid)}')
        self.logger.info(f'Best {config.metric} Loss: {best_loss}')
        # print the best three
        for rank, individual in enumerate(ranked_individuals[:3]):
            self.logger.details(f"Best {rank + 1}: {self.logger_format_individual(individual)}")


    def extract_pareto_front(self, ranked_individuals):

        def get_complexity(individual):
            postfix_expr = individual['tree'].postfix_formula
            complexity_weights = {
                '+': 1, '*': 1,
                'A':2, # free scalar
                '-': 2, '/': 2, '**': 3,  # Basic arithmetic
                'np.log(': 3, 'np.exp(': 3, 'np.sqrt(': 3,  # Exponential/log functions
                'np.sin(': 4, 'np.cos(': 4, 'np.tan(': 4,  # Trigonometric functions
                'np.sinh(': 5, 'np.cosh(': 5, 'np.tanh(': 5,  # Hyperbolic functions
                'np.arcsin(': 5, 'np.arccos(': 5, 'np.arctan(': 5,  # Inverse trigonometric functions
                'np.arcsinh(': 6, 'np.arccosh(': 6, 'np.arctanh(': 6,  # Inverse trigonometric functions
            }

            complexity = 0
            for token in postfix_expr:
                complexity += complexity_weights.get(token, 1)  # Default weight 1 for unknown tokens : will be all the variables or norms
            return complexity

        key = 'loss_no_noise' if not self.use_noise else 'loss_noise' #not really needed as pareto front is used only in case of noise; anyway
        if self.metric == 'R2':
            # Group by complexity and select max R^2 per complexity
            complexity_dict = {}
            for individual in ranked_individuals:
                complexity = get_complexity(individual)
                r2 = individual[key]
                if complexity not in complexity_dict or r2 > complexity_dict[complexity][0]:
                    complexity_dict[complexity] = [r2, individual]

        elif self.metric == 'NRMSE':
            # Group by complexity and select min NRMSE per complexity
            complexity_dict = {}
            for individual in ranked_individuals:
                complexity = get_complexity(individual)
                nrmse = individual[key]
                if complexity not in complexity_dict or nrmse < complexity_dict[complexity][0]:
                    complexity_dict[complexity] = [nrmse, individual]
        else:
            raise NotImplementedError

        # sort them by complexity
        pareto_front = [complexity_dict[k][1] for k in sorted(complexity_dict.keys())]
        self.logger.info(f'Pareto front size: {len(pareto_front)}')
        return pareto_front

    def check_early_stopping(self, ranked_individuals, local_target, best_loss):

        #case no noise : we just check the loss against the termination loss
        if not self.use_noise:
            if (self.metric == 'NRMSE' and best_loss < config.termination_loss) or (self.metric == 'R2' and best_loss > config.termination_loss):
                self.logger.info("Early stopping criterion met.")
                return True, ranked_individuals[0]
            return False, ranked_individuals[0]

        else:
            # with noise, termination loss will almost never be hit. Here literature is very vague on the exact methodology
            # we will first retrieve the best elements (as given by the loss on noisy data) along some commplexity - Pareto front
            # then we will check if these elements, when fitted on the *noiseless data*, meet the termination criterion
            # (pareto front is itself a bit vague and vary depending on the authors)
            pareto_front = self.extract_pareto_front(ranked_individuals)
            best_loss = -1e6 if self.metric == 'R2' else 1e6
            best_individual = None

            if not len(pareto_front):
                self.logger.info("No Pareto front found.")
                return False, None

            for individual in pareto_front:
                #retrieve candidate equation:
                tree = individual['tree']
                formula = tree.infix_formula
                # Find best scalars ON THE NOISELESS DATA to check for exact recovery.
                # Basic idea is not to check the whole pool as it would be cheating,
                # but check only few elements of it given some loss-complexity optimization
                ef = Evaluate_Formula(formula, local_target, config.optimizer, False, self.ground_truth)
                scalar_values, loss = ef.evaluate()
                #we need to save them in the individual
                individual['scalar_values_no_noise'] = scalar_values
                individual['loss_no_noise'] = loss
                if self.metric == 'NRMSE' and loss < best_loss:
                    best_loss = loss
                    best_individual = individual
                elif self.metric == 'R2' and loss > best_loss:
                    best_loss = loss
                    best_individual = individual
                if (self.metric == 'NRMSE' and loss < config.termination_loss) or (self.metric == 'R2' and loss > config.termination_loss):
                    self.logger.info("Early stopping criterion met.")
                    return True, best_individual
            return False, best_individual

    def execute(self, initial_qd_grid = None, all_equations_seen = None) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]], int, Dict[str, Any], QD]:
        """
        Execute the genetic programming (GP) algorithm with a Quality-Diversity (QD) framework.

        This method iteratively generates, evaluates, and evolves a population of symbolic trees
        to optimize a target function. The QD framework projects solutions into a feature space
        and maintains a diverse population across iterations.

        Parameters:
            initial_qd_grid (Optional[Dict[str, Any]]): The initial QD grid to start with, containing pre-computed solutions.

        Returns:
                - early_stopping (bool): Indicates if the algorithm stopped early based on a termination criterion.
                - candidate_solution (Dict[str, Any]): The best individual found, if any.
                - ranked_individuals (List[Dict[str, Any]]): Ranked evaluation results, from best to worst.
                - iterations (int): Total number of iterations completed.
                - all_equations_seen (Dict[str, Any]): Dictionary of all unique equations seen during the search.
                - QD_object (QD): The final QD object containing the grid of solutions.
        """

        self.start_time = time.time()
        self.eval_time = 0
        if initial_qd_grid is not None: self.logger.important('Starting with a non empty initial qd grid')

        # Initialization
        all_equations_seen = {} if all_equations_seen is None else all_equations_seen
        QD_object, GP_object, pool_of_trees = self.initialize_qd_gp(initial_qd_grid)

        early_stopping = False
        candidate_solution = None
        ranked_individuals = None

        for iteration in range(self.genetic_iterations):
            if time.time() - self.start_time > self.cfg['timeout']:
                self.logger.critical(f'Timeout reached after {iteration} iterations')
                break
            if len(all_equations_seen) >= self.cfg['max_equations_seen']:
                self.logger.critical(f'Maximum number of equations seen reached after {iteration} iterations')
                break
            self.logger.important(f'This is iteration {iteration} for eq {self.equation_label}, Maximal Length: {self.maximal_length}')

            # Pool extension
            pool_of_trees = self.extend_pool(GP_object, QD_object, pool_of_trees, iteration)
            for tree in pool_of_trees:
                tree.check_tree(self.ground_truth['target_dimension'])
            self.logger.info(f'Pool Size to Evaluate: {len(pool_of_trees)}')

            # Resample the target data
            local_target = self.dataloader.get_subsample_target(self.cfg['target_size'])

            # Pool evaluation
            t = time.time()
            results = self.evaluate_pool(pool_of_trees, local_target)
            self.eval_time += time.time() - t

            all_equations_seen = self.save_all_equations_seen(results, all_equations_seen)

            # Update the QD pool
            binned_results = QD_object.project_results_in_bins(results)
            QD_object.update_qd_grid(binned_results)
            #QD_object.save(self.output_folder, self.vocabulary, self.cfg)

            # Analysis
            ranked_individuals, best_loss = self.analyze_results(QD_object)
            self.log_iteration_details(len(all_equations_seen), QD_object, best_loss, ranked_individuals)

            #debug infos
            qd_grid_elements = QD_object.return_all_results()
            revert = True if self.metric == 'R2' else False
            sorting_key = 'loss_no_noise'
            spure = sorted(qd_grid_elements, key=lambda x: x[sorting_key], reverse=revert)
            best_loss_no = spure[0]['loss_no_noise']
            sorting_key = 'loss_noise'
            spure = sorted(qd_grid_elements, key=lambda x: x[sorting_key], reverse=revert)
            best_loss = spure[0]['loss_noise']
            self.logger.info(f'Best loss no noise: {best_loss_no} and best loss noise: {best_loss}')
            # Early stopping
            early_stopping, candidate_solution = self.check_early_stopping(ranked_individuals, local_target, best_loss)
            if early_stopping:
                break

        # end search loop
        return early_stopping, candidate_solution, ranked_individuals, iteration, all_equations_seen, QD_object

    def solve(self, initial_qd_grid = None, all_equations_seen = None) -> Tuple[str, bool, bool, bool, int, float, Dict[str, Any], Any]:
        """
        Main method to solve the target equation using genetic programming with QD grid.

        This method integrates dimensional analysis, genetic programming execution, and post-processing
        to discover symbolic solutions for a given target equation. It handles trivial or impossible cases
        early and logs relevant updates throughout the process.

        Args:
            initial_qd_grid (Optional[Dict[str, Any]]): An optional initial QD grid to bootstrap the search.

        Returns:
            Tuple[str, bool, bool, int]:
                - best_individual (str): The symbolic representation of the best individual found.
                - is_equation_trivial (bool): Indicates if the equation was trivial to solve.
                - found (bool): Whether a solution was successfully found.
                - tot_iter (int): The total number of iterations completed.
                - tot_time (float): The total time taken to complete the search.
        """

        if not self.is_valid_data:
            self.logger.critical(f"Invalid data for target {self.equation_label}; aborting.")
            return "data invalid", True, False, False, 0, 0, {}, None

        # Use dimensional analysis results
        solution_found, final_form = self.initial_dimensional_analysis()
        if solution_found:
            return final_form, True, solution_found, False, 0, 0, {}, None

        # Execute search
        solution_found, candidate_solution, ranked_individuals, iteration, all_equations_seen, QD_object = self.execute(initial_qd_grid, all_equations_seen)
        # Post processing
        sympy_formula, total_time, exact_recovery = self.post_process_results(candidate_solution, solution_found)

        return str(sympy_formula), False, solution_found, exact_recovery, iteration, time.time() - self.start_time, all_equations_seen, QD_object

    def _apply_minus_one(self, formula):
        if self.apply_global_minus_one:
            formula = f'-({formula})'
        return formula


    ############## Post processing ################
    def not_found_case(self, candidate_solution: Dict[str, Any]):
        total_time = time.time() - self.start_time
        self.logger.critical(f'End, equation {self.equation_label} has not been found')
        self.logger.important(f'Total Elapsed time: {total_time}')
        best_formula = candidate_solution['tree'].infix_formula
        best_formula = self._apply_minus_one(best_formula)
        scalars = candidate_solution['scalar_values_no_noise']
        self.logger.important(f'Best raw formula found: {best_formula} with scalars {scalars}')
        timeout, exact_recovery, sympy_formula, simplified = to_sympy(best_formula, scalars, self.ground_truth)

        if timeout:
            self.logger.critical(f'Sympy.simplify timed out ; most likely, consider other options for simplification manually')
            sympy_formula = to_sympy_without_simplification(best_formula, scalars, self.ground_truth)
            self.logger.critical(f'Best formula found: {sympy_formula} vs true target {self.ground_truth["formula"]}')
        else:
            self.logger.critical(f'Best formula found: {sympy_formula} vs true target {self.ground_truth["formula"]}')
        return sympy_formula, total_time, exact_recovery

    def found_case(self, candidate_solution: Dict[str, Any]):
        total_time = time.time() - self.start_time
        self.logger.critical(f'End, equation {self.equation_label} found')
        self.logger.important(f'Total Elapsed time: {total_time}')
        best_formula = candidate_solution['tree'].infix_formula
        best_formula = self._apply_minus_one(best_formula)
        scalars = candidate_solution['scalar_values_no_noise']
        self.logger.important(f'Best raw formula found: {best_formula} with scalars {scalars}')
        timeout, exact_recovery, sympy_formula, simplified = to_sympy(best_formula, scalars, self.ground_truth)

        if timeout:
            self.logger.critical(f'Sympy.simplify timed out ; most likely, consider other options for simplification manually')
            sympy_formula = to_sympy_without_simplification(best_formula, scalars, self.ground_truth)
            self.logger.critical(f'Best formula found: {sympy_formula} vs true target {self.ground_truth["formula"]}')

        else:
            if exact_recovery:
                self.logger.important(f'After simplification: {simplified}')
                self.logger.critical(f'After mapping to variable names we have an exact recovery: {sympy_formula} vs true target {self.ground_truth["formula"]}')
            else:
                self.logger.critical(
                    f'No exact recovery detected - check by hand or copy-paste to Wolfram Alpha (sympy simplification is not as powerful)')
                self.logger.critical(
                    f'Best formula found is {sympy_formula} vs true target {self.ground_truth["formula"]}')
        return sympy_formula, total_time, exact_recovery

    def post_process_results(self, candidate_solution: Dict[str, Any], early_stopping: bool) -> Tuple[str, float, bool]:
        if not early_stopping:
            return self.not_found_case(candidate_solution)
        return self.found_case(candidate_solution)





