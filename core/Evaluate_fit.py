"""
Evaluate_fit.py

This module contains classes and methods for evaluating mathematical formulas against target data using various optimization techniques.
The main classes included are:

1. Evaluate_Formula: Evaluates a given formula against target data using specified optimization methods.
2. Least_Square_Optimizer: Performs least squares optimization to find the best free scalar values for a formula.
3. CMAES_Optimizer: Uses the CMA-ES algorithm to optimize free scalar values for a formula.
4. Other optimizers: Additional optimizers can be added to the module as needed.
"""

import random
from copy import deepcopy

import numpy as np
from config import config
from scipy.optimize import least_squares
from core import Target
from typing import List, Tuple, Optional
import cma


def compute_loss(formula: str, y_true: np.ndarray, X : np.ndarray, A: Optional[List] = None) -> float:
    """
    Evaluate a formula and compute the chosen loss.

    Parameters:
    - formula (str): The formula to be evaluated.
    - y_true (np.ndarray): The target data to be fitted.
    - X (np.ndarray): The variables array.
    - A (list): The free scalar values, optional, since there may be no free scalars.

    Returns: float: The computed loss. If the evaluation fails, return the failure loss.
    Evaluation may fail due to various reasons, such as division by zero, overflow, complex numbers, etc.

    """
    if config.metric == 'R2' and np.sum((y_true - np.mean(y_true)) ** 2) == 0:
        raise ValueError('R2 is not defined when the target data is constant, you must use NRMSE instead')

    try:
        target_size = y_true.shape[0]
        equation = eval(formula)
        if (not isinstance(equation, np.ndarray) or
                (
                        np.isnan(equation).any() or
                        np.isinf(equation).any() or
                        np.iscomplex(equation).any())
        ):
            return config.failure_loss

        if config.metric == 'NRMSE':
            quadratic_cost = np.sum((y_true - equation) ** 2)
            RMSE = np.sqrt(quadratic_cost / target_size)
            NRMSE = RMSE / np.std(y_true)
            return float(NRMSE)

        elif config.metric == 'R2':
            mean_data = np.mean(y_true)
            cost = np.sum((y_true - equation) ** 2) / np.sum((y_true - mean_data) ** 2)
            R2 = 1 - cost
            return float(R2)

        else:
            raise NotImplementedError('Unknown metric')

    except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError,
            TypeError) as e:
        # print('evaluation loss error', e)
        return config.failure_loss

class Evaluate_Formula:

    def __init__(self, formula: str, target: Target, optimizer: str, apply_noise : bool, ground_truth) -> None:
        """
        Initialize the Evaluate_Formula class.

        Parameters:
        - formula (str): The formula to be evaluated.
        - target (Target): The target object against which the formula will be evaluated.
        - optimizer (str): The optimization method to be used. Supported are : least_squares, CMA-ES

        Attributes:
        - formula (str): The formula to be evaluated.
        - scalar_numbers (int): The number of free scalar in the formula.
        - use_noise (bool): Flag indicating whether to use noisy target data.
        - variables (np.ndarray): The variables describing the target.
        - n_variables (int): The number of variables of the target function.
        - ranges_var (np.ndarray): The ranges of the variables.
        - target_size (int): The size of the target data.
        - f (np.ndarray): The target function values considered as a function of variables.
        - f_noise (np.ndarray): Same function with noise.
        - optimizer (str): The optimization method to be used.
        - data (np.ndarray): The target data to be fitted.
        - std_data (float): The standard deviation of the target data.
        """
        self.formula = formula
        self.scalar_numbers = 0
        self.variables = target.variables
        self.f, self.f_noise = target.f, target.f_noise
        self.target_size = target.target_size
        self.optimizer = optimizer
        self.target = self.f if not apply_noise else self.f_noise #which one to fit
        self.true_name_to_internal_name_dict = ground_truth['true_name_to_internal_name_dict']
        self.internal_name_to_true_name_dict = ground_truth['internal_name_to_true_name_dict']
        self.n_var = len(self.internal_name_to_true_name_dict)
        self.adimensional_dict = ground_truth['adimensional_dict'] #adimensional_dict {'y0': '(x2)/((x1))', 'y1': '(x2)/((x1))*(x3)/((x1))', 'y2': '(x1)/((x2))*(x1)/((x3))', 'y3': '(x3)/((x1))', 'y4': '(x1)/((x2))', 'y5': '(x1)/((x3))'}
        self.norms = ground_truth['norms'] #norms {'norm0': 'np.sqrt((x1)**2 + (x2)**2 + (x3)**2)'}

    def prepare_formula(self) -> None:
        """
            String manipulation on a formula to make it evaluable.
            This method updates the formula by:
            - Replacing the internal names of variables xk with the corresponding array column X[:, k].
            - Replacing the adimensional variables y_i with the appropriate array indexing.
            - Replacing the norms with the appropriate array indexing.
            - Index free scalars 'A' by A[1], A[2], etc.
        """

        for k in reversed(range(self.n_var)):
            self.formula = self.formula.replace(f'x{k}', f'X[:, {k}]')

        n_adim = len(self.adimensional_dict)
        for k in reversed(range(n_adim)):
            self.formula = self.formula.replace(f'y{k}', f'X[:, {self.n_var + k}]')

        n_norms = len(self.norms)

        for k in reversed(range(n_norms)):
            self.formula = self.formula.replace(f'norm{k}',  f'X[:, {self.n_var + n_adim + k}]')

        # Replace 'A' with indexed versions like 'A[0]', 'A[1]', etc.
        new_eq = ''
        A_count = 0
        for char in self.formula:
            new_eq += f'A[{A_count}]' if char == 'A' else char
            A_count += 1 if char == 'A' else 0

        self.formula = new_eq


    def evaluate(self) -> Tuple[List[float], float]:
        """
        Main call to evaluate a formula.

        Returns:
        - Tuple[List[float], float]: A tuple containing the optimized scalar values and the loss.
        """

        np.seterr(all = 'ignore') # ignore all warnings

        # count the numbet of free scalars in the formula
        self.scalar_numbers = self.formula.count('A')
        # cast the formula to a form that can be evaluated:
        self.prepare_formula() # eg : 'x0 + A*np.exp(x1)' -> 'X[:, 0] + A[0]*np.exp(X[:, 1])' ; X is a placeholder for the array of variables

        if self.scalar_numbers == 0: # no free scalar to optimize
            return [], compute_loss(self.formula, self.target, self.variables)

        if self.optimizer == 'CMA-ES':
            # CMA-ES optimization says : CAVEAT: Optimization in 1-D is poorly tested.
            # thus use least_squares for 1D optimization
            if self.scalar_numbers == 1:
                Optimizer = Least_Square_Optimizer(self.scalar_numbers, self.variables, self.target, self.formula)
                initial_guess = [2*random.random() - 1 for _ in range(self.scalar_numbers)] #why not
                success, scalar_values = Optimizer.best_A_least_squares(initial_guess, config.with_bounds)
            else:
                Optimizer = CMAES_Optimizer(self.scalar_numbers, self.variables, self.target, self.formula)
                success, scalar_values = Optimizer.best_A_cmaes()

        elif self.optimizer == 'least_squares':
            Optimizer = Least_Square_Optimizer(self.scalar_numbers, self.variables, self.target, self.formula)
            initial_guess = [2*random.random() - 1 for _ in range(self.scalar_numbers)] #why not
            success, scalar_values = Optimizer.best_A_least_squares(initial_guess, config.with_bounds)
        else:
            raise NotImplementedError('Unknown optimizer')

        if not success:
            return [1]*self.scalar_numbers, config.failure_loss
        else:
            return scalar_values, compute_loss(self.formula, self.target, self.variables, scalar_values)


class Least_Square_Optimizer:

    def __init__(self, scalar_numbers: int, variables: np.ndarray, target: np.ndarray, formula: str) -> None:
        """
        Initialize the Least_Square_Optimizer class.

        Parameters:
        - scalar_numbers (int): The number of free scalars in the formula.
        - variables (np.ndarray): The variables array.
        - target (np.ndarray): The target function values.
        - formula (str): The formula to be evaluated.
        """
        self.scalar_numbers = scalar_numbers
        self.variables = variables
        self.target = target
        self.formula = formula

    def func(self, A: np.ndarray, f: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the formula minus the actual target data with the given parameters.

        Parameters:
        - A (np.ndarray): The free scalar values.
        - f (np.ndarray): The target function values.
        - X (np.ndarray): The variables array.

        Returns:
        - np.ndarray: The evaluated result.
        """
        return eval(self.formula + '-f')

    def best_A_least_squares(self, initial_guess, with_bounds = False):
        """
        Apply least squares optimization starting from the given recommendation.

        Parameters:
        - initial_guess (np.ndarray): The initial recommendation for the scalar values.
        - with_bounds (bool): Flag indicating whether to use bounds in the optimization.

        Returns:
        - Tuple[bool, List[float]]: A tuple containing a boolean indicating success and the optimized scalar values.
        """
        try:
            bounds = (-np.inf, np.inf) if not with_bounds else (
                config.bounds[0]*np.ones_like(initial_guess), config.bounds[1]*np.ones_like(initial_guess)
            )
            least_square_result = least_squares(
                lambda a: self.func(a, self.target, self.variables).flatten(),
                initial_guess,
                jac='2-point',
                loss='cauchy',
                bounds = bounds
            )
            success = least_square_result.success
            if success:
                return True, least_square_result.x.tolist()
            else:
                return False, [1] * self.scalar_numbers
        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError, TypeError):
            return False, [1] * self.scalar_numbers


class CMAES_Optimizer:

    def __init__(self, scalar_numbers: int, variables: np.ndarray, target: np.ndarray, formula: str) -> None:
        """
        Initialize the CMAES_Optimizer class.
        Parameters:
        - scalar_numbers (int): The number of free scalars in the formula.
        - variables (np.ndarray): The variables array.
        - target (np.ndarray): The target function values.
        - formula (str): The formula to be evaluated.
        """
        self.scalar_numbers = scalar_numbers
        self.variables = variables
        self.target = target
        self.formula = formula
        self.n = self.target.shape[0]

    def best_A_cmaes(self, initial_guess =None) -> Tuple[bool, List[float]]:
        """
        Apply the CMA-ES optimization to find the best scalar values.

        Returns:
        - Tuple[bool, List[float]]: A tuple containing a boolean indicating success and the optimized scalar values.
        """
        np.seterr(all = 'ignore')
        initial_guess = 2 * np.random.rand(self.scalar_numbers) - 1 if initial_guess is None else initial_guess
        initial_sigma = np.random.randint(1, 5)

        try:
            res = cma.CMAEvolutionStrategy(
                initial_guess, initial_sigma, {'verb_disp': 0}
            ).optimize(self.compute_loss).result

            reco = res.xfavorite
            return True, reco.tolist()
        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError, UserWarning) as e:
            #print('CMA-ES optimization failed:', e)
            return False, [1] * self.scalar_numbers

    def compute_loss(self, A = None):
        """
        Evaluate the formula and compute the normalized root mean squared error (NRMSE).

        Parameters:
        - A (np.ndarray): The free scalar values.

        Returns:
        - float: The NRMSE value.
        """
        try:
            equation = eval(self.formula.replace('X', 'self.variables'))
            if isinstance(equation, np.ndarray) and not (np.isnan(equation).any() or np.isinf(equation).any() or np.iscomplex(equation).any()):
                quadratic_cost = np.sum((self.target - equation) ** 2)
                RMSE = np.sqrt(quadratic_cost / self.n)
                NRMSE = RMSE / np.std(self.target)
                return NRMSE
            else:
                return config.failure_loss
        except Exception as e:
            #print('eval loss error', e)
            return config.failure_loss

