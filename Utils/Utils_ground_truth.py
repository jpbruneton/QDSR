import pickle
import os
import csv
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict
import pandas as pd
from config import config
from sympy import Matrix
import numpy as np
from typing import List, Tuple, Generator
from Utils.Utils_misc import remove_unknown_cols_in_df
from Utils.Utils_parsing import infix_to_postfix
import Utils.Utils_io as Utils_io

def convert_eigenvector_to_infix_formula(eigenvector, variables_name):
    idx = [i for i in range(len(eigenvector)) if eigenvector[i] != 0]
    smallest = min([abs(eigenvector[i]) for i in idx])
    eigenvector = eigenvector / smallest

    num = []
    den = []

    for i in idx:
        variable = f'({variables_name[i]})'
        count = abs(int(eigenvector[i]))
        if eigenvector[i] > 0:
            num.extend([variable] * count)
        else:
            den.extend([variable] * count)

    num_str = '*'.join(num)
    den_str = '*'.join(den)

    if num_str and den_str:
        return f'{num_str}/({den_str})'
    elif num_str:
        return num_str
    else:
        return f'1/({den_str})'


def compute_dimension_of_postfix_list(
        postfix_as_a_list: List[str],
        dimensional_dict: Dict[str, List[int]],
        debug=False
) -> Tuple[bool, List[int]]:
    """
    Computes the dimension of a given formula in postfix notation.

    Parameters:
    - postfix_as_a_list (List[str]): The formula in postfix notation. Functions should be
      written as math-style names (e.g., 'sqrt'), not numpy-style names (e.g., 'np.sqrt').
    - dimensional_dict (Dict[str, List[int]]): A dictionary mapping variables to their dimensions.

    Returns:
    - Tuple[bool, List[int]]:
        - True if the formula is dimensionally correct, False otherwise (may come from either invalid postfix expression or incorrect dimensions).
        - The dimensions of the resulting formula as a list, or None if incorrect.
    """
    null_dimension = [0] * config.num_fundamental_units
    stack = []

    def add_dimensions(dim1: List[int], dim2: List[int]) -> List[int]:
        return [a + b for a, b in zip(dim1, dim2)]

    def subtract_dimensions(dim1: List[int], dim2: List[int]) -> List[int]:
        return [a - b for a, b in zip(dim1, dim2)]

    def divide_dimension_by_scalar(dim: List[int], scalar: int) -> List[float]:
        return [a / scalar for a in dim]

    if not postfix_as_a_list:
        if debug:
            print("Empty postfix expression.")
        return False, None

    for token in postfix_as_a_list:
        if 'np.' in token:
            raise ValueError(
                f"Function {token} is not allowed in the postfix expression. Use math-style names instead.")
            return False, f"Unsupported numpy function in formula: {token}"

    for i, token in enumerate(postfix_as_a_list):
        if token in dimensional_dict:
            stack.append(dimensional_dict[token])

        elif token in config.all_function_list:
            if token == 'sqrt':
                if not stack:
                    if debug:
                        print("Stack is empty.")
                    return False, None
                dim1 = stack.pop()
                stack.append(divide_dimension_by_scalar(dim1, 2))
            else:
                if not stack:
                    if debug:
                        print("Stack is empty.")
                    return False, None
                dim1 = stack.pop()
                if dim1 != null_dimension:
                    if debug:
                        print("Dimension is not null. for function", token)
                    return False, None
                stack.append(null_dimension)

        elif token in {'*', '/'}:
            if len(stack) < 2:
                if debug:
                    print("Stack has less than 2 elements. * /")
                return False, None
            dim2 = stack.pop()
            dim1 = stack.pop()
            result_dim = (
                add_dimensions(dim1, dim2) if token == '*' else subtract_dimensions(dim1, dim2)
            )
            stack.append(result_dim)

        elif token in {'+', '-'}:
            if len(stack) < 2:
                if debug:
                    print("Stack has less than 2 elements. + -")
                return False, None
            dim2 = stack.pop()
            dim1 = stack.pop()
            if dim1 != dim2:
                if debug:
                    print("Dimensions are not equal. + -")
                return False, None
            stack.append(dim1)

        elif token == '**':
            if len(stack) < 2:
                if debug:
                    print("Stack has less than 2 elements. **")
                return False, None
            dim2 = stack.pop()  # Exponent
            dim1 = stack.pop()  # Base
            if dim2 != null_dimension:
                if debug:
                    print("Exponent is not null dim.")
                return False, None
            try:
                if dim1 == null_dimension:
                    stack.append(null_dimension)
                else:
                    scalar = float(postfix_as_a_list[i - 1])
                    stack.append([a * scalar for a in dim1])
            except ValueError:
                if debug:
                    print(stack, postfix_as_a_list)
                    print("Exponent is not a scalar.", dim1, dim2, postfix_as_a_list[i - 1])
                return False, None

        else:
            if debug:
                print("Unknown token.", token)
            stack.append(null_dimension)

    if len(stack) != 1:
        if debug:
            print("Stack has more than 1 element.")
        return False, None

    resulting_dimension = stack.pop()
    return True, [int(x) for x in resulting_dimension]


class Dimensional_Analysis:
    """
       A class to perform dimensional analysis for a set of variables and equations.

       Attributes:
       - internal_variables_names (list): List of internal variable names used in the analysis.
       - true_name_variables (list): The original names of the variables.
       - dimension_of_variables (dict): Dictionary mapping variables in internal names to their dimensions.
       - dimension_list (list): List of dimensions of the variables.
       - verbose_debug (bool): Flag to enable detailed debug output.

       Methods:
       - analyse_dimensions: Main method to analyze dimensions and determine eigenvectors of the dimensional equation.
       - is_everything_dimensionless: Check if all dimensions in the system are zero (dimensionless).
       - define_matrix: Build the matrix representing the dimensional system.
       - remove_unused_dim: Remove unused dimensions from the target and variable dimensions.
       - solve_dimensional_system: Solve the linear system to find eigenvectors of the dimensional system.
       - analyse_solution: Analyze the solutions of the dimensional system.
       - get_adimensional_variables: Generate adimensional variables based on eigenvectors.
   """

    def __init__(self,
                 internal_variables_names: list,
                 true_name_variables: list,
                 dimension_of_variables: dict,
                 dimension_list: list,
                 verbose_debug=False) -> None:

        self.null_dimension = [0] * config.num_fundamental_units
        self.num_fundamental_dim = config.num_fundamental_units
        self.dimension_of_variables = dimension_of_variables
        self.true_name_variables = true_name_variables
        self.internal_variables_names = internal_variables_names
        self.dimension_list = dimension_list
        self.verbose_debug = verbose_debug

    # main method
    def analyse_dimensions(self, target_dimension):
        """
        Main method to analyze the dimensions of a target equation.

        Parameters:
        - target_dimension (list): The target dimension to analyze.

        Returns:
        - is_everything_dimensionless (bool): True if all dimensions are zero.
        - is_equation_possible (bool): True if the equation is dimensionally valid.
        - is_equation_trivial (bool): True if the solution is trivial.
        - eigenvectors_with_target (list): Eigenvectors including the target.
        - eigenvectors_no_target (list): Eigenvectors excluding the target.
        """

        self.target_dimension = target_dimension
        is_equation_possible = True
        is_equation_trivial = False
        eigenvectors_with_target, eigenvectors_no_target = None, None
        is_everything_dimensionless = self.is_everything_dimensionless(target_dimension, self.dimension_list)

        if is_everything_dimensionless:
            return is_everything_dimensionless, is_equation_possible, is_equation_trivial, eigenvectors_with_target, eigenvectors_no_target
        else:
            solution = self.solve_dimensional_system()
            if self.verbose_debug:
                print('complete solution', len(solution))
                for x in solution:
                    print(x)

            if not solution:
                is_equation_possible = False
            else:  # returns the eigenvectors including the target and not including target of said system
                # these are the eigenvectors of the nullspace of the matrix M, including or not the first dimension (the target)
                eigenvectors_with_target, eigenvectors_no_target = self.analyse_solution(solution)
                if len(solution) == 1:
                    is_equation_trivial = True

            return is_everything_dimensionless, is_equation_possible, is_equation_trivial, eigenvectors_with_target, eigenvectors_no_target

    def is_everything_dimensionless(self, target_dimension, dimension_list):
        """
        Check if all dimensions (target and variables) are zero.

        Parameters:
        - target_dimension (list): Target dimension.
        - dimension_list (list): List of variable dimensions.

        Returns:
        - bool: True if all dimensions are zero, False otherwise.
        """
        return all(all(x == 0 for x in dim) for dim in [target_dimension] + dimension_list)

    def define_matrix(self, all_dims, reduced_dim_number):
        """
        Construct the matrix (M) associated with the dimensional system.

        Parameters:
        - all_dims (list): List of all dimensions (target + variables).
        - reduced_dim_number (int): Number of reduced dimensions.

        Returns:
        - Matrix: The constructed matrix.
        """

        n = len(all_dims)
        M = [[0] * n for _ in range(reduced_dim_number)]
        for j in range(n):
            for i in range(reduced_dim_number):
                M[i][j] = all_dims[j][i]
        return Matrix(M)

    def remove_unused_dim(self):
        """
        Remove dimensions that are not used in the target or variables.

        Returns:
        - reduced_target_dimension (list): Reduced target dimensions.
        - reduced_dimension_list (list): Reduced variable dimensions.
        """
        used_dims = set()
        for i in range(self.num_fundamental_dim):
            if self.target_dimension[i] != 0 or any(dim[i] != 0 for dim in self.dimension_list):
                used_dims.add(i)
        reduced_target_dimension = [self.target_dimension[i] for i in used_dims]
        reduced_dimension_list = [[dim[i] for i in used_dims] for dim in self.dimension_list]
        return reduced_target_dimension, reduced_dimension_list

    def solve_dimensional_system(self):
        """
        Solve the system of equations.

        Returns:
        - list: Nullspace solutions of the dimensional matrix.
        """
        reduced_target_dimension, reduced_dimension_list = self.remove_unused_dim()
        reduced_dim_number = len(reduced_target_dimension)

        all_dims = [reduced_target_dimension] + reduced_dimension_list
        M = self.define_matrix(all_dims, reduced_dim_number)
        return M.nullspace()

    def analyse_solution(self, solutions):
        """
            Analyze the solutions of the dimensional system.

            Parameters:
            - solutions (list): List of solutions (eigenvectors) from the nullspace.

            Returns:
            - eigenvectors_with_target (list): Eigenvectors where the target dimension has a non-zero coefficient.
            - eigenvectors_no_target (list): Eigenvectors where the target dimension has a zero coefficient
                                             or derived by subtracting scaled target eigenvectors.
        """

        eigenvectors_no_target = []
        eigenvectors_with_target = []

        for solution in solutions:
            eigenvector = np.array(solution).astype(float)
            if eigenvector[0] == 0:  # Target coefficient is zero: does not depend on the target dimension
                eigenvectors_no_target.append(eigenvector)
            else:
                # Target coefficient is non-zero: normalize by dividing by the target coefficient
                normalized_eigenvector = eigenvector / eigenvector[0]
                eigenvectors_with_target.append(normalized_eigenvector)

        # eg target is f(x,y) = x, with dim f = dim x = dim y = L, then you have two eigenvalues_with_target describing f/x and f/y
        # eigenvectors_with_target [array([[ 1.],
        #        [-1.],
        #        [-0.]]), array([[ 1.],
        #        [-0.],
        #        [-1.]])]

        # So, refine eigenvectors without the target dimension if multiple target eigenvectors exist in order to
        # find corresponding eigenvectors (= adimensional variables)
        if len(eigenvectors_with_target) >= 2:
            reference_vector = eigenvectors_with_target[0]
            for i in range(1, len(eigenvectors_with_target)):
                adjusted_vector = reference_vector - eigenvectors_with_target[i]
                eigenvectors_no_target.append(adjusted_vector)

        eigenvectors_with_target = [array.flatten() for array in eigenvectors_with_target]
        eigenvectors_no_target = [array.flatten() for array in eigenvectors_no_target]
        if self.verbose_debug:
            print('eigenvectors_with_target', eigenvectors_with_target)
            print('eigenvectors_no_target', eigenvectors_no_target)

        # in this example case, we would get eigenvectors_with_target = [array([ 1., -1., -0.]), array([ 1., -0., -1.])],
        # eigenvectors_no_target = [array([ 0., -1.,  1.])]
        return eigenvectors_with_target, eigenvectors_no_target

    def get_adimensional_variables(self, eigenvectors_no_target, add_products=False, add_inverses=False):
        """
        Generate adimensional variables based on eigenvectors without the target dimension.

        Parameters:
        - eigenvectors_no_target (list): List of eigenvectors with power 0 for the target dimension.
          Example: [[0, 1, -1]] represents variables like 'x0/x1'.
        - add_products (bool): Whether to include products of adimensional variables themselves (default: False).
        - add_inverses (bool): Whether to include inverses of adimensional variables (default: False).

        Returns:
        - list: List of unique adimensional variables.
        """
        # remove the zero target coefficient from the eigenvectors
        eigenvectors_no_target = [vector[1:] for vector in eigenvectors_no_target]
        adim_variables = []
        adim_inverses = []
        variable_names = self.internal_variables_names
        for vector in eigenvectors_no_target:
            if np.sum(np.abs(vector)) == 1:
                continue  # already taken into account in the original variables (these are dimensionless variables already present from the start)
            adim_variable = convert_eigenvector_to_infix_formula(vector, variable_names)
            adim_variables.append(adim_variable)
            if add_inverses:
                adim_inverse = convert_eigenvector_to_infix_formula(-np.array(vector), variable_names)
                adim_inverses.append(adim_inverse)

        product_variables = []
        if add_products:
            for i, var1 in enumerate(adim_variables):
                for var2 in adim_variables[i + 1:]:
                    product_variables.append(f"{var1}*{var2}")
                    if add_inverses:
                        inverse1 = adim_inverses[i] if i < len(adim_inverses) else None
                        inverse2 = adim_inverses[i + 1] if i + 1 < len(adim_inverses) else None
                        if inverse1 and inverse2:
                            product_variables.append(f"{inverse1}*{inverse2}")

        final_variables = adim_variables + adim_inverses + product_variables
        unique_variables = list(set(final_variables) - set(variable_names))
        return unique_variables


def group_variables_by_dimension(dimension_list: List, internal_variables_names: List[str]) -> Dict[Tuple, List[str]]:
    """
    Group variables by their dimensions.

    Args:
        dimension_list (list): A list where each element is a list or tuple
                               representing the dimensions of a variable.
        internal_variables_names (list): A list of variable names corresponding
                                          to the dimensions in `dimension_list`.

    Returns:
        dict: A dictionary where keys are tuples representing dimensions and
              values are lists of variable names with those dimensions.
    """
    dimension_groups = defaultdict(list)
    for var_name, dimension in zip(internal_variables_names, dimension_list):
        dimension_tuple = tuple(dimension)
        dimension_groups[dimension_tuple].append(var_name)
    return dimension_groups


def all_pairs(lst: List[Any]) -> Generator[List[Tuple[Any, Any]], None, None]:
    """
    Generate all unique pairings of elements in the input list.

    Args:
        lst (List[Any]): A list of elements to be paired. The list must have an even number of elements
                         for proper pairing.

    Yields:
        List[Tuple[Any, Any]]: A list of tuples representing one unique pairing of the input elements.

    Example:
        list(all_pairs([1, 2, 3, 4]))
        [[(1, 2), (3, 4)], [(1, 3), (2, 4)], [(1, 4), (2, 3)]]
    """
    if len(lst) < 2:
        yield []
        return

    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in all_pairs(lst[:i] + lst[i + 1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1, len(lst)):
            pair = (a, lst[i])
            for rest in all_pairs(lst[1:i] + lst[i + 1:]):
                yield [pair] + rest


def pairing_to_vector(pairings: List[Tuple[str, str]]) -> List[str]:
    """
    Generate a vector for a given list of pairings.

    Args:
        pairings (List[Tuple[str, str]]): A list of tuples representing the pairs. example [('x3', 'x4'), ('x5', 'x6')]

    Returns:
        str: The vector ['(x3 - x4)', '(x5 - x6)']
    """
    vector = []
    for pair in pairings:
        vector.append(f"({pair[0]} - {pair[1]})")
    return vector


def to_scalar_product(vector1, vector2):
    """
    Create a scalar product string from two lists of strings.

    Args:
        vector1 (list): The first vector. example ['a', 'b']
        vector2 (list): The second vector. example ['c', 'd']

    Returns:
        str: A string representing the scalar product in the form
             "(vector1[0]) * (vector2[0]) + (vector1[1]) * (vector2[1]) + ...".
    """
    if len(vector1) != len(vector2):
        raise ValueError("Both vectors must have the same length.")
    if vector1 == vector2:
        return ' + '.join([f'({vector1[i]})**2' for i in range(len(vector1))])
    return ' + '.join([f'({vector1[i]}) * ({vector2[i]})' for i in range(len(vector1))])



def create_ground_truth_dict_for_equation(equation_label: str,
                                          formula: Optional[str],
                                          target_dimension: Optional[list],
                                          n_var: int,
                                          true_name_variables: list[str],
                                          unit_dict: dict,
                                          fundamental_units: list[str],
                                          cfg: dict,
                                          verbose: bool) -> dict:
    """
        Create a dictionary with metadata for the given equation, including variable names (internal/true name), dimensions,
        other dimensional analysis, positivity property of the target, and adimensional combinations and further specific
        norms between variables of same dimension.

        Args:
            equation_label (str): Unique identifier for the equation.
            formula (str): The formula for the equation in infix notation ; if known, else None
            target_dimension (list): Dimension of the target variable : must be provided if formula is None
            n_var (int): Number of variables in the equation.
            true_name_variables (list[str]): true names of variables as per 'FeynmanEquations.csv'.
            unit_dict (dict): Dictionary of units for variables.
            fundamental_units (list[str]): Fundamental unit symbols (e.g., ['m', 's', 'kg', 'T', 'V']).
            verbose (bool): If True, prints additional debugging information.

        Returns:
            dict: A dictionary with equation metadata and derived properties.
        """
    assert formula is not None or target_dimension is not None, "Either explicit target formula or target_dimension must be provided"
    for function in config.all_function_list:
        for var in true_name_variables:
            if function in var:
                raise ValueError(
                    f"Variable: {var} contains the name: {function} as a substring, and this is not allowed; change variable name")
    # Initialize variable mappings and dimensions
    internal_variables_names = [f'x{i}' for i in range(len(true_name_variables))]
    dimension_list = []
    dimension_of_variables = {}
    internal_name_to_true_name_dict = {}
    true_name_to_internal_name_dict = {}

    for i, var_name in enumerate(true_name_variables):
        if var_name not in unit_dict:
            raise ValueError(f"Variable {var_name} not found in unit_dict for equation {equation_label}")
        else:
            var_dim = [unit_dict[var_name].get(key, 0) for key in fundamental_units]
            dimension_of_variables[f'x{i}'] = var_dim
            dimension_list.append(var_dim)
            internal_name_to_true_name_dict[f'x{i}'] = var_name
            true_name_to_internal_name_dict[var_name] = f'x{i}'

    # Dimensional analysis : compute target dimension if formula is provided
    if formula is None:
        pass
    else:
        postfix_list = infix_to_postfix(formula)  # given here in variables true nam
        dimensional_dict = {internal_name_to_true_name_dict[f'x{i}']: dimension_list[i] for i in range(n_var)}
        success, target_dimension = compute_dimension_of_postfix_list(postfix_list, dimensional_dict)
        if not success:
            raise ValueError(
                f"Problem with dimensional analysis of {equation_label}: {formula} does not seem to be dimensionally consistent"
                f"computed target dimension is {target_dimension}")

    # Further dimensional analysis : solve the dimension equation
    dimensional_motor = Dimensional_Analysis(internal_variables_names,
                                             true_name_variables,
                                             dimension_of_variables,
                                             dimension_list,
                                             False)
    is_everything_dimensionless, is_equation_possible, is_equation_trivial, eigenvectors_with_target, eigenvectors_no_target \
        = (dimensional_motor.analyse_dimensions(target_dimension))

    # set up flags for target positivity/negativity (will be computed later in preprocessing)
    is_target_everywhere_positive = None
    apply_global_minus_one = None

    if verbose:
        print(f"{equation_label} is always {'positive' if is_target_everywhere_positive else 'negative'}"
              if is_target_everywhere_positive or apply_global_minus_one
              else f"{equation_label} is not always positive or negative")

    # Adimensional variables
    add_inverse = cfg[
        'add_inverse']  # add the inverse of the adimensional variables eg. if y0 = x0/x1 is adim, then also add y1 = x1/x0 to the voc
    add_products = cfg[
        'add_products']  # add the products of the adimensional variables eg. if y0 = x0/x1 and y1 = x2/x3 are adim, then also add y2 = y0*y1 to the voc
    if cfg['add_adim_variables']:
        adimensional_combinations = dimensional_motor.get_adimensional_variables(eigenvectors_no_target, add_products,
                                                                                 add_inverse) if eigenvectors_no_target else []
    else:
        adimensional_combinations = []
    n_new_adim = len(adimensional_combinations)
    adimensional_variables = [f'y{i}' for i in range(n_new_adim)]
    adimensional_dict = {f'y{i}': adimensional_combinations[i] for i in range(n_new_adim)}

    # add "norms" (sqrt of scalar-like products)
    usual_norms = []
    special_norms = []
    grouped_variables = group_variables_by_dimension(dimension_list, internal_variables_names)
    all_dims = []

    for dim, vars_ in grouped_variables.items():
        # this forms a new variable = x1^2 + x2^2 + x3^2... iff all xi have the same (non zero) dimension
        # you may also add its sqrt (not included : default : only the sum of squares)
        if len(vars_) > 1 and np.sum(np.abs(dim)) != 0:
            usual_norms.append(f'{to_scalar_product(vars_, vars_)}')
            all_dims.append([2*x for x in dim])

        # if you have two vars of the same dim (and only two), you may want to add x**2 - y**2
        if len(vars_) == 2 and np.sum(np.abs(dim)) != 0:
            squared_diff = f'{vars_[0]}**2 - {vars_[1]}**2'
            special_norms.append(squared_diff)
            all_dims.append([2 * x for x in dim])

        # this forms a new variable = np.sqrt(sum (xi - xj)^2) iff (len x >=3 and if all x have the same non-zero dimension), for all pairs of the group
        if len(vars_) > 2 and np.sum(np.abs(dim)) != 0:
            pairings = list(all_pairs(vars_))
            for pairing in pairings:
                vector = pairing_to_vector(pairing)
                scalar_product = f'{to_scalar_product(vector, vector)}'
                special_norms.append(scalar_product)
                all_dims.append([2 * x for x in dim])

    norms_list = usual_norms + special_norms
    if cfg['add_norms']:
        norms = {f'norm{i}': norms_list[i] for i in range(len(norms_list))}
        norms_dim = {f'norm{i}': all_dims[i] for i in range(len(all_dims))}
    else:
        norms = {}
        norms_dim = {}

    # return the dictionary
    equation_dict = {'formula': formula,
                     'n_variables': n_var,
                     'variables_true_name': true_name_variables,
                     'variables_internal_name': internal_variables_names,
                     'dimension_list': dimension_list,
                     "dimension_of_variables": dimension_of_variables,
                     'internal_name_to_true_name_dict': internal_name_to_true_name_dict,
                     'true_name_to_internal_name_dict': true_name_to_internal_name_dict,
                     'target_dimension': target_dimension,
                     'is_target_everywhere_positive': is_target_everywhere_positive,
                     'is_everything_dimensionless': is_everything_dimensionless,
                     'is_equation_possible': is_equation_possible,
                     'is_equation_trivial': is_equation_trivial,
                     'eigenvectors_with_target': eigenvectors_with_target,
                     'eigenvectors_no_target': eigenvectors_no_target,
                     'apply_global_minus_one': apply_global_minus_one,
                     'adimensional_variables': adimensional_variables,
                     'adimensional_dict': adimensional_dict,
                     'norms': norms,
                     'norms_dim': norms_dim,
                     }

    return equation_dict


def Build_Feynmann_Formula_Dict(
        dataset_name: str,
        fundamental_units: List[str],
        cfg: dict,
        verbose: int = 0
) -> Dict[str, dict]:
    """
    Create a ground truth dictionary from FeynmanEquations.csv.
    Saves the dictionary to a file `formula_dict_with_units` with the format:
        formula_dict[equation_label] =
                    {'formula': formula,
                     'n_variables': n_var,
                     'variables_true_name': true_name_variables,
                     'variables_internal_name': internal_variables_names,
                     'dimension_list': dimension_list,
                     "dimension_of_variables": dimension_of_variables,
                     'internal_name_to_true_name_dict': internal_name_to_true_name_dict,
                     'true_name_to_internal_name_dict': true_name_to_internal_name_dict,
                     'target_dimension': target_dimension,
                     'is_target_everywhere_positive': is_target_everywhere_positive,
                     'is_everything_dimensionless': is_everything_dimensionless,
                     'is_equation_possible': is_equation_possible,
                     'is_equation_trivial': is_equation_trivial,
                     'eigenvectors_with_target': eigenvectors_with_target,
                     'eigenvectors_no_target': eigenvectors_no_target,
                     'apply_global_minus_one': apply_global_minus_one,
                     'adimensional_variables': adimensional_variables,
                     'adimensional_dict': adimensional_dict,
                     'norms': norms,
                     'norms_dim': norms_dim,
                     }
    Args:
        dataset_name (str): Name of the dataset.
        fundamental_units (List[str]): List of fundamental unit symbols.
        verbose (int): Verbosity level (0 for silent, higher values for more output).

    Returns:
        Dict[str, dict]: A dictionary of Feynman equations with metadata.
    """
    not_in_file = []
    formula_dict = {}
    unit_dict = Utils_io.get_unit_dict(dataset_name)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loading_path = os.path.join(root_dir, f'Targets/{dataset_name}/FeynmanEquations_cleaned.csv')
    if not os.path.exists(loading_path):
        #try with
        loading_path = os.path.join(root_dir, f'Targets/{dataset_name}/FeynmanEquations.csv')
        if not os.path.exists(loading_path):
            raise ValueError(f"File 'FeynmanEquations.csv' not found in {root_dir}/Targets/{dataset_name}/")

    save_path = os.path.join(root_dir, f'Targets/{dataset_name}/formula_dict_with_units')
    eq_path = os.path.join(root_dir, f'Targets/{dataset_name}/raw_data')
    with open(loading_path, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0 or not line[0]:
                continue  # Skip header or empty lines
            # Parse the equation details
            equation_label = line[0]  # str
            formula = line[1] if line[1] else None  # str or None
            n_var = eval(line[2])  # Convert to int
            variables_list = [line[k] for k in range(3, len(line), 3) if line[k]]

            # Autocorrect variable count if necessary
            if len(variables_list) != n_var:
                # print(f'bug in description file {equation_label} num var is {n_var} but var names are {variables_list}')
                # print('autocorrecting...')
                n_var = len(variables_list)

            try:
                equation_dict = create_ground_truth_dict_for_equation(
                    equation_label, formula, None, n_var, variables_list,
                    unit_dict, fundamental_units, cfg, verbose)
                formula_dict[equation_label] = equation_dict
            except Exception as e:
                print(f"Error processing {equation_label}: {e}")
            if os.path.exists(f'{eq_path}/{equation_label}'):
                pass
            else:
                not_in_file.append(equation_label)

    with open(save_path, 'wb') as f:
        pickle.dump(formula_dict, f)
    #print('WARNING : no data exists for ', not_in_file, ' in ' + dataset_name)
    return formula_dict


def Build_units_dict(dataset_name) -> dict:
    """
    Create a dictionary of units for all variables based on the description provided in 'units.csv' and save it to a file.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loading_path = os.path.join(root_dir, f'Targets/{dataset_name}/units_cleaned.csv')
    if not os.path.exists(loading_path):
        #try with
        loading_path = os.path.join(root_dir, f'Targets/{dataset_name}/units.csv')
        if not os.path.exists(loading_path):
            raise ValueError(f"File 'units.csv' not found in {root_dir}/Targets/{dataset_name}/")

    target_path = os.path.join(root_dir, f'Targets/{dataset_name}/unit_dict.pkl')
    unit_dict = {}
    try:
        df = pd.read_csv(loading_path, sep=',')
    except Exception as e:
        raise ValueError(f"Error loading data from {loading_path}: {e}")

    # Clean the DataFrame
    df = remove_unknown_cols_in_df(df).dropna()

    # Extract unit data and build dictionary
    try:
        for _, row in df.iterrows():
            #assert 'A' not in row['Variable'], f"Variable {row['Variable']} contains invalid character 'A' that is reserved for scalar placeholder"
            unit_dict[row['Variable']] = {
                'm': int(row['m']),
                's': int(row['s']),
                'kg': int(row['kg']),
                'T': int(row['T']),
                'V': int(row['V']),
            }
    except KeyError as e:
        raise ValueError(f"Missing expected column in 'units.csv': {e}")
    except Exception as e:
        raise ValueError(f"Error creating unit_dict: {e}")

    # Save the dictionary for future use
    with open(target_path, 'wb') as f:
        pickle.dump(unit_dict, f)

    return unit_dict


def remove_all_dimensional_info_in_groud_truth(ground_truth):
    """
    Remove all dimensional information in the ground truth dictionary.
    This sets the target dimension to [0, 0, 0, 0, 0] and the units of the variables to [0, 0, 0, 0, 0]
    This also remove any adimensional variables
    """
    null_dimension = [0] * len(ground_truth['target_dimension'])
    ground_truth['dimension_list'] = [null_dimension] * len(ground_truth['dimension_list'])
    ground_truth['dimension_of_variables'] = {key: null_dimension for key in ground_truth['dimension_of_variables']}
    ground_truth['target_dimension'] = [0] * len(ground_truth['target_dimension'])
    ground_truth['is_everything_dimensionless'] = True
    ground_truth['is_equation_possible'] = True
    ground_truth['is_equation_trivial'] = False
    ground_truth['eigenvectors_with_target'] = []
    ground_truth['eigenvectors_no_target'] = []
    ground_truth['adimensional_variables'] = []
    ground_truth['adimensional_dict'] = {}

    return ground_truth