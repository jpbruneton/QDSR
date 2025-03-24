import re
from config import config
import shunting_yard as sy

def infix_to_postfix(formula : str) -> list:
    """
    Convert an infix formula to a postfix formula, using the shunting yard algorithm.
    Note that the sy module uses '^' for power, so we replace '**' with '^' before calling the shunting yard algorithm.
    """
    expression = formula.replace('**', '^')
    postfix_list = sy.shunting_yard(expression).split()
    postfix_list = [x.replace('^', '**') for x in postfix_list]
    return postfix_list

def count_unique_x_variables(formula):
    x_variables = re.findall(r'x\d+', formula)
    unique_x_variables = set(x_variables)
    return len(unique_x_variables)


def process_scalars(formula, scalars):
    """
    Assign scalar numbers to 'A' in the formula and replace them with provided scalar values.

    Args:
        formula (str): The input formula containing 'A' placeholders.
        scalars (list): A list of scalar values to replace 'A[i]' with.

    Returns:
        str: The formula with scalar placeholders replaced by their values.
    """
    A_count = 0
    neweq = ''

    # Assign scalar numbers
    for char in formula:
        if char == 'A':
            neweq += f'A[{A_count}]'
            A_count += 1
        else:
            neweq += char

    # Replace scalars
    for k in range(len(scalars)):
        neweq = neweq.replace(f'A[{k}]', str(scalars[k]))

    return neweq

def convert_to_internal_names_and_numpy_representation(formula, true_name_to_internal_name_dict):
    """
    Convert the true names of variables and functions to their internal names in the formula.
    :param formula: like 'x + exp(y)' # and not 'x0 + pi*np.exp(x1)'
    :param true_name_to_internal_name_dict: like {'x': 'x0', 'y': 'x1'}
    :return: the derived formula with internal names (like 'x0 + np.pi*np.exp(x1)')
    It uses placeholders to avoid collisions during replacements, and also convert the functions to numpy functions
    """
    # to avoid collisions, we first replace the true names by placeholders, then we replace the placeholders by the internal names
    # but we need to sort the keys by length to avoid replacing 'x' by 'x0' before 'x0' by 'x1' etc
    # also, collisions might occur with x in np.exp( ... ) so we need to replace the functions first

    functions = config.all_function_list

    sorted_functions = sorted(functions, key=lambda x: len(x), reverse=True)
    for i, func in enumerate(sorted_functions):
        formula = formula.replace(func, f'__{i}__') #placeholder for functions
    # same for true variables names : our internal names are x0, x1, ... but some of the true variables names are also x0 etc.
    keys = list(true_name_to_internal_name_dict.keys())
    sorted_keys = sorted(keys, key=lambda x: len(x), reverse=True)
    for i, key in enumerate(sorted_keys):
        formula = formula.replace(key, f'@{i}') # placeholder for true names
    # then replace the placeholders
    for i, key in enumerate(sorted_keys):
        formula = formula.replace(f'@{i}', true_name_to_internal_name_dict[key])
    for i, func in enumerate(sorted_functions):
        formula = formula.replace(f'__{i}__', 'np.' + func)

    # special case for np.log ; pi
    formula = formula.replace('np.ln', 'np.log')  # ln is not a function in numpy
    formula = formula.replace('pi', 'np.pi')  # pi is not a variable in numpy

    return formula

def convert_to_internal_names(formula, norms, adimensional_dict):
    n_norms = len(norms)
    if n_norms:
        for k in reversed(range(n_norms)):
            formula = formula.replace(f'norm{k}', norms[f'norm{k}'])
    n_adim = len(adimensional_dict)
    if n_adim:
        for k in reversed(range(n_adim)):
            formula = formula.replace(f'y{k}', adimensional_dict[f'y{k}'])
    return formula