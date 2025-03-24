import sympy
from sympy import symbols, Number, Rational, Pow
import threading
from Utils.Utils_parsing import process_scalars, count_unique_x_variables, convert_to_internal_names
from Utils.Utils_sympy import contains_trig_or_inv_trig, custom_trig_simplify
from typing import Dict, Any, Tuple, List
from typing import Tuple, Optional, List, Dict
from fractions import Fraction
import numpy as np
from mpmath import mp, mpf, sqrt, pi


def identify_float(x: float, precision: int = 50) -> Optional[List[int]]:
    """
    Find whether a float representation of a number in the form of a product of powers of pi and square roots
    of fractions, i.e., looks for a scalar that can be expressed as:
        x = r * (pi ** power_pi) * sqrt(sqrt_num ** eps_num)
    where power_pi is an integer, sqrt_num is a number, and eps_num, eps_pi are either -1 or 1,r a rational number with a small denominator.

    Args:
        x (float): The number to be analyzed and expressed as a scalar.

    Returns:
        Optional[List[int]]: A list containing [power_pi, sqrt_num, eps_num, eps_pi, rational_approx]
                             if a matching pattern is found, otherwise None.
    """
    # Set high precision for mpmath
    mp.dps = precision  # Set decimal places for precision
    x = mpf(x)  # Convert input to high-precision mpmath float

    def closest_rational(number, max_denominator=100):
        return Fraction(float(number)).limit_denominator(max_denominator)

    ans = None

    # Iterate over possible values for power_pi, sqrt_num, eps_num, and eps_pi
    for power_pi in range(-6, 6):
        for sqrt_num in range(1, 10):
            for eps_num in [-1, 1]:
                for eps_pi in [-1, 1]:
                    if eps_pi == 1:
                        target = x/(pi**power_pi*np.sqrt(sqrt_num**eps_num))
                    else:
                        if power_pi == 0:
                            continue
                        target = x/(pi**(mpf(1)/power_pi)*np.sqrt(sqrt_num**eps_num))

                    # Find the closest rational approximation of the target expression
                    r = closest_rational(target)

                    if abs(target - r) < mpf('1e-9'):
                        ans = [power_pi, sqrt_num, eps_num, eps_pi, r]
                        break
    return ans


def convert_back(power_pi: int, sqrt_num: int, eps_num: int, eps_pi: int, r: float) -> Optional:
    """
    Converts the individual components (power of pi, square root of a number, etc.)
    back into a symbolic expression. The components are assumed to be in the form:
    (pi ** power_pi) * sqrt(sqrt_num ** eps_num) multiplied by a rational number `r`.

    Args:
        power_pi (int): The power of pi. Can be positive, negative, or zero.
        sqrt_num (int): The number inside the square root.
        eps_num (int): Exponent for sqrt_num, either 1 or -1.
        eps_pi (int): Sign for the power of pi, either 1 or -1.
        r (float): The rational coefficient (numerator/denominator).

    Returns:
        sympy.Basic: A symbolic expression involving pi and square roots,
                      or None if an invalid combination is provided (e.g., power_pi == 0 and eps_pi == -1).
    """
    from sympy import pi, sqrt

    sympy_expr = Rational(r)
    if eps_pi == 1:
        if power_pi == 1:
            sympy_expr *= pi
        else:
            sympy_expr *= (pi**power_pi)
    else:
        if power_pi == 0:
            pass
        elif power_pi == 1:
            sympy_expr *= pi
        else:
            sympy_expr *= (pi**(1/power_pi))
    sympy_expr *= sqrt(sqrt_num ** eps_num)
    return sympy_expr

def floats_to_pi_and_fraction_scalars(expr):
    """Replacement motor for sympy expressions to replace floats by specific fractions when applicable,
    as defined by identify_float"""

    def identify_pi_and_fraction_scalars(num):
        ans = identify_float(num)
        if ans is None:
            return num
        else:
            power_pi, sqrt_num, eps_num, eps_pi, r = ans
            sp_num = convert_back(power_pi, sqrt_num, eps_num, eps_pi, r)
            return sympy.simplify(sp_num)

    return expr.replace(
        lambda x: isinstance(x, Number) and x.is_Float,  # Match float numbers
        lambda x: identify_pi_and_fraction_scalars(float(x))
    )

def replace_floats_with_ints(expr):
    """
    Replaces float exponents in power expressions with integers when possible to help sympy simplify.

    This function specifically handles `Pow` expressions where the exponent is a float. If the exponent
    is a whole number (after rounding), it is replaced with an integer. Otherwise, the exponent remains unchanged.

    Args:
        expr (sympy.Expr): The expression to be processed.

    Returns:
        sympy.Expr: The expression with float exponents replaced by integers if applicable.
    """
    if isinstance(expr, Pow) and expr.exp.is_Float:
        new_exp = int(expr.exp) if expr.exp == int(expr.exp) else expr.exp
        return Pow(expr.base, new_exp)
    # Also globally for floats elsewhere, if they are integers
    return expr.replace(
        lambda x: x.is_Number and x.is_Float,
        lambda x: int(x) if x == int(x) else x
    )

def full_simplify(equation, positive_assumptions):
    """
    Performs a full 'custom simplification' of a sympy expression.

    This function performs multiple rounds of simplification:
    1. Replaces floats with integers where possible in power expressions.
    2. Simplifies the expression based on assumptions.
    3. Looks for potential match of floats to some products of rational, square roots, and powers of pi
    4. Performs trigonometric simplifications if necessary.

    Args:
        equation (sympy.Expr): The equation to be simplified.
        symbols_list (list): The list of symbols to be used in the simplification.
        positive_assumptions (list): A list of assumptions that the symbols are positive.

    Returns:
        Tuple[sympy.Expr, sympy.Expr]: The simplified expressions after full simplification.
    """

    # first round of simplification
    equation = replace_floats_with_ints(equation)
    simplified = sympy.simplify(equation, assumptions=positive_assumptions)
    # second round of simplification
    simplified = floats_to_pi_and_fraction_scalars(simplified)
    simplified = replace_floats_with_ints(simplified)
    simplified = sympy.simplify(simplified, assumptions=positive_assumptions)

    # finally reduce to trig simplifications : gives the expression via custom_trig_simplify
    if contains_trig_or_inv_trig(simplified):
        final_expr1 = custom_trig_simplify(simplified)  # eg cos(asin(x)) -> sqrt(1-x^2)
    else:
        final_expr1 = simplified
    final_expr2 = sympy.simplify(simplified, assumptions=positive_assumptions)  # it may simplify back exp to cosh, so lets keep both
    return final_expr1, final_expr2


def prepare_formula(formula, scalars, norms, adimensional_dict):
    # cast the formula in internal names like x0, x1, x2, etc
    formula = convert_to_internal_names(formula, norms, adimensional_dict)

    # replace scalars A[0], A[1], etc by the actual values
    formula = process_scalars(formula, scalars)

    # misc string replacements for sympy compatibility
    formula = formula.replace('np.', '')

    # mismatch between numpy and sympy
    trig_rename = {
        'arcsin': 'asin',
        'arccos': 'acos',
        'arctan': 'atan',
        'arcsinh': 'asinh',
        'arccosh': 'acosh',
        'arctanh': 'atanh',
    }

    for old, new in trig_rename.items():
        formula = formula.replace(old, new)

    formula = formula.replace("(0)", "0")
    formula = formula.replace('**', '^')

    return formula, trig_rename


def compare_with_ground_truth(expr, ground_truth, locals, assumptions, trig_rename):
    """
    Compares a given expression with the ground truth expression to check if they are equal.

    This function processes the ground truth formula, applies any necessary symbol replacements
    (for trigonometric functions), and simplifies the difference between the given expression
    and the ground truth expression to check for equality.

    Args:
        expr (Any): The sympy expression to compare against the ground truth.
        ground_truth (Dict[str, Any]): A dictionary containing the ground truth formula (in 'formula') and other metadata.
        locals (Dict[str, Any]): A dictionary of local variables for sympy expressions.
        assumptions (list): A list of assumptions to be used during simplification (e.g., symbols > 0).
        trig_rename (Dict[str, str]): A mapping of trigonometric function names that need to be replaced.

    Returns:
        bool: `True` if the given expression matches the ground truth, `False` otherwise.
    """

    gt_formula = ground_truth['formula']
    for old, new in trig_rename.items():
        gt_formula = gt_formula.replace(old, new)

    gt_expr = sympy.sympify(gt_formula, locals=locals)
    diff = sympy.simplify(expr - gt_expr, assumptions=assumptions)

    return diff == 0

def map_to_internal_names(final_expr1: str, final_expr2: str, mapping: Dict[str, str]) -> List[str]:
    """
    Maps the symbols in the given expressions to new symbols based on the provided mapping.

    This function substitutes each symbol in `final_expr1` and `final_expr2` according to the mapping dictionary,
    and returns the modified expressions.

    Args:
        final_expr1 (str): The first mathematical expression to be processed.
        final_expr2 (str): The second mathematical expression to be processed.
        mapping (Dict[str, str]): A dictionary that maps original symbol names to new symbol names.

    Returns:
        List[Any]: A list containing the modified expressions, with symbols replaced according to the mapping.
    """
    # Create a mapping from old (sympy) symbols to new (sympy) symbols
    mapped_symbols = {symbols(name): symbols(new_name) for name, new_name in mapping.items()}

    # Create a placeholder mapping to handle symbol substitution
    placeholder_mapping = {k: symbols(f"__tmp_{k}__") for k in mapped_symbols.keys()}

    expresions = []
    for final_expr in [final_expr1, final_expr2]:
        expr = sympy.sympify(final_expr, locals=locals)

        # Substitute the original symbols with placeholders to avoid conflicts
        for original, placeholder in placeholder_mapping.items():
            expr = expr.subs(original, placeholder)
        # Substitute the placeholders with the mapped symbols
        for placeholder, target in placeholder_mapping.items():
            expr = expr.subs(target, mapped_symbols[placeholder])

        expresions.append(expr)
    return expresions

def to_sympy(formula: str, scalars: List[float], ground_truth: Dict[str, Any], timeout: int = 20) -> Tuple[bool, bool, Any, Any]:
    """
    Converts a formula to a SymPy expression, simplifies it, and compares it with the given ground truth formula.

    This function performs the following steps:
    1. Prepares the formula by renaming and handling variables.
    2. Defines symbolic representations for the variables.
    3. Simplifies the formula using SymPy's simplification tools.
    4. Compares the simplified expression with the provided ground truth formula.

    Args:
        formula (str): The mathematical formula to be processed.
        scalars (Dict[str, Any]): A dictionary of scalar values or parameters used in the formula.
        ground_truth (Dict[str, Any]): A dictionary containing the ground truth formula and other metadata like 'norms',
                                       'adimensional_dict', and 'internal_name_to_true_name_dict'.
        timeout (int, optional): A timeout value (in seconds) for the whole operation. Default is 20 seconds.
        (Sympy may be stuck in some cases)

    Returns:
        Tuple[bool, bool,Any, Any]: A tuple where:
            - The first element is a boolean indicating if a timeout occured.
            - The second element is a boolean indicating if a match with the ground truth was found.
            - The third element is the simplified expression or None if no match is found.
            - The fourth element is the original expression before simplification.
    """


    result = [None]

    def worker(formula, scalars, ground_truth):
        """
        Helper function to process the formula and compare it with the ground truth.

        Args:
            formula (str): The mathematical formula to be processed.
            scalars (Dict[str, Any]): The scalar values or parameters used in the formula.
            ground_truth (Dict[str, Any]): The ground truth dictionary containing the expected formula and other metadata.
        """
        try:
            # Extract norms and adimensional dictionary from the ground truth
            norms = ground_truth['norms']
            adimensional_dict = ground_truth['adimensional_dict']

            # Define a locals dictionary to handle possible conflicts with SymPy functions (e.g., beta, gamma)
            locals = {}
            for confusing_symbol in ['beta', 'gamma']:
                if confusing_symbol in ground_truth['internal_name_to_true_name_dict'].values():
                    locals.update({confusing_symbol: symbols(confusing_symbol)})

            # Prepare the formula for SymPy
            formula, trig_rename = prepare_formula(formula, scalars, norms, adimensional_dict)
            # Count unique variables and create corresponding symbols in SymPy
            n_x = count_unique_x_variables(formula)
            symbol_names = [f'x{i}' for i in range(n_x)]
            symbols_list = symbols(' '.join(symbol_names), real=True, positive=True)
            if n_x == 1:
                symbols_list = (symbols_list,)  # handle shape issue when only one symbol
            positive_assumptions = [(symbol > 0) for symbol in symbols_list]

            # Convert the formula to a SymPy expression and simplify it
            equation = sympy.sympify(formula, locals=locals)
            expressions = full_simplify(equation, positive_assumptions)
            tmp = expressions[0]
            expressions = map_to_internal_names(*expressions, ground_truth['internal_name_to_true_name_dict'])

            # Process the ground truth formula for comparison
            gt = ground_truth['formula']
            for old, new in trig_rename.items():
                gt = gt.replace(old, new)

            # Compare simplified expressions with the ground truth
            for expr in expressions:
                if compare_with_ground_truth(expr, ground_truth, locals, positive_assumptions, trig_rename):
                    result[:] = [True, expr, tmp]
                    return

            # If no match found, return the first expression and indicate no match
            result[:] = [False, expressions[0], tmp]
        except Exception as e:
            print(f"Exception occurred in worker: {e}")
            result[:] =  [False, None, None]


    thread = threading.Thread(target=worker, args=(formula, scalars, ground_truth))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Timeout occurred!")
        return (True, False, None, None)  # Timeout flag as True and no result

    return (False, *result)

def to_sympy_without_simplification(formula, scalars, ground_truth):
    """Same as to_sympy but without simplification and thus without the timeout"""
    norms = ground_truth['norms']
    adimensional_dict = ground_truth['adimensional_dict']

    # Define a locals dictionary to handle possible conflicts with SymPy functions (e.g., beta, gamma)
    locals = {}
    for confusing_symbol in ['beta', 'gamma']:
        if confusing_symbol in ground_truth['internal_name_to_true_name_dict'].values():
            locals.update({confusing_symbol: symbols(confusing_symbol)})

    formula, trig_rename = prepare_formula(formula, scalars, norms, adimensional_dict)

    n_x = count_unique_x_variables(formula)
    symbol_names = [f'x{i}' for i in range(n_x)]
    symbols_list = symbols(' '.join(symbol_names), real=True,positive=True)
    if n_x == 1:
        symbols_list = (symbols_list,)  # else shape issue

    # convert to sympy expression and full simplify
    equation = sympy.sympify(formula, locals=locals)
    expressions = map_to_internal_names(equation, equation, ground_truth['internal_name_to_true_name_dict'])

    return expressions[0]
