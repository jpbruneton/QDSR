from sympy import Basic
from sympy import cos, sin, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, sqrt, exp


######## Trig related ########
def custom_trig_simplify(expr):
    """Simplify expressions involving inverse trigonometric and hyperbolic functions."""

    # Define simplification rules as (condition, replacement) pairs
    rules = [ # standard trigonometric identities
        # cos
        (lambda x: x.func == cos and x.args[0].func == acos,
         lambda x: x.args[0]),
        (lambda x: x.func == cos and x.args[0].func == asin,
         lambda x: sqrt(1 - x.args[0] ** 2)),
        (lambda x: x.func == cos and x.args[0].func == atan,
         lambda x: 1 / sqrt(1 + x.args[0] ** 2)),
        # sin
        (lambda x: x.func == sin and x.args[0].func == asin,
         lambda x: x.args[0]),
        (lambda x: x.func == sin and x.args[0].func == acos,
         lambda x: sqrt(1 - x.args[0] ** 2)),
        (lambda x: x.func == sin and x.args[0].func == atan,
         lambda x: x.args[0] / sqrt(1 + x.args[0] ** 2)),
        # tan
        (lambda x: x.func == tan and x.args[0].func == atan,
         lambda x: x.args[0]),
        (lambda x: x.func == tan and x.args[0].func == asin,
         lambda x: x.args[0] / sqrt(1 - x.args[0] ** 2)),
        (lambda x: x.func == tan and x.args[0].func == acos,
         lambda x: sqrt(1 - x.args[0] ** 2) / x.args[0]),

        # Reverse simplifications
        (lambda x: x.func == asin and x.args[0].func == sin,
         lambda x: x.args[0]),
        (lambda x: x.func == acos and x.args[0].func == cos,
         lambda x: x.args[0]),
        (lambda x: x.func == atan and x.args[0].func == tan,
         lambda x: x.args[0]),

        # hyperbolic trigonometric identities
        (lambda x: x.func == sinh and x.args[0].func == asinh,
            lambda x: x.args[0]),
        (lambda x: x.func == sinh and x.args[0].func == acosh,
         lambda x: sqrt(x.args[0] ** 2 - 1)),
        (lambda x: x.func == sinh and x.args[0].func == atanh,
         lambda x: x.args[0] / sqrt(1 - x.args[0] ** 2)),

        (lambda x: x.func == cosh and x.args[0].func == acosh,
         lambda x: x.args[0]),
        (lambda x: x.func == cosh and x.args[0].func == asinh,
         lambda x: sqrt(1 + x.args[0] ** 2)),
        (lambda x: x.func == cosh and x.args[0].func == atanh,
         lambda x: 1 / sqrt(1 - x.args[0] ** 2)),

        (lambda x: x.func == tanh and x.args[0].func == atanh,
         lambda x: x.args[0]),
        (lambda x: x.func == tanh and x.args[0].func == asinh,
         lambda x: x.args[0] / sqrt(1 + x.args[0] ** 2)),
        (lambda x: x.func == tanh and x.args[0].func == acosh,
         lambda x: sqrt(x.args[0] ** 2 - 1) / x.args[0]),

        # Reverse simplifications
        (lambda x: x.func == asinh and x.args[0].func == sinh,
            lambda x: x.args[0]),
        (lambda x: x.func == acosh and x.args[0].func == cosh,
            lambda x: x.args[0]),
        (lambda x: x.func == atanh and x.args[0].func == tanh,
            lambda x: x.args[0]),

        #else back to exponential
        (lambda x: x.func == cosh,
         lambda x: (exp(x.args[0]) + exp(-x.args[0])) / 2),
        (lambda x: x.func == sinh,
         lambda x: (exp(x.args[0]) - exp(-x.args[0])) / 2),
    ]

    # Apply each rule to the expression
    for condition, replacement in rules:
        expr = expr.replace(condition, replacement)

    return expr

def contains_trig_or_inv_trig(expr):
    # List of functions to check
    trig_functions = [cos, sin, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh]

    # Recursively check if the expression contains any of the specified functions
    if isinstance(expr, Basic):
        if any(func == expr.func for func in trig_functions):
            return True
        # If the expression is an operation (e.g., Add, Mul, etc.), check its arguments
        for arg in expr.args:
            if contains_trig_or_inv_trig(arg):
                return True
    return False