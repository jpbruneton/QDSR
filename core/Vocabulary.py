from copy import deepcopy

class Vocabulary():
    """
        A class that represents the vocabulary of an equation, including symbols, functions,
        variables, and their corresponding dimensions.

        This class is responsible for defining various categories of symbols such as scalars,
        functions, variables, and operations, based on the provided configuration. It also
        stores the dimensional information of these symbols.

        Methods:
            build_vocabulary(cfg, ground_truth, use_original_variables=True,
                             use_adim_variables=True, use_norms=True):
                Builds the vocabulary based on the provided configuration, scalar type, and ground truth.
            __repr__(): Returns a string representation of the Vocabulary object.
    """

    def __init__(self) -> None:
        pass

    def build_vocabulary(self,
                         cfg: dict,
                         ground_truth: dict,
                         use_original_variables = True,
                         use_adim_variables = True,
                         use_norms = True,
                         ) -> None:
        """
         Builds the vocabulary based on the provided configuration, scalar type, and ground truth.

         This method sets up symbols (such as dimensionless scalars, variables, functions,
         and operations), their corresponding dimensions, and stores them as attributes of
         the Vocabulary class.

         Parameters:
             cfg (dict): Configuration dictionary containing settings such as function list, scalar list, etc.
             ground_truth (dict): Ground truth data containing information on variables, dimensions, etc.
             use_original_variables (bool): Flag to include original variables from ground_truth (default: True).
             use_adim_variables (bool): Flag to include adimensional variables (default: True).
             use_norms (bool): Flag to include norm-based variables (default: True).

         """

        # Set the scalar type and initialize null dimension
        null_dimension = [0] * len(ground_truth['target_dimension'])

        # Extract configuration values for power, sqrt, and max power
        use_power = cfg['use_power']
        use_sqrt = cfg['use_sqrt']
        max_power = cfg['max_power']


        self.dimensionless_scalars = ['A'] #placeholder for free scalars

        # Initialize empty lists for other symbols
        self.variables = []

        # Add original and adimensional variables to the vocabulary
        if use_original_variables:
            self.variables += deepcopy(ground_truth['variables_internal_name'])
        if use_adim_variables:
            self.variables += ground_truth['adimensional_variables'] #considérer les variables adimensionnelles comme des vars en plus et non des adim scalars

        # Include norms in the vocabulary if specified
        if use_norms:
            self.variables += list(ground_truth['norms_dim'].keys())

        # Define integers for power operations if applicable
        if use_power:
            self.integers_for_power = [str(i) for i in range(-max_power, max_power + 1)]
            self.integers_for_power.remove('0')
        else:
            self.integers_for_power = []

        # Arity 0 symbols (without/with power)
        self.arity_0_symbols_no_power = self.dimensionless_scalars + self.variables
        self.arity_0_symbols_with_power = self.arity_0_symbols_no_power + self.integers_for_power

        # Initialize dimensional dictionary for arity 0 symbols
        self.dimensional_dict = {}
        for symbol in self.dimensionless_scalars:
            self.dimensional_dict[symbol] = null_dimension

        for symbol in self.variables:
            if symbol in ground_truth.get('dimension_of_variables', {}):
                self.dimensional_dict[symbol] = ground_truth['dimension_of_variables'][symbol]
            elif symbol in ground_truth.get('adimensional_variables', {}):
                self.dimensional_dict[symbol] = null_dimension
            elif symbol in ground_truth.get('norms', {}):
                self.dimensional_dict[symbol] = list(ground_truth['norms_dim'][symbol])
        for symbol in self.integers_for_power:
            self.dimensional_dict[symbol] = null_dimension

        # Arity 1 symbols (functions) with optional square root
        self.arity_1_symbols_no_sqrt = cfg.get('function_list', [])
        self.sqrt_function = ['np.sqrt('] if use_sqrt else []
        self.arity_1_symbols_with_sqrt = self.arity_1_symbols_no_sqrt + self.sqrt_function

        # Arity 2 symbols (operations) with optional power
        self.arity_2_no_power = ['+', '-', '*', '/']
        self.power = ['**'] if use_power else []
        self.multiply_div = ['*', '/']
        self.add_sub = ['+', '-']
        self.arity_2_with_power = self.arity_2_no_power + self.power
        self.all_symbols = self.arity_0_symbols_with_power + self.arity_1_symbols_with_sqrt + self.arity_2_with_power

        # get alos all arity 0 of null dim, but not power integers
        self.all_arity_0_null_dim = [x for x in self.arity_0_symbols_no_power if self.dimensional_dict[x] == null_dimension]

    def __repr__(self):
        return (f'Vocabulary object with {len(self.all_symbols)} symbols\n'
                f'Arity 0 symbols: {self.arity_0_symbols_with_power}\n'
                f'Arity 1 symbols: {self.arity_1_symbols_with_sqrt}\n'
                f'Arity 2 symbols: {self.arity_2_with_power}\n'
                f'Function list: {self.arity_1_symbols_no_sqrt}\n'
                f'Dimensionless scalars: {self.dimensionless_scalars}\n'
                f'Variables: {self.variables}\n'
                f'Dimensional dict: {self.dimensional_dict}\n')

