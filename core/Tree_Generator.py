import time
import random
from copy import deepcopy
import numpy as np
from Utils import infix_to_postfix
from core import Tree, post_fix_to_tree, Node
from core import Vocabulary
from config import config
from core import simplify_one_tree
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Any, Dict
import concurrent.futures
from ast import literal_eval  # Safer alternative to eval
from Utils import generate_lists

class Tree_Builder():
    def __init__(self, vocabulary: Vocabulary,
                 target_dimension : list,
                 cfg: dict,
                 hyper_parameters: dict,
                 is_target_everywhere_positive = False
                 ):

        self.n_units = len(target_dimension)
        self.null_dimension = [0] * self.n_units
        self.vocabulary = vocabulary
        self.cfg = cfg
        self.is_target_everywhere_positive = is_target_everywhere_positive
        self.target_dimension = target_dimension

        self.all_dimension_list = [self.vocabulary.dimensional_dict[x] for x in self.vocabulary.variables]

        # unpack hyper parameters
        self.max_number_of_nodes = hyper_parameters['max_number_of_nodes']
        self.inf = hyper_parameters['inf']
        self.sup = hyper_parameters['sup']
        self.proba_mult = hyper_parameters['proba_mult']
        self.proba_div = hyper_parameters['proba_div']
        self.proba_plus = hyper_parameters['proba_plus']
        self.proba_minus = hyper_parameters['proba_minus']
        self.proba_power = hyper_parameters['proba_power']
        self.max_addition_number = hyper_parameters['max_addition_number']

        self.verbose_debug = hyper_parameters['verbose_debug']
        self.arity_probability = hyper_parameters['arity_probability']
        self.use_sqrt = hyper_parameters['use_sqrt']
        self.use_power = hyper_parameters['use_power']
        self.max_power = hyper_parameters['max_power']
        self.root_arity_1_probability = hyper_parameters['proba_arity1_root_node']
        self.root_arity_2_probability = hyper_parameters['proba_arity2_root_node']
        self.favor_in = hyper_parameters['favor_in']
        self.favor_out = hyper_parameters['favor_out']
        self.power_ranges = [x for x in range(-self.max_power, self.max_power+1) if x not in [0, 1]]
        self.proba_scalar_in_exponent = hyper_parameters['proba_scalar_in_exponent']

    def build_dimensions_of_monomials(self):
        """
        Build a dictionary of all possible dimensions of monomials powers of the variables between self.inf and self.sup.
        Parameters:
        Returns:
        dict: Dictionary of all possible dimensions.
        """

        unique_dimensions = [list(x) for x in set(tuple(d) for d in self.all_dimension_list)]
        num_unique_dimension = len(unique_dimensions)

        all_lists = generate_lists(self.inf, self.sup, num_unique_dimension)
        if self.verbose_debug:
            print('all lists:', all_lists)
            print(len(all_lists))

        dimensions_of_monomials = {}

        for lst in all_lists:
            resulting_dim = [0] * self.n_units
            for i in range(num_unique_dimension):
                if lst[i]:
                    resulting_dim = [
                        resulting_dim[k] + lst[i] * unique_dimensions[i][k]
                        for k in range(self.n_units)
                    ]
            dim_str = str(resulting_dim)
            dimensions_of_monomials[dim_str] = dimensions_of_monomials.get(dim_str, 0) + 1

        # also add the square of the dimensions if sqrt is allowed
        if self.use_sqrt:
            for dim_str, _ in list(dimensions_of_monomials.items()):
                dim = literal_eval(dim_str)
                double_dim = [2 * x for x in dim]
                double_dim_str = str(double_dim)
                dimensions_of_monomials[double_dim_str] = dimensions_of_monomials.get(double_dim_str, 0) + 1

        target_dim_str = str(self.target_dimension)
        if target_dim_str not in dimensions_of_monomials:
            dimensions_of_monomials[target_dim_str] = 1

        self.dimensions_of_monomials = dimensions_of_monomials

    def split_dimension(self, parent_dimension, operator):
        """
        Splits a parent dimension into possible child dimensions based on the operator and a given dict of dimensions to draw from (dict of monomials dim).

        Args:
            parent_dimension (List[int]): The dimension of the parent node.
            operator (str): The operator symbol.

        Returns:
            Tuple[List[Tuple[List[int], List[int]]], np.ndarray]: Possible dimension splits and their probabilities.
        """
        possible_splits = []
        # we apply a generalized two sums or two differences algorithm
        if operator == '*':
            for key, _ in self.dimensions_of_monomials.items():
                left_dim = literal_eval(key)
                right_dim = [parent_dimension[i] - left_dim[i] for i in range(self.n_units)]
                if str(right_dim) in self.dimensions_of_monomials:
                    possible_splits.append([left_dim, right_dim])
        elif operator == "/":
            for key, _ in self.dimensions_of_monomials.items():
                left_dim = literal_eval(key)
                right_dim = [left_dim[i] - parent_dimension[i] for i in range(self.n_units)]
                if str(right_dim) in self.dimensions_of_monomials:
                    possible_splits.append((left_dim, right_dim))
        elif operator in ["+", "-"]:
            for key, _ in self.dimensions_of_monomials.items():
                if key == str(parent_dimension):
                    possible_splits.append((literal_eval(key), literal_eval(key)))
        else: # operator is a power ; will only be used if self.use_power is True
            if not self.use_power:
                raise ValueError('Power operator not allowed')
            # two cases : parent is dimensionless or not
            if parent_dimension == self.null_dimension:
                possible_splits.append((self.null_dimension, self.null_dimension))
            else:
                for exponent in self.power_ranges:
                    if all([x % exponent == 0 for x in parent_dimension]):
                        left_dim = [x // exponent for x in parent_dimension]
                        right_dim = self.null_dimension
                        possible_splits.append((left_dim, right_dim))

        # Build proba of splits : favorize splits that are in the list of all possible dimensions
        proba = []
        for left_dim, right_dim in possible_splits:
            if left_dim in self.all_dimension_list or right_dim in self.all_dimension_list:
                proba.append(self.favor_in)
            else:
                proba.append(self.favor_out)

        proba = np.array(proba)/sum(proba)

        return possible_splits, proba

    def can_be_power(self, dimension):
        if not self.use_power:
            return False
        condition1 = dimension == self.null_dimension
        condition2 = False
        for exponent in self.power_ranges:
            if all([x % exponent == 0 for x in dimension]):
                condition2 = True
                break
        return condition1 or condition2

    def get_root_options(self):
        root_options = {'arity1': [], 'arity2': []}

        # Handle arity 1 functions
        if self.target_dimension == self.null_dimension:
            root_options[
                'arity1'] = self.vocabulary.arity_1_symbols_with_sqrt if self.use_sqrt else self.vocabulary.arity_1_symbols_no_sqrt
        elif self.use_sqrt:
            # Allow sqrt if conditions permit or when noise may affect positivity
            if self.cfg['use_noise'] == 0 and self.is_target_everywhere_positive:
                root_options['arity1'] = self.vocabulary.sqrt_function
            else:
                root_options['arity1'] = self.vocabulary.sqrt_function

        # deal with arity 2 now ; check if power is possible
        root_options['arity2'] = self.vocabulary.arity_2_with_power if self.can_be_power(
            self.target_dimension) else self.vocabulary.arity_2_no_power

        return root_options

    def choose_root(self):
        """
          Choose the root node for a tree based on target dimension, configuration, and vocabulary constraints.
          """
        root_options = self.get_root_options()
        if not root_options['arity1']:
            arity = 2  # Default to arity 2 if no arity 1 options are available
        else:
            arity = random.choices(
                [1, 2], weights=[self.root_arity_1_probability, self.root_arity_2_probability], k=1
            )[0]

        # Choose the root symbol
        root_symbol = random.choice(root_options[f'arity{arity}'])
        # Create the root node
        root_node = Node(root_symbol, arity, [], self.target_dimension)
        attributes = root_node.return_node_attributes(self.vocabulary)
        root_node.downward_attributes = attributes
        if self.verbose_debug:
            print('root options', root_options, [self.root_arity_1_probability, self.root_arity_2_probability])
            print('chosen root:', root_node.symbol)

        return root_node

    def choose_arity_of_child(self, child_dimension, parent_arity, nested_functions):
        # we can terminate this branch if child_dimension is in the list of the dimension of variables
        # also, if dimension is null_dimension, we can only have a scalar placeholder A
        # note that second condition here is to avoid having cos(A) for instance
        arity_options = [0] if child_dimension in self.all_dimension_list or (
                child_dimension == self.null_dimension and parent_arity != 1) else []

        # 1 is possible if node.dimension == null_dimension (example cos( ... null_dim_expr... ), or if sqrt in the list of functions
        if (len(self.vocabulary.arity_1_symbols_with_sqrt) and (nested_functions < config.MAX_NESTED_FUNCTIONS)
                and (child_dimension == self.null_dimension or self.use_sqrt)):
                arity_options.append(1)

        arity_options.append(2) # operators are always possible

        arity_probability = [self.arity_probability[i] for i in arity_options]
        arity_probability = [x / sum(arity_probability) for x in arity_probability]
        chosen_arity = random.choices(arity_options, weights = arity_probability, k=1)[0]
        if self.verbose_debug:
            print('arity options, proba, chosen arity', arity_options, arity_probability, chosen_arity)
        return chosen_arity

    def choose_arity_0_symbol(self, guiding_choice):
        child_dimension = guiding_choice['child_dimension']
        exclude_pure_numbers = guiding_choice['exclude_pure_numbers']
        favor_free_scalars = guiding_choice['favor_free_scalars']
        # Check if the parent node's dimension matches any variable's dimension
        if exclude_pure_numbers:
            matching_vars = [k for k, v in self.vocabulary.dimensional_dict.items() if (child_dimension == v)
                             and (k not in self.vocabulary.integers_for_power)
                             and (k not in self.vocabulary.dimensionless_scalars)]
            if self.verbose_debug:
                print('matching vars in arity 0', matching_vars, child_dimension)
            if len(matching_vars):
                return random.choice(matching_vars)
        else:
            #do we want to favor free scalars? #yes but only for exponents
            if favor_free_scalars:
                if random.random() < self.proba_scalar_in_exponent:
                    matching_vars = self.vocabulary.dimensionless_scalars if child_dimension == self.null_dimension else []
                else:
                    matching_vars = [k for k, v in self.vocabulary.dimensional_dict.items() if (child_dimension == v)
                                     and (k not in self.vocabulary.integers_for_power)
                                     and (k not in self.vocabulary.dimensionless_scalars)]
            else:
                matching_vars = [k for k, v in self.vocabulary.dimensional_dict.items() if (child_dimension == v)
                                 and (k not in self.vocabulary.integers_for_power)
                                 ]

            #if we dont have a choice:
            if not matching_vars and child_dimension == self.null_dimension:
                matching_vars = self.vocabulary.dimensionless_scalars
            if self.verbose_debug:
                print('matching vars in arity 0', matching_vars, child_dimension)
            if len(matching_vars):
                return random.choice(matching_vars)

        # else, no matching var found (should never happen)
        raise ValueError('No valid arity 0 symbol found for the given parent node dimension ; should never happen?')

    def choose_arity_1_symbol(self, guiding_choice):
        child_dimension = guiding_choice['child_dimension']
        if child_dimension == self.null_dimension:
            return random.choice(self.vocabulary.arity_1_symbols_with_sqrt)
        elif self.use_sqrt: #only sqrt possible
            return self.vocabulary.sqrt_function[0]
        else:
            raise ValueError('No valid arity 1 symbol found for the given parent node dimension')

    def choose_arity_2_symbol(self, guiding_choice):
        child_dimension = guiding_choice['child_dimension']
        nested_additions = guiding_choice['nested_additions']
        nested_power = guiding_choice['nested_power']

        if nested_additions >= self.max_addition_number:
            possible_operators = deepcopy(self.vocabulary.multiply_div)
            probas = [self.proba_mult, self.proba_div]
        else:
            possible_operators = deepcopy(self.vocabulary.arity_2_no_power)
            probas = [self.proba_plus, self.proba_minus, self.proba_mult, self.proba_div]

        # Check if the node dimension is valid for a power operator
        if self.can_be_power(child_dimension) and nested_power < config.MAX_NESTED_POWERS:
            possible_operators += self.vocabulary.power
            probas.append(self.proba_power)

        probas = [x / sum(probas) for x in probas]
        return random.choices(possible_operators, weights=probas, k=1)[0]

    def update_guiding_choice(self, guiding_choice, side = None):
        #0 means child 0 == left, 1 means child 1 == right
        if guiding_choice['parent_arity'] == 1:
            exclude_pure_numbers = True
            favor_free_scalars = False
        elif guiding_choice['parent_arity'] == 2:
            exclude_pure_numbers = False
            favor_free_scalars = False
            if guiding_choice['parent_symbol'] == '**' and side == 1:
                favor_free_scalars = True
        else:
            raise ValueError('Parent arity should be 1 or 2 here')
        # update
        guiding_choice['exclude_pure_numbers'] = exclude_pure_numbers
        guiding_choice['favor_free_scalars'] = favor_free_scalars
        return guiding_choice

    def generate_one_tree(self):
        """Generate a tree with a maximum number of nodes."""

        if self.verbose_debug:
            print('target dimension', self.target_dimension)
            print('dimensions_of_monomials dict', self.dimensions_of_monomials)
            print('len of', len(self.dimensions_of_monomials))

        # Initialize the root node
        root_node = self.choose_root()
        n_nodes = 1
        all_nodes = [root_node]

        # map arity to the corresponding method
        arity_to_method = {
            0: self.choose_arity_0_symbol,
            1: self.choose_arity_1_symbol,
            2: self.choose_arity_2_symbol
        }
        if self.verbose_debug:
            print('START NODE', all_nodes[0].symbol)

        while n_nodes < self.max_number_of_nodes and all_nodes:

            parent_node = all_nodes.pop(0)  # BFS like
            parent_dimension = parent_node.dimension
            parent_arity = parent_node.arity
            parent_symbol = parent_node.symbol
            nested_functions = parent_node.downward_attributes['functions']
            nested_additions = parent_node.downward_attributes['additions']
            nested_power = parent_node.downward_attributes['powers']
            guiding_choice = {
                'parent_dimension': parent_dimension,
                'parent_arity': parent_arity,
                'parent_symbol': parent_symbol,
                'nested_additions': nested_additions,
                'nested_functions': nested_functions,
                'nested_power': nested_power
            }

            if self.verbose_debug:
                print('EXPANDING', parent_arity, parent_symbol, parent_dimension)

            # we have to build its children if it is not a leaf
            if parent_node.arity == 0:
                pass

            elif parent_node.arity == 1:
                child_dimension = [2 * x for x in parent_dimension] if parent_symbol == 'np.sqrt(' else parent_dimension
                child_arity = self.choose_arity_of_child(child_dimension, parent_arity, nested_functions)
                guiding_choice['child_dimension'] = child_dimension
                guiding_choice = self.update_guiding_choice(guiding_choice)
                child_symbol = arity_to_method[child_arity](guiding_choice)

                if self.verbose_debug:
                    if parent_node.symbol == 'np.sqrt(':
                        print('sqrt', parent_node.dimension, child_dimension, child_arity, child_symbol)

                child_node = Node(child_symbol, child_arity, [], child_dimension)
                child_attributes = child_node.return_node_attributes(self.vocabulary)
                parent_attributes = parent_node.downward_attributes
                child_node.downward_attributes = child_node.add_attribute(parent_attributes, child_attributes)

                if self.verbose_debug:
                    print('child chosen symb and dim', child_node.symbol, child_node.dimension, 'parent symb and dim', parent_node.symbol, parent_node.dimension)
                parent_node.children = [child_node]
                all_nodes.append(child_node)
                n_nodes += 1

            else:  # parent_node.arity == 2 is an operator
                possible_splits, proba = self.split_dimension(parent_node.dimension, parent_node.symbol)
                if not possible_splits:
                    if self.verbose_debug: print('impossible tree')
                    return None, False
                if self.verbose_debug: print('possible splits', possible_splits)
                child_dimensions = random.choices(possible_splits, weights=proba, k=1)[0]
                if self.verbose_debug: print('chosen split', child_dimensions[0], child_dimensions[1])

                if (parent_symbol in self.vocabulary.arity_2_no_power) or (parent_symbol == '**' and child_dimensions[0] == self.null_dimension) :
                    # regular case ; we have expr (operator) expr
                    # valid also for power with dimensionless left child : eg. x**x with x dimensionless
                    # only difference between child_dimensions[0] == self.null_dimension or not,
                    # is that we want more often (null expr)**A than (null expr)**(null expr)
                    chosen_arities = []
                    chosen_symbols = []
                    child_nodes = []

                    for i in range(2):
                        chosen_arities.append(self.choose_arity_of_child(child_dimensions[i], parent_arity, nested_functions))
                        guiding_choice['child_dimension'] = child_dimensions[i]
                        guiding_choice = self.update_guiding_choice(guiding_choice, i)
                        chosen_symbols.append(arity_to_method[chosen_arities[i]](guiding_choice))

                    if self.verbose_debug: print('chosen arities',chosen_arities)
                    if self.verbose_debug: print('chosen symbols', chosen_symbols)

                    for i in range(2):
                        child_nodes.append(Node(chosen_symbols[i], chosen_arities[i], [], child_dimensions[i]))
                        child_attributes = child_nodes[i].return_node_attributes(self.vocabulary)
                        parent_attributes = parent_node.downward_attributes
                        child_nodes[i].downward_attributes = child_nodes[i].add_attribute(parent_attributes, child_attributes)

                else: # we have the case eg target_dim = length**2 , x == length, then form x**2 : right child must be an integer
                    # first retrieve the correct power
                    if self.verbose_debug:
                        assert parent_symbol == '**', 'should never happen'
                        assert child_dimensions[0] != self.null_dimension, 'left shld be dimensioned'
                        assert child_dimensions[1] == self.null_dimension, 'right shd be dimensionless'
                        print('parent symbol', parent_symbol)
                        print('child dims', child_dimensions)
                        print('parent dim', parent_dimension)

                    exponent = 0
                    for i, x in enumerate(child_dimensions[0]):
                        if x != 0:
                            exponent = parent_dimension[i] // x
                            break
                    if self.verbose_debug: print('exponent', exponent)
                    chosen_arities = []
                    chosen_symbols = []
                    child_nodes = []
                    #choose left child
                    chosen_arities.append(self.choose_arity_of_child(child_dimensions[0], parent_arity, nested_functions))
                    guiding_choice['child_dimension'] = child_dimensions[0]
                    guiding_choice = self.update_guiding_choice(guiding_choice, 0)
                    chosen_symbols.append(arity_to_method[chosen_arities[0]](guiding_choice))

                    # right child is fixed
                    chosen_arities.append(0)
                    chosen_symbols.append(str(exponent))

                    # define the nodes
                    for i in range(2):
                        child_nodes.append(Node(chosen_symbols[i], chosen_arities[i], [], child_dimensions[i]))
                        child_attributes = child_nodes[i].return_node_attributes(self.vocabulary)
                        parent_attributes = parent_node.downward_attributes
                        child_nodes[i].downward_attributes = child_nodes[i].add_attribute(parent_attributes, child_attributes)

                parent_node.children = child_nodes
                all_nodes.extend(child_nodes)
                n_nodes += 2
            if self.verbose_debug:
                print('------')

        success = n_nodes < self.max_number_of_nodes
        if self.verbose_debug:
            if n_nodes >= self.max_number_of_nodes:
                print('max number of nodes reached')
            else:
                print('done')

        return root_node, success


def choose_hyper_parameters(max_length, cfg, inf = None, sup = None):
    proba_arity1_root_node = 0.2
    proba_arity1_root_node = proba_arity1_root_node*random.random()
    proba_operators = [1, 1/2, 1/4, 1/4, 1/4]  # proba operator : mult, div, plus, minus, power
    proba_operators = [x*random.random() for x in proba_operators]
    proba_operators /= np.sum(proba_operators)

    proba_arity = [0.5, 0.1, 0.4]  # proba arity 0, 1, 2 # 0.5194174757281553 0.03964401294498382 0.44093851132686085
    proba_arity = [x*random.random() for x in proba_arity]
    proba_arity /= np.sum(proba_arity)

    sup = np.random.choice([1, 2], 1, p=[0.9, 0.1])[0] if sup is None else sup
    usesqrt = random.choice([True, False]) if cfg['use_sqrt'] else False
    usepower = random.choice([True, False]) if cfg['use_power'] else False

    proba_scalar_in_exponent = 0.9
    hyper_parameters = {
        'proba_arity1_root_node': proba_arity1_root_node,
        'proba_arity2_root_node': 1 - proba_arity1_root_node,
        'max_number_of_nodes': max_length,
        'inf': 0 if inf is None else inf,
        'sup': sup,
        'proba_mult': proba_operators[0],
        'proba_div': proba_operators[1],
        'proba_plus': proba_operators[2],
        'proba_minus': proba_operators[3],
        'proba_power': proba_operators[4],
        'max_addition_number': random.randint(1, 8),  # included
        'max_power': cfg['max_power'],  # included,
        'arity_probability': proba_arity,  # 0, 1, 2
        'use_sqrt': usesqrt,
        'use_power': usepower,
        'reduce_to_unique_dim': 1,
        'favor_in': 1,
        'favor_out': 0.1,
        'proba_scalar_in_exponent': proba_scalar_in_exponent,
        'verbose_debug': 0
    }
    return hyper_parameters

def generate_from_scratch(num_trees, target_dimension, vocabulary, cfg, max_length, is_target_everywhere_positive = False):
    total_tries = 0
    trees = []
    trees_postfix_formula = []
    defects = 0
    ts = []

    if num_trees < 1: num_trees = 1

    gentime = []
    tbbuild = []
    while total_tries < num_trees*5 and len(trees) < num_trees:
        st = time.time()
        if total_tries >= 50 and len(trees)==0: #too big fail rate
            break
        hyper_parameters = choose_hyper_parameters(max_length, cfg)
        #if hyper_parameters['verbose_debug']:
        #    print('hyper parameters')
        #    for k,v in hyper_parameters.items():
        #        print(k, v)

        TB = Tree_Builder(vocabulary, target_dimension, cfg, hyper_parameters, is_target_everywhere_positive)
        TB.build_dimensions_of_monomials()
        total_tries += 1
        tbbuild.append(time.time() - st)
        st = time.time()
        root_node, success = TB.generate_one_tree()
        gentime.append(time.time() - st)
        if success:
            new_tree = Tree(root_node, vocabulary)

        if not success:
            defects +=1
            t = time.time() - st
            ts.append(t)
            continue
        elif new_tree.length >= 4 and new_tree.postfix_formula not in trees_postfix_formula:
            trees.append(new_tree)

            trees_postfix_formula.append(new_tree.postfix_formula)
        else:
            defects +=1
        t = time.time() - st
        ts.append(t)
    if 0 : print('gentime', np.mean(gentime), 'tbbuild', np.mean(tbbuild), defects, len(trees), total_tries)

    #fail_rate = 1 - len(trees) / total_tries
    #print(time.time() - p, 'total tries', total_tries, 'defects', defects, 'fail rate', fail_rate, np.mean(ts))
    return trees

def get_num_denom(eigenvector, variables_name):
    indices = [i for i in range(len(eigenvector)) if eigenvector[i] != 0]
    num = []
    den = []

    for i in indices:
        variable = f'({variables_name[i]})'
        count = abs(int(eigenvector[i]))
        if eigenvector[i] > 0:
            num.extend([variable] * count)
        else:
            den.extend([variable] * count)

    numerator_infix = '*'.join(num)
    denominator_infix = '*'.join(den)
    return numerator_infix, denominator_infix


def run_fraction(left_dim, right_dim, TB, new_budget, vocabulary, cfg, max_length,not_working, k, global_vocabulary):
    TB.target_dimension = left_dim
    numerator_trees = generate_from_scratch(new_budget, left_dim, vocabulary, cfg, max_length//2)
    TB.target_dimension = right_dim
    denominator_trees = generate_from_scratch(new_budget, right_dim, vocabulary, cfg, max_length//2)

    local_pool = []
    if len(numerator_trees) == 0 or len(denominator_trees) == 0:
        not_working[k] = 1
        return [], not_working
    while len(local_pool) < new_budget:
        i, j = random.choice(range(len(numerator_trees))), random.choice(range(len(denominator_trees)))
        left_tree = numerator_trees[i]
        right_tree = denominator_trees[j]
        left_pf = left_tree.postfix_formula
        right_pf = right_tree.postfix_formula
        new_pf = left_pf + right_pf + ['/']
        new_tree = post_fix_to_tree(new_pf, global_vocabulary)
        local_pool.append(new_tree)
    return local_pool, not_working

def generate_from_fraction_old(num_trees, ground_truth, vocabulary, cfg, max_length, eigenvector, global_vocabulary):
    if num_trees <1 : num_trees = 1
    variables_name = ground_truth['variables_internal_name']

    numerator, denominator = get_num_denom(eigenvector, variables_name)  #returns like (x0)*(x1), (x3)*(x3)*(x2)
    if not numerator or not denominator:
        return []

    #get their dimension
    pfnum = infix_to_postfix(numerator)
    pfden = infix_to_postfix(denominator)
    numtree = post_fix_to_tree(pfnum, vocabulary)
    dentree = post_fix_to_tree(pfden, vocabulary)

    numdim = numtree.root.dimension
    dendim = dentree.root.dimension

    # generate the trees
    numerator_trees = generate_from_scratch(num_trees, numdim, vocabulary, cfg, max_length//2)
    denominator_trees = generate_from_scratch(num_trees, dendim, vocabulary, cfg, max_length//2)

    n = min(len(numerator_trees), len(denominator_trees))
    final_trees = []

    for i in range(n):
        left_tree = numerator_trees[i]
        right_tree = denominator_trees[i]
        left_pf = left_tree.postfix_formula
        right_pf = right_tree.postfix_formula
        new_pf = left_pf + right_pf + ['/']
        new_tree = post_fix_to_tree(new_pf, global_vocabulary)
        final_trees.append(new_tree)

    return final_trees

def execute_fraction(num_trees, ground_truth, vocabulary, cfg, max_length, global_vocabulary, inf = 0, sup=1):
    # test
    hyper_parameters = choose_hyper_parameters(max_length, cfg, inf, sup)
    is_target_everywhere_positive = ground_truth['is_target_everywhere_positive']
    TB = Tree_Builder(vocabulary, ground_truth['target_dimension'], cfg, hyper_parameters, is_target_everywhere_positive)
    TB.build_dimensions_of_monomials()
    splits = TB.split_dimension(ground_truth['target_dimension'], '/')[0]
    N = len(splits)
    if N == 0:
        return []
    new_budget = num_trees // N
    if new_budget == 0:
        new_budget = 1
    final_trees = []
    not_working = [0]*len(splits)
    for k, (left_dim, right_dim) in enumerate(splits):
        local_pool, not_working = run_fraction(left_dim, right_dim, TB, new_budget, vocabulary, cfg, max_length, not_working, k, global_vocabulary)
        final_trees.extend(local_pool)
    if len(final_trees) < num_trees:
        remaining = num_trees - len(final_trees)
        indices = [i for i in range(len(not_working)) if not_working[i] == 0]
        if len(indices) == 0:
            return final_trees
        new_budget = remaining // len(indices)
        for k, (left_dim, right_dim) in enumerate([splits[i] for i in indices]):
            local_pool, not_working = run_fraction(left_dim, right_dim, TB, new_budget, vocabulary, cfg, max_length, not_working, k, global_vocabulary)
            final_trees.extend(local_pool)

    return final_trees

def generate_from_fraction(num_trees, ground_truth, vocabulary, cfg, max_length, eigenvector, global_vocabulary):
    if num_trees <1 : num_trees = 1
    final_trees = execute_fraction(num_trees, ground_truth, vocabulary, cfg, max_length, global_vocabulary)
    if not final_trees:
        final_trees = execute_fraction(num_trees, ground_truth, vocabulary, cfg, max_length,global_vocabulary, inf=0,sup=2)
        if not final_trees:
            final_trees = execute_fraction(num_trees, ground_truth, vocabulary, cfg, max_length, global_vocabulary,inf=-1, sup=1)
            if not final_trees:
                final_trees = generate_from_fraction_old(num_trees, ground_truth, vocabulary, cfg, max_length, eigenvector,global_vocabulary)
    return final_trees

def handle_non_integers(eigenvector, variables_name):

    num = []
    den = []
    non_zero_indices = [i for i in range(len(eigenvector)) if eigenvector[i] != 0]
    smallest = min(np.abs([eigenvector[i] for i in non_zero_indices]))
    eigenvector = eigenvector / smallest
    for i in non_zero_indices:
        if eigenvector[i] != int(eigenvector[i]):
            return [], []
    if smallest != 0.5:
        raise NotImplementedError('Te provided formula suggest that is it is some nth-sqrt ofa product of variables, but only sqrt is implemented')
        return [], [] #only implemented are sqrt
    else:
        num_str, den_str = get_num_denom(eigenvector, variables_name)

        if num_str and den_str:
            return [f'{num_str}/({den_str})']
        elif num_str:
            return [num_str]
        else:
            return [f'1/({den_str})']

def eigen_to_infix(eigenvector, variables_name, dimension_of_variables):
    idx = [i for i in range(len(eigenvector)) if eigenvector[i] != 0]
    random.shuffle(idx)
    #check if they are all integers
    not_int = [eigenvector[i] for i in idx if eigenvector[i] != int(eigenvector[i])]
    if len(not_int) > 0:
        return handle_non_integers(eigenvector, variables_name), True

    cnt = 0
    ans = []
    while cnt < 10:
        num = []
        den = []

        for i in idx:
            choices = [var for var in variables_name if dimension_of_variables[var] == dimension_of_variables[variables_name[i]]]
            variable = '(' + random.choice(choices) + ')'
            #variable = f'({variables_name[i]})'
            count = abs(int(eigenvector[i]))
            if eigenvector[i] > 0:
                num.extend([variable] * count)
            else:
                den.extend([variable] * count)

        num_str = '*'.join(num)
        den_str = '*'.join(den)

        if num_str and den_str:
            ans.append(f'{num_str}/({den_str})')
        elif num_str:
            ans.append(num_str)
        else:
            ans.append(f'1/({den_str})')

        cnt += 1
    return ans, False


def generate_from_prefactoring(num_trees, ground_truth, vocabulary, cfg, max_length, eigenvector, n_units, global_vocabulary):

    if num_trees <1 : num_trees = 1
    if np.sum(np.abs(eigenvector)) == 0:
        return []
    variables_name = ground_truth['variables_internal_name']
    dimension_of_variables = ground_truth['dimension_of_variables']
    infix_prefactors, need_sqrt = eigen_to_infix(eigenvector, variables_name, dimension_of_variables) # returns like '(G)*(m1)*(m2)/((r)**2)' in infix form but with internal variable names
    postfix_prefactors = [] #convert to postfix

    for infix in infix_prefactors:
        pf = infix_to_postfix(infix)
        pf = ['A' if x == '1' else x for x in pf]
        postfix_prefactors.append(pf)

    if need_sqrt:
        pfs = []
        if not cfg['use_sqrt']:
            return []
        else:
            for pf in postfix_prefactors:
                pfs.append(pf + ['np.sqrt('])
        postfix_prefactors = pfs

    adimensional_trees = generate_from_scratch(num_trees, [0] * n_units, vocabulary, cfg, max_length)
    final_trees = []
    for right_tree in adimensional_trees:
        right_postfix = right_tree.postfix_formula
        left_prefactor = random.choice(postfix_prefactors)
        new_postfix = left_prefactor + right_postfix + ['*']
        new_tree = post_fix_to_tree(new_postfix, global_vocabulary)
        final_trees.append(new_tree)

    return final_trees


def generate_random_trees(num_trees, ground_truth, cfg, max_length, global_vocabulary, use_simplification, k):
    """Main function to generate random equations.
    Will follow three different strategies:
    - random equations are built from scratch, trying to reach the target dimension by completing the tree from the root, given a constraint on the maximal length of the equation.
    - if it fails, it will use dimensional analysis to build the equation as numerator/denominator, and then complete both trees and join them
    - if it fails, it will use stronger result of dimensional analysis to build the equation as a product of two trees, the first have teh target dimension, the second a null dimenion
    """
    if config.use_seed:
        random.seed(k)
        np.random.seed(k)
    #print('entering generation of trees', num_trees)
    target_dimension = ground_truth['target_dimension']
    eigenvectors_with_target = ground_truth['eigenvectors_with_target']
    n_units = len(target_dimension)
    is_target_everywhere_positive = ground_truth['is_target_everywhere_positive']

    vocabulary_original_var_only = Vocabulary()
    vocabulary_original_var_only.build_vocabulary(cfg, ground_truth, use_original_variables=True,
                                                  use_adim_variables=False)

    vocabulary_both_var = Vocabulary()
    vocabulary_both_var.build_vocabulary(cfg, ground_truth, use_original_variables=True,
                                         use_adim_variables=True)

    vocabulary_adim_only = Vocabulary()
    vocabulary_adim_only.build_vocabulary(cfg, ground_truth, use_original_variables=False,
                                          use_adim_variables=True)

    local_debug = 0
    new_trees = []
    # if both target and all variables are dimensionless, we can only generate trees using dimensionless variables
    if ground_truth['is_everything_dimensionless']:
        #no need to replace vocabulary here
        vocabulary = Vocabulary()
        vocabulary.build_vocabulary(cfg, ground_truth)
        new_trees += generate_from_scratch(num_trees, target_dimension, vocabulary_original_var_only, cfg, max_length,
                                      is_target_everywhere_positive)

    elif target_dimension == [0]*n_units: #target is dimensionless but not the variables
        # we can generate trees : - using only original variables
        #                         - or with the help of adim variables as well
        #                         - or using only adim variables
        #                          but need to be checked that adim var exists indeed
        if len(ground_truth['adimensional_variables']):
            budget = num_trees//3
            new_trees += generate_from_scratch(budget, target_dimension, vocabulary_original_var_only, cfg, max_length,
                                              is_target_everywhere_positive)
            new_trees += generate_from_scratch(budget, target_dimension, vocabulary_both_var, cfg, max_length,
                                              is_target_everywhere_positive)
            #last one should be the most easy one, hence:
            new_trees += generate_from_scratch(num_trees - len(new_trees), target_dimension, vocabulary_adim_only, cfg, max_length,
                                              is_target_everywhere_positive)

        else:
            new_trees += generate_from_scratch(num_trees, target_dimension, vocabulary_original_var_only, cfg, max_length,
                                              is_target_everywhere_positive)

    else: #target has non zero dimension
        if local_debug : print('target has non zero dimension')
        eigenvector = -1 * eigenvectors_with_target[0][1:]

        if len(ground_truth['adimensional_variables']):
            budget = num_trees // 7
            new_trees += generate_from_scratch(budget, target_dimension, vocabulary_original_var_only, cfg, max_length,
                                              is_target_everywhere_positive)
            new_trees += generate_from_scratch(budget, target_dimension, vocabulary_both_var, cfg, max_length,
                                              is_target_everywhere_positive)
            new_trees += generate_from_fraction(budget, ground_truth, vocabulary_original_var_only, cfg, max_length,
                                                eigenvector, global_vocabulary)
            new_trees += generate_from_fraction(budget, ground_truth, vocabulary_both_var, cfg, max_length, eigenvector, global_vocabulary)
            new_trees += generate_from_prefactoring(budget, ground_truth, vocabulary_original_var_only, cfg, max_length,
                                                    eigenvector, n_units, global_vocabulary)
            new_trees += generate_from_prefactoring(budget, ground_truth, vocabulary_both_var, cfg, max_length,
                                                    eigenvector, n_units, global_vocabulary)
            # in case still not enough trees
            budget = max(budget, num_trees - len(new_trees))
            new_trees += generate_from_prefactoring(budget, ground_truth, vocabulary_adim_only, cfg, max_length,
                                                    eigenvector, n_units, global_vocabulary)
        else:
            budget = num_trees // 3
            new_trees += generate_from_scratch(budget, target_dimension, vocabulary_original_var_only, cfg, max_length,
                                              is_target_everywhere_positive)
            new_trees += generate_from_fraction(budget, ground_truth, vocabulary_original_var_only, cfg, max_length, eigenvector, global_vocabulary)
            new_trees += generate_from_prefactoring(num_trees-l, ground_truth, vocabulary_original_var_only, cfg, max_length, eigenvector,  n_units, global_vocabulary)

    if use_simplification:
        new_trees = [simplify_one_tree(tree, global_vocabulary) for tree in new_trees]

    return new_trees

def parallel_generation(tasks, n_cores: int):
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(generate_random_trees, *task) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

def keep_unique(trees):
    # remove duplicates
    final_trees = []
    all_postfix = []
    for tree in trees:
        postfix = tree.postfix_formula
        if postfix not in all_postfix:
            final_trees.append(tree)
            all_postfix.append(postfix)
    return final_trees


def main_tree_generation(num_trees, ground_truth, cfg, max_length, global_vocabulary, use_simplification, run_number):
    if config.use_parallel_tree_generation:
        tasks = []
        n = num_trees //config.n_cores
        for k in range(config.n_cores):
            tasks.append([n, ground_truth, cfg, max_length, global_vocabulary, use_simplification,10*run_number+1000*k])
        results = parallel_generation(tasks, config.n_cores)
        final_trees = []
        for r in results:
            final_trees.extend(r)
        return keep_unique(final_trees)
    else:
        return keep_unique(generate_random_trees(num_trees, ground_truth, cfg, max_length, global_vocabulary, use_simplification, 10*run_number))