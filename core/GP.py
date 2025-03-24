import numpy as np
import random
import time
from typing import Any, List, Dict,  Optional
from config import config
from core.Tree_Generator import main_tree_generation
from core import mutate_tree, swap_trees
from core import post_fix_to_tree
from core.Environment import Tree

class GP():
    """
    A Genetic Programming (GP) framework for evolving mathematical expressions.

    This class implements key functionalities for generating initial pools, and
    evolving them through mutation, crossover, and arity-1 deletion makking use of Genetic Operations module to do so.
    Attributes:
        vocabulary (Vocabulary): The vocabulary of operations, functions, and symbols
            used for constructing mathematical expressions.
        use_simplification (bool): Indicates whether expressions should be simplified
            during evolution to reduce complexity.
        cfg (dict): A configuration dictionary containing parameters for the GP process,
            such as mutation probabilities and pool sizes.
        ground_truth (dict): Properties of the target expression, including its
            dimensions and structure.
        run_number (int): A unique identifier for the current GP execution, useful for seeding initializations.
        logger (Logger): A logger instance for tracking progress, outputs, and debug
            information during the GP run.
    Methods:
        create_pool(num_trees, maximal_length): Creates the initial pool of random equations.
        generate_random_equations(num_trees, maximal_length): Generates random equations for the initial pool.
        prepare_selection_data(QD): Prepares sorted trees and losses for parent selection.
        select_individual(method, pool): Selects an individual from the pool based on the specified method.
        extend_pool(QD, maximal_length): Extends the pool of equations by applying mutation,
            crossover, and arity-1 deletions.

    """

    def __init__(self, vocabulary,
                 use_simplification, cfg,
                 ground_truth, run_number,
                 logger):

        self.run_number = run_number
        self.vocabulary = vocabulary
        self.use_simplification = use_simplification
        self.logger = logger
        self.initial_pool_size = cfg['initial_pool_size']
        self.extend_pool_factor = cfg['extend_pool_factor']
        self.use_noise = cfg['use_noise']
        self.mutation_probability = cfg['p_mutate']
        self.crossover_probability = cfg['p_cross']
        self.double_mutation_probability = cfg['proba_double_mutation']
        self.ground_truth = ground_truth
        self.null_dimension = [0]*len(config.fundamental_units)
        self.cfg = cfg
        self.target_dimension = ground_truth['target_dimension']
        self.selection_method = self.cfg['parent_selection_method']

    def create_pool(self, num_trees: int, maximal_length: int) -> List[Tree]:
        """
        Create the initial pool of random equations.

        Parameters:
            num_trees: Number of random equations to generate.
            maximal_length: Maximum length of each equation.

        Returns:
            List of generated equation trees.
        """
        self.logger.important('Creating initial pool of equations')
        return self.generate_random_equations(num_trees, maximal_length)

    def generate_random_equations(self, num_trees, maximal_length):
        """
        Creates N random equations

        Parameters:
        maximal_length (int): The maximum size of the equation.

        Returns:
        list: List of random equation states.
        """

        trees = main_tree_generation(num_trees, self.ground_truth,
                     self.cfg, maximal_length, self.vocabulary, self.use_simplification, self.run_number)
        self.logger.details(f'mean length {np.mean([t.length for t in trees])}')
        self.logger.details(f'min and max length {np.min([t.length for t in trees])}, {np.max([t.length for t in trees])}')

        len_trees = []
        for tree in trees:
            len_trees.append(tree.length)
            #self.logger.details(f'added to the pool {tree.infix_formula}')

        self.logger.details(f'average length of trees {np.mean(len_trees)}')
        self.logger.details(f'max length of trees {np.max(len_trees)}')
        self.logger.details(f'min length of trees {np.min(len_trees)}')
        self.logger.info(f'added {len(trees)} random trees to the initial pool')

        return trees

    def prepare_selection_data(self, QD: Any) -> (List[Tree], List[float], List[Tree]):
        """
        Prepare sorted trees and losses for parent selection.

        Parameters:
            QD: Object containing the quality-diversity grid.

        Returns:
            Tuple containing sorted trees, sorted losses, and sorted trees with functions.
        """

        grid_items = [
            (QD.qd_grid[k]['loss_no_noise'] if not self.use_noise else QD.qd_grid[k]['loss_noise'],
             QD.qd_grid[k]['tree'],
             QD.qd_grid[k])
            for k in QD.qd_grid
        ]
        revert = True if config.metric == 'R2' else False
        grid_items.sort(key=lambda x: x[0], reverse=revert) # sort by loss : from best to worst

        sorted_losses = [item[0] for item in grid_items]
        sorted_trees = [item[1] for item in grid_items]
        sorted_results = [item[2] for item in grid_items]

        # Extract equations that use functions
        sorted_trees_with_function = [res['tree'] for res in sorted_results if res['num_functions'] > 0]

        return sorted_trees, sorted_losses, sorted_trees_with_function

    def select_individual(self, method: str, pool: List[Tree]) -> Tree:
        """
        Select an individual from the pool based on the specified method.

        Parameters:
            method (str): The selection method. Supported values: 'random', 'best_loss'.
            pool (List[Tree]): The pool of individuals to select from.

        Returns:
            Tree: The selected individual.

        Raises:
            ValueError: If the provided method is not supported.
        """
        methods = {
            'random': lambda p: random.choice(p),
            'best_loss': self._select_best_loss,
        }
        if method not in methods:
            raise ValueError(f"Invalid parent selection method: {method}")
        return methods[method](pool)

    def _select_best_loss(self, pool: List[Tree]) -> Tree:
        """
        Select an individual based on loss, with a higher probability for better losses.

        Parameters:
            pool (List[Tree]): The pool of individuals to select from.

        Returns:
            Tree: The selected individual.
        """
        probabilities = np.flip(np.linspace(0.1, 1, len(pool)))
        probabilities /= probabilities.sum()
        return np.random.choice(pool, p=probabilities)

    def delete_arity_1(self, tree: Tree) -> Optional[Tree]:
        """
        Remove an arity-1 function from the given tree, if present.

        Parameters:
            tree (Tree): The tree to process.

        Returns:
            Optional[Tree]: A new tree with an arity-1 function removed, or None if no such function exists.
        """
        postfix_formula = tuple(tree.postfix_formula)
        arity1_indices = [i for i, x in enumerate(postfix_formula) if x in self.vocabulary.arity_1_symbols_no_sqrt]
        if arity1_indices:
            index = random.choice(arity1_indices)
            new_postfix = [x for i, x in enumerate(tree.postfix_formula) if i != index]
            return post_fix_to_tree(list(new_postfix), self.vocabulary)
        return None

    def _choose_action(self) -> str:
        """
        Randomly choose an evolutionary action based on configured probabilities.

        Returns:
            str: The chosen action ('mutation', 'crossover', 'arity_deletion').
        """
        u = random.random()
        if u <= self.mutation_probability:
            return 'mutation'
        elif u <= self.crossover_probability:
            return 'crossover'
        return 'arity_deletion'

    def _apply_mutation(self, sorted_trees: List[Tree]) -> List[Tree]:
        """
        Apply mutation to an individual from the sorted trees.

        Parameters:
            sorted_trees (List[Tree]): The sorted pool of trees.

        Returns:
            List[Tree]: A list containing the mutated tree(s), or an empty list if unsuccessful.
        """
        tree = self.select_individual(self.selection_method, sorted_trees)
        success, mutated_tree, _ = mutate_tree(tree, self.vocabulary, self.null_dimension)
        if success:
            v = random.random()
            if v > self.double_mutation_probability:
                self.total_mutation += 1
                return [mutated_tree]
            # Double mutation
            success, mutated_tree_2, _ = mutate_tree(mutated_tree, self.vocabulary, self.null_dimension)
            if success:
                self.total_mutation += 1
                return [mutated_tree_2]
        return []

    def _apply_crossover(self, sorted_trees: List[Tree], maximal_length: int) -> List[Tree]:
        """
        Apply crossover between two individuals from the sorted trees.

        Parameters:
            sorted_trees (List[Tree]): The sorted pool of trees.
            maximal_length (int): The maximum allowed length for the resulting trees.

        Returns:
            List[Tree]: A list containing the offspring trees, or an empty list if unsuccessful.
        """
        tree1 = self.select_individual(self.selection_method, sorted_trees)
        tree2 = self.select_individual(self.selection_method, sorted_trees)
        success, new_tree_1, new_tree_2, _ = swap_trees(
            tree1, tree2, self.vocabulary, maximal_length)
        if success:
            self.total_crossover += 1
            return [new_tree_1, new_tree_2]
        return []

    def _apply_arity_deletion(self, sorted_trees_with_function: List[Tree]) -> List[Tree]:
        """
        Apply arity-1 deletion to an individual from the sorted trees with functions.

        Parameters:
            sorted_trees_with_function (List[Tree]): Trees that contain arity-1 functions.

        Returns:
            List[Tree]: A list containing the new tree(s) after deletion, or an empty list if unsuccessful.
        """
        if not sorted_trees_with_function:
            return []
        tree = self.select_individual(self.selection_method, sorted_trees_with_function)
        new_tree = self.delete_arity_1(tree)
        if new_tree:
            self.total_suppressions += 1
            return [new_tree]
        return []

    def extend_pool(self, QD : Any, maximal_length : int) -> List[Tree]:
        """
        Extend the pool of equations by applying mutation, crossover, and arity-1 deletions.
        The latter, while less conventional, helps mitigate the overuse of arity-1 functions,
        which tend to emerge naturally during the evolutionary process.

        Parameters:
            QD: Object containing the quality-diversity grid.
            maximal_length: Maximum length of generated equations.

        Returns:
            A list of new trees.
        """

        # Initializations
        new_pool = []
        count = 0
        num_new_trees = self.extend_pool_factor * len(QD.qd_grid)
        sorted_trees, sorted_losses, sorted_trees_with_function = self.prepare_selection_data(QD)
        self.total_mutation, self.total_crossover, self.total_suppressions = 0, 0, 0
        store_times = {'mut': [], 'cross': [], 'suppr': []}

        while len(new_pool) < num_new_trees and count < 3*num_new_trees:
            action = self._choose_action()

            if action == 'mutation':
                start = time.time()
                new_individuals = self._apply_mutation(sorted_trees)
                store_times['mut'].append(time.time()-start)

            elif action == 'crossover':
                start = time.time()
                new_individuals = self._apply_crossover(sorted_trees, maximal_length)
                store_times['cross'].append(time.time()-start)

            else:
                start = time.time()
                new_individuals = self._apply_arity_deletion(sorted_trees_with_function)
                store_times['suppr'].append(time.time()-start)

            if new_individuals:
                new_pool.extend(new_individuals)
            count += 1

        self.logger.details(f'{self.total_mutation} mutations, {self.total_crossover} crossovers, {self.total_suppressions} arity-1 deletions')
        self.logger.details(f'Average time for mutation: {np.mean(store_times["mut"])}')
        self.logger.details(f'Average time for crossover: {np.mean(store_times["cross"])}')
        self.logger.details(f'Average time for arity-1 deletion: {np.mean(store_times["suppr"])}')

        return new_pool



