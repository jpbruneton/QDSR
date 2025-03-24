import random
from core import Node, Tree, post_fix_to_tree, postfix_traversal, get_postfix_formula
from config import config
from typing import List, Tuple, Optional
from core import Vocabulary

# Mutations are replacements of a node with another node of the same arity
# we could also consider mutations that always shorten the tree, like mapping arity 2 or 1 to arity 0 when applicable
# (may be an interesting option to consider, but not implemented here)

# powers/sqrt need special treatment in swaps and muts
# we recall that in the subtree structure left - power - right, there are two cases:
# 1. left has a dimension, then right must be dimensionless and in a list of integer such that node(power).dim = (left_node.dim)*right_node.value
# 2. left is dimensionless, then right must be dimensionless and can be any subtree
# when we try to mutate right nodes that carry an integer, we can't, since the dimension of the left subtree must be non zero

# thus, regarding mutations :
# 1. if the node is a power, then we can mutate to another operator iff the left subtree is dimensionless,
# equivalent to saying that power node is itself dimensionless
# 2. if the node is not a power, then we can mutate to another operator including power when it is dimensionless
# 3. Similarly, we can mutate the sqrt node only when it is dimensionless ; converse is allways possible since non sqrt arity 1 nodes are dimensionless

# regarding swaps : see details below

############### MUTATIONS ####################

def mutate_node(node: Node, vocabulary: Vocabulary, null_dimension: List[int]) -> Tuple[bool, Optional[Node]]:
    """
    Mutate a node by changing its symbol based on its arity and allowed vocabulary.
    :param node: Node to mutate.
    :param vocabulary: Vocabulary of symbols.
    :param null_dimension: Null dimension for validation.
    :return: Tuple indicating success and the mutated node (if successful), else False and None.
    """
    if node.arity == 0:
        return mutate_arity_0(vocabulary, node)
    if node.arity == 1:
        return mutate_arity_1(vocabulary, node, null_dimension)
    return mutate_arity_2(vocabulary, node, null_dimension)

def mutate_arity_0(vocabulary: Vocabulary, node: Node) -> Tuple[bool, Optional[Node]]:
    """
    Mutate a node with arity 0 (leaf node).
    :param vocabulary: Vocabulary of symbols.
    :param node: The node to mutate.
    :return: Tuple indicating success and the mutated node (if successful).
    """
    options = [
        x for x in vocabulary.arity_0_symbols_no_power
        if x != node.symbol and node.dimension == vocabulary.dimensional_dict.get(x)
    ]

    if not options:
        return False, None

    new_symbol = random.choice(options)
    return True, Node(new_symbol, 0, [], node.dimension)

def mutate_arity_1(vocabulary: Vocabulary, node : Node, null_dimension : List[int]) -> Tuple[bool, Optional[Node]]:
    """
    Mutate a node with arity 1 (unary operator).

    :param vocabulary: Vocabulary of symbols.
    :param node: The node to mutate.
    :param null_dimension: Null dimension for conditioning.
    :return: Tuple indicating success and the mutated node (if successful).
    """

    if node.symbol == 'np.sqrt(':
        if node.dimension != null_dimension:
            return False, None
        else:
            options = vocabulary.arity_1_symbols_no_sqrt

    else: #node is not sqrt, hence is dimensionless : sqrt always possible as well as other arity 1 nodes
        options = [x for x in vocabulary.arity_1_symbols_with_sqrt if x != node.symbol]

    if not options:
        return False, None

    new_symbol = random.choice(options)
    return True, Node(new_symbol, 1, [node.children[0]], node.dimension)

def mutate_arity_2(vocabulary, node, null_dimension):
    """
    Mutate a node with arity 2 (binary operator).

    :param node: Node to mutate.
    :param vocabulary: Vocabulary of symbols.
    :param null_dimension: Null dimension for validation.
    :param symbol: Current symbol of the node.
    :return: Tuple indicating success and the mutated node (if successful).
    """

    left_dim, right_dim = node.children[0].dimension, node.children[1].dimension

    if node.symbol == '**':
        if left_dim == null_dimension:
            assert right_dim == null_dimension, 'should be dimensionless1'
            assert node.dimension == null_dimension, 'should be dimensionless2'
            options = vocabulary.arity_2_no_power
        else:
            return False, None
    elif node.symbol in ['+', '-']:
        options = get_options_for_add_sub(vocabulary, null_dimension, node.symbol, left_dim)
    elif node.symbol in ['*', '/']:
        options = get_options_for_mul_div(vocabulary, null_dimension, node.symbol, left_dim, right_dim)
    else:
        return False, None

    if not options:
        return False, None
    new_symbol = random.choice(options)
    return True, Node(new_symbol, 2, [node.children[0], node.children[1]], node.dimension)

def get_options_for_add_sub(vocabulary: Vocabulary, null_dimension: List[int], symbol: str, left_dim: List[int]) -> List[str]:
    """
    Get valid mutation options for addition and subtraction nodes.

    :param vocabulary: Vocabulary of symbols.
    :param null_dimension: Null dimension for validation.
    :param symbol: Current symbol of the node.
    :param left_dim: Dimension of the left child node.
    :return: List of valid symbols for mutation.
    """
    if left_dim == null_dimension: #node was +  or - meaning that both left and right are dimensionless, thus we can use any oprator here
        return [x for x in vocabulary.arity_2_with_power if x != symbol]
    else: #nodes have dimensions thus + can become - and vice versa but nothing else
        return [x for x in ['+', '-'] if x != symbol]

def get_options_for_mul_div(vocabulary: Vocabulary, null_dimension: List[int], symbol: str, left_dim: List[int], right_dim: List[int]) -> List[str]:
    """
    Get valid mutation options for multiplication and division nodes.

    :param vocabulary: Vocabulary of symbols.
    :param null_dimension: Null dimension for validation.
    :param symbol: Current symbol of the node.
    :param left_dim: Dimension of the left child node.
    :param right_dim: Dimension of the right child node.
    :return: List of valid symbols for mutation.
    """
    if left_dim == null_dimension and right_dim == null_dimension:
        return [x for x in vocabulary.arity_2_with_power if x != symbol]
    elif left_dim == null_dimension:
        return None
    elif right_dim == null_dimension:
        return [x for x in ['*', '/'] if x != symbol]
    else:
        return None

def mutate_tree(tree: Tree, vocabulary: Vocabulary, null_dimension: List[int]) -> Tuple[bool, Optional[Tree], int]:
    """
    Mutate a tree by randomly selecting a node and mutating it.

    :param tree: Tree to mutate.
    :param vocabulary: Vocabulary of symbols.
    :param null_dimension: Null dimension for validation.
    :return: Tuple indicating success, the mutated tree (if successful), and the number of tries.
    """
    tries = 0
    previous_postfix = tuple(tree.postfix_formula)

    while tries < 10:
        idx = random.randint(0, len(tree.traversal) - 1)
        picked_node = tree.traversal[idx]
        if picked_node.symbol in vocabulary.integers_for_power: # cant be mutated
            continue
        success, new_node = mutate_node(picked_node, vocabulary, null_dimension)
        if success:
            new_pf = previous_postfix[:idx] + (new_node.symbol,) + previous_postfix[idx + 1:]
            new_tree = post_fix_to_tree(new_pf, vocabulary)
            return True, new_tree, tries
        tries += 1

    return False, None, tries


################# CROSSOVERS ####################
# This implements the following logic for node swapping:
# 1. Find all the possible swaps between the two trees
# 2. Randomly select a swap and swap the subtrees
# 3. If the new trees are within the maximum length, return them
# 4. else try again
# 5. If no possible swaps are found, return None

# more details about all the possible swaps between the two trees:
# - root node of subtree 1 and rootnode of subtree 2 :
#   - must have the same dimension
#   - cant be both leaves (as it would be equivalent to a simple mutation)
#   - cant be both roots of the whole trees as it would do nothing
#   - regarding powers :
#       - clearly cant be a power integer
#       - must respect the max nested power constraint
#   - regarding functions :
#       - must respect the max nested function constraint

def are_not_both_roots(l1: int, l2: int, idx_1: int, idx_2: int) -> bool:
    """
    Checks if the given indices do not correspond to the roots of the respective trees.

    Args:
        l1 (int): The length of the first tree.
        l2 (int): The length of the second tree.
        idx_1 (int): Index in the first tree.
        idx_2 (int): Index in the second tree.

    Returns:
        bool: True if neither index corresponds to the root of its tree; False otherwise.
    """
    return not (idx_1 == l1 - 1 and idx_2 == l2 - 1)

def are_not_both_leaves(node1: 'Node', node2: 'Node') -> bool:
    """
    Checks if the given nodes are not both leaves.

    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.

    Returns:
        bool: True if the nodes are not both leaves; False otherwise.
    """
    return not (node1.arity == 0 and node2.arity == 0)

def is_valid_dimension(dim_1: tuple, dimensions_dict_2: dict) -> bool:
    """
    Checks if the given dimension exists in the dimensions dictionary.

    Args:
        dim_1 (tuple): The dimension to check.
        dimensions_dict_2 (dict): The dictionary containing valid dimensions.

    Returns:
        bool: True if the dimension is valid; False otherwise.
    """
    return tuple(dim_1) in dimensions_dict_2

def meets_function_and_power_constraints(tree1: 'Tree', tree2: 'Tree', idx_1: int, idx_2: int, max_f: int, max_p: int) -> bool:
    """
    Checks if the given trees and indices meet the function and power constraints.

    Args:
        tree1 (Tree): The first tree.
        tree2 (Tree): The second tree.
        idx_1 (int): Index in the first tree.
        idx_2 (int): Index in the second tree.
        max_f (int): Maximum allowable functions.
        max_p (int): Maximum allowable powers.

    Returns:
        bool: True if the constraints are met; False otherwise.
    """
    return (
        tree2.traversal[idx_2].upward_attributes['functions'] + tree1.traversal[idx_1].downward_attributes['functions'] <= max_f and
        tree1.traversal[idx_1].upward_attributes['functions'] + tree2.traversal[idx_2].downward_attributes['functions'] <= max_f and
        tree2.traversal[idx_2].upward_attributes['powers'] + tree1.traversal[idx_1].downward_attributes['powers'] <= max_p and
        tree1.traversal[idx_1].upward_attributes['powers'] + tree2.traversal[idx_2].downward_attributes['powers'] <= max_p
    )

def find_possible_swaps(tree1: Tree, tree2: Tree, vocabulary: Vocabulary) -> List[List[int]]:
    """
    Identify all valid node pairs for subtree swaps between two trees.
    :param tree1: First tree.
    :param tree2: Second tree.
    :param vocabulary: Vocabulary of symbols.
    :return: List of valid node index pairs for swapping.
    """
    # Precompute dimensions and traversal lengths
    dimensions_tree1 = [node.dimension for node in tree1.traversal]
    dimensions_tree2 = [node.dimension for node in tree2.traversal]
    l1, l2 = len(dimensions_tree1), len(dimensions_tree2)

    # Create a dictionary to map dimensions to indices in tree2
    dimensions_dict_2 = {}
    for idx_2, dim_2 in enumerate(dimensions_tree2):
        dimensions_dict_2.setdefault(tuple(dim_2), []).append(idx_2)

    # Collect valid swaps
    possible_swaps = []
    max_f, max_p = config.MAX_NESTED_FUNCTIONS, config.MAX_NESTED_POWERS
    for idx_1, dim_1 in enumerate(dimensions_tree1):
        if tuple(dim_1) not in dimensions_dict_2:
            continue
        if tree1.traversal[idx_1].symbol in vocabulary.integers_for_power:
            continue
        for idx_2 in dimensions_dict_2[tuple(dim_1)]:
            if (
                are_not_both_roots(l1, l2, idx_1, idx_2)
                and are_not_both_leaves(tree1.traversal[idx_1], tree2.traversal[idx_2])
                and tree2.traversal[idx_2].symbol not in vocabulary.integers_for_power
                and meets_function_and_power_constraints(tree1, tree2, idx_1, idx_2, max_f, max_p)
            ):
                possible_swaps.append([idx_1, idx_2])

    return possible_swaps


def swap_subtrees(tree1: Tree, tree2: Tree, vocabulary: Vocabulary, max_length: int, possible_swap_indices: list) -> tuple:
    """
    Attempts to swap subtrees between two trees while adhering to constraints.

    Args:
        tree1 (Tree): The first tree to modify.
        tree2 (Tree): The second tree to modify.
        vocabulary (Vocabulary): The vocabulary for constructing new trees.
        max_length (int): The maximum allowable length of the resulting postfix formulas.
        possible_swap_indices (list): List of tuple pairs (i, j) indicating indices of subtrees to attempt swapping.

    Returns:
        tuple: A tuple containing:
            - success (bool): True if a valid swap was performed, False otherwise.
            - new_tree_1 (Tree or None): The modified version of `tree1` if successful, otherwise None.
            - new_tree_2 (Tree or None): The modified version of `tree2` if successful, otherwise None.
            - tries (int): The number of swap attempts made.
    """
    success = False
    tries = 0
    postfix_1 = tree1.postfix_formula
    postfix_2 = tree2.postfix_formula

    while not success and possible_swap_indices:
        tries += 1
        i, j = possible_swap_indices.pop()

        # Get the subtree postfix formulas for both trees
        traversal1 = postfix_traversal(tree1.traversal[i])
        sub_postfix_1 = get_postfix_formula(traversal1)
        traversal2 = postfix_traversal(tree2.traversal[j])
        sub_postfix_2 = get_postfix_formula(traversal2)

        # Swap the subtrees in their respective postfix formulas
        new_pf1 = postfix_1[:i - len(sub_postfix_1) + 1] + sub_postfix_2 + postfix_1[i + 1:]
        new_pf2 = postfix_2[:j - len(sub_postfix_2) + 1] + sub_postfix_1 + postfix_2[j + 1:]

        # Check if the new formulas are within the allowable length
        if len(new_pf1) <= max_length and len(new_pf2) <= max_length:
            success = True

    # Generate new trees if successful, otherwise return None
    if success:
        new_tree_1 = post_fix_to_tree(new_pf1, vocabulary)
        new_tree_2 = post_fix_to_tree(new_pf2, vocabulary)
    else:
        new_tree_1 = None
        new_tree_2 = None

    return success, new_tree_1, new_tree_2, tries

def swap_trees(tree1: Tree, tree2: Tree, vocabulary: Vocabulary, max_length: int) -> tuple:
    """
    Attempts to swap subtrees between two trees.

    Args:
        tree1 (Tree): The first tree to modify.
        tree2 (Tree): The second tree to modify.
        vocabulary (Vocabulary): The vocabulary of symbols used to construct trees.
        max_length (int): The maximum allowable length of the resulting trees.

    Returns:
        tuple: A tuple containing:
            - success (bool): True if a valid swap was performed, False otherwise.
            - new_tree_1 (Tree or None): The modified version of `tree1` if successful, otherwise None.
            - new_tree_2 (Tree or None): The modified version of `tree2` if successful, otherwise None.
            - tries (int): The number of swap attempts made.
    """

    possible_swaps = find_possible_swaps(tree1, tree2, vocabulary)
    if not possible_swaps:
        return False, None, None, 1

    # Shuffle the possible swaps to randomize attempts
    random.shuffle(possible_swaps)

    # Attempt to swap subtrees
    success, new_tree_1, new_tree_2, tries = swap_subtrees(
        tree1, tree2, vocabulary, max_length, possible_swaps
    )
    return success, new_tree_1, new_tree_2, tries