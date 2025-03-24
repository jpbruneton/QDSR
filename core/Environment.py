from .Vocabulary import Vocabulary
from config import config
from Utils.Utils_ground_truth import compute_dimension_of_postfix_list
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

class Node:
    """
    Implements a Node class for building syntax trees.

    Each node:
    - Represents a single symbol in the tree (e.g., operators, functions, variables).
    - Has a list of children (no explicit parent references for flexibility).
    - Carries a dictionary of attributes related to the symbol's role in the tree.

    Key Features:
    - Downward attributes: Aggregated attributes from the root to this node, including the node itself.
    - Upward attributes: Aggregated attributes from this node to its leaves, including the node itself.
    - Attributes track structural and symbolic properties, aiding tree manipulation (e.g., genetic operations).

    Example Use Cases:
    - Limiting maximum nested functions or operations.
    - Computing tree features such as the number of variables, functions, or operators, by using the upward attributes of the rootnode.
    """
    def __init__(self,
                 symbol: str,
                 arity: int,
                 children: list,
                 dimension : list,
                 downward_attributes = None,
                 upward_attributes = None,
                 ):

        self.symbol = symbol
        self.arity = arity
        self.children = children
        self.dimension = dimension
        self.downward_attributes = downward_attributes
        self.upward_attributes = upward_attributes

    def return_node_attributes(self, vocabulary):
        attributes = {
            'functions': 0,
            'additions': 0,
            'minus': 0,
            'multiplications': 0,
            'divisions': 0,
            'powers': 0,
            'sqrt': 0,
            'free_scalars': 0,
            'variables': 0,
        }

        # Mapping of symbols to attribute updates
        symbol_mapping = {
            'np.sqrt(': {'sqrt': 1, 'functions': 1},
            '**': {'powers': 1},
            '+': {'additions': 1},
            '-': {'minus': 1},
            '*': {'multiplications': 1},
            '/': {'divisions': 1},
        }

        # Check symbol in predefined mappings
        if self.symbol in symbol_mapping:
            for key, value in symbol_mapping[self.symbol].items():
                attributes[key] += value

        elif self.symbol in vocabulary.arity_1_symbols_no_sqrt:
            attributes['functions'] += 1
        elif self.symbol in vocabulary.dimensionless_scalars:
            attributes['free_scalars'] += 1
        elif self.symbol in vocabulary.variables:
            attributes['variables'] += 1
        else:
            #print(f"Symbol {self.symbol} not found in the vocabulary.")
            pass
        return attributes

    def add_attribute(self, attribute1, attribute2):
        for key in attribute1:
            attribute1[key] += attribute2[key]
        return attribute1


def postfix_traversal(node: Node) -> List[Node]:
    """
    Execute a postfix traversal from node
    :param node: (Node) starting node
    :return: a list of all children of node, in postfix order
    """

    def _postfix_traversal(node: Node, result: List[Node]) -> None:
        for child in node.children:
            _postfix_traversal(child, result)
        result.append(node)

    result = []
    _postfix_traversal(node, result)
    return result

def get_postfix_formula(traversal: List[Node]) -> List[str]:
    """
    :param (List) a list of nodes in postfix order
    :returns (List)) : list of symbols carried by the nodes in traversal
    """
    postfix_formula = []
    for node in traversal:
        postfix_formula.append(node.symbol)
    return postfix_formula


def postfix_to_infix(postfix_formula, vocabulary):
    """
    Converts a postfix mathematical expression to infix notation with proper parentheses.

    Args:
        postfix_formula (list): Symbols in postfix notation.
        vocabulary (object): Contains:
            - `arity_0_symbols_with_power`: Zero-arity symbols (e.g., variables, constants).
            - `arity_1_symbols_with_sqrt`: Unary operators (e.g., `np.sqrt`).
            - `arity_2_with_power`: Binary operators (e.g., `+`, `*`, `**`).

    Returns:
        str: The formula in infix notation.

    Raises:
        ValueError: If the postfix formula is invalid.

    Example:
        postfix_to_infix(['x', 'y', '+', 'np.sqrt'], vocabulary) -> 'np.sqrt((x)+(y))'
    """
    stack = []
    for symbol in postfix_formula:
        if symbol in vocabulary.arity_0_symbols_with_power:
            stack.append(symbol)
        elif symbol in vocabulary.arity_1_symbols_with_sqrt:
            operand = stack.pop()
            stack.append(f"{symbol}{operand})")
        elif symbol in vocabulary.arity_2_with_power:
            right = stack.pop()
            left = stack.pop()
            stack.append(f"({left}){symbol}({right})")
        else:
            stack.append(symbol)
    if len(stack) != 1:
        raise ValueError("Invalid postfix formula: Unbalanced expression.")
    return stack[0]


def get_number_of_nested_functions(from_node : Any, list_of_functions : List[str]) -> int:
    """
    Calculates the maximum count of unary functions (e.g., square roots) encountered
    along any path from the given node to a leaf in the tree.

    Args:
        from_node (Node): The starting node of the tree or subtree.

    Returns:
        int: The maximum count of unary functions encountered along any path.
    """
    def max_f_on_path(node, current_f=0):
        """
        Recursively calculates the maximum count of unary functions on any path
        from the given node to a leaf.

        Args:
            node (Node or None): The current node in the recursion.
            current_f (int): The count of unary functions encountered so far.

        Returns:
            int: The maximum count of unary functions encountered from this node to a leaf.
        """
        if node is None:
            print('euh?')
            # Base case: If the node is None (e.g., an empty child), return the count so far.
            return current_f

        # Increment the count if the current node represents a unary function (e.g., sqrt).
        current_f += (1 if node.symbol in list_of_functions else 0)

        # If the node is a leaf (no children), return the current count.
        if not len(node.children):
            return current_f

        # Determine the left and right children of the node.
        # For nodes with one child, the right child is None.
        if len(node.children) == 1:
            left_node = node.children[0]
            right_node = None
        elif len(node.children) == 2:
            left_node = node.children[0]
            right_node = node.children[1]
        # Recursively calculate the maximum count for the left and right subtrees.

        left_f_count = max_f_on_path(left_node, current_f)
        right_f_count = max_f_on_path(right_node, current_f) if right_node else 0

        # Return the maximum of the counts from the left and right subtrees.
        return max(left_f_count, right_f_count)

    return max_f_on_path(from_node)

class Tree:
    """
    Represents a mathematical expression tree.

    Attributes:
        root (Node): The root node of the tree.
        vocabulary (Vocabulary): The vocabulary defining symbols and their properties.
        null_dimension (list[int]): A list of zeros, representing a null dimension in the tree.
        traversal (list[Node]): A list of nodes in postfix order.
        length (int): The number of nodes in the tree.
        node_dimensions (list[list[int]]): A list of dimensions for each node in the traversal.
        postfix_formula (list[str]): A postfix representation of the tree as a list of symbols.
        infix_formula (str): An infix representation of the tree as a string with parentheses = the formula.
    """
    def __init__(self, root: Node, vocabulary : Vocabulary):
        self.root = root
        self.vocabulary = vocabulary
        self.null_dimension = [0]*config.num_fundamental_units
        # Automatically compute the tree attributes :
        self.traversal = postfix_traversal(self.root)
        self.length = len(self.traversal)
        self.node_dimensions = [node.dimension for node in self.traversal]
        self.postfix_formula = get_postfix_formula(self.traversal)  # is a list of symbols in postfix order
        self.infix_formula = postfix_to_infix(self.postfix_formula, vocabulary)  # is an equation as a string including parentheses
        self.compute_upward_attributes(self.root)
        self.compute_downward_attributes(self.root)


    def compute_upward_attributes(self, node: Node) -> Dict[str, int]:
        """
        Recursively computes and aggregates the upward attributes for a node and its subtree.

        The upward attributes of a node are calculated by summing its own attributes with the aggregated
        attributes of all its descendant nodes. This process traverses the subtree bottom-up.

        Args:
            node (Node): The root of the subtree for which to compute upward attributes.

        Returns:
            Dict[str, int]: The aggregated attributes of the subtree rooted at the given node.
        """
        # If the node is a leaf, initialize its attributes
        if not node.children:
            leaf_attributes = node.return_node_attributes(self.vocabulary)
            node.upward_attributes = leaf_attributes
            return leaf_attributes

        # Initialize the subtree attributes
        subtree_attributes = node.return_node_attributes(self.vocabulary)
        # Recompute attributes for each child
        for child in node.children:
            child_attributes = self.compute_upward_attributes(child)
            for key in subtree_attributes:
                subtree_attributes[key] += child_attributes[key]

        # Update the node's attributes
        node.upward_attributes = subtree_attributes
        return subtree_attributes

    def compute_downward_attributes(self, node: Node, parent_attributes = None) -> None:
        """
        Recursively computes and propagates downward attributes for the given node and its subtree.

        Each node's downward attributes are calculated by summing its own attributes with the attributes passed
        down from its parent. The process continues recursively for all child nodes.

        Args:
            node (Node): The current node for which to compute downward attributes.
            parent_attributes (dict, optional): Attributes from the parent node to propagate.
                If None, the node is assumed to be the root, and its attributes are initialized as its own.

        Returns:
            None: The downward attributes are directly assigned to each node in the subtree.
        """
        if parent_attributes is None:
            root_attributes = node.return_node_attributes(self.vocabulary)
            node.downward_attributes = root_attributes
        else:
            node.downward_attributes = parent_attributes.copy()
            node_attributes = node.return_node_attributes(self.vocabulary)
            for key in node.downward_attributes:
                node.downward_attributes[key] += node_attributes[key]

        # Recompute attributes for each child
        for child in node.children:
            self.compute_downward_attributes(child, node.downward_attributes)

    def _validate_node_structure(self, node: Node) -> None:
        """
        Validates the structure of a single node, including its symbol and children count.

        Args:
            node (Node): The node to validate.

        Raises:
            AssertionError: If the node structure is invalid.
        """
        if node.arity == 0:
            assert node.symbol in self.vocabulary.arity_0_symbols_with_power, (
                f"Invalid arity 0 symbol: {node.symbol}"
            )
        elif node.arity == 1:
            assert node.symbol in self.vocabulary.arity_1_symbols_with_sqrt, (
                f"Invalid arity 1 symbol: {node.symbol}"
            )
            assert len(node.children) == 1, (
                f"Arity 1 node has invalid number of children: {len(node.children)}"
            )
        elif node.arity == 2:
            assert node.symbol in self.vocabulary.arity_2_with_power, (
                f"Invalid arity 2 symbol: {node.symbol}"
            )
            assert len(node.children) == 2, (
                f"Arity 2 node has invalid number of children: {len(node.children)}"
            )

    def check_tree(self, target_dimension: List[int]) -> None:
        """
        Validates the tree structure, node attributes, and dimensions against the target dimension.

        Args:
            target_dimension (list): Expected dimension at the root of the tree.

        Raises:
            AssertionError: If any structural or dimensional checks fail.
        """
        if config.skip_tree_validation:
            return
        # Ensure root dimension matches the target dimension
        assert self.root.dimension == target_dimension, (
            f"Root dimension {self.root.dimension} does not match target dimension {target_dimension}"
        )

        # Verify the postfix formula
        assert all(symbol in self.vocabulary.all_symbols for symbol in self.postfix_formula), (
            "Postfix formula contains invalid symbols"
        )

        # Compute and validate dimensions using the vocabulary

        try:
            #need to cast in math-style symbols
            postfix_formula = [x.replace('np.', '').replace('(', '') for x in self.postfix_formula]
            success, resulting_dim = compute_dimension_of_postfix_list(postfix_formula,
                                                                       self.vocabulary.dimensional_dict)
        except Exception as e:
            raise AssertionError(f"Dimension computation failed: {e}")

        assert success, "Dimension computation failed"
        assert resulting_dim == target_dimension, (
            f"Resulting dimension {resulting_dim} does not match target dimension {target_dimension}"
        )

        # Recompute and validate node structure and dimensions
        try:
            self.check_nodes([0] * len(target_dimension))
        except Exception as e:
            raise AssertionError(f"Node dimension check failed: {e}")

        #print("Tree validation successful.")


    def check_nodes(self, null_dimension: List[int]) -> None:
        """
        Validates node dimensions and arities for all nodes in the tree traversal.

        Args:
            null_dimension (list): The expected null dimension for certain operations.

        Raises:
            AssertionError: If any node's dimensions or arities are invalid.
        """
        for node in self.traversal:
            # Validate node arity and structure
            self._validate_node_structure(node)

            # Validate node dimensions based on its arity and symbol
            if node.arity == 0:
                expected_dim = self.vocabulary.dimensional_dict.get(node.symbol)
                assert node.dimension == expected_dim, (
                    f"Dimension mismatch for arity 0 node '{node.symbol}' (expected {expected_dim}, got {node.dimension})"
                )
            elif node.arity == 1:
                child = node.children[0]
                if node.symbol == 'np.sqrt(':
                    assert child.dimension == [2 * x for x in node.dimension], (
                        f"Dimension mismatch in sqrt operation for node {node}"
                    )
                else:
                    assert child.dimension == null_dimension, (
                        f"Non-zero dimension found in unary node {node}"
                    )
            elif node.arity == 2:
                left, right = node.children
                if node.symbol in ['+', '-']:
                    assert left.dimension == right.dimension, (
                        f"Dimension mismatch in addition/subtraction for node {node}"
                    )
                elif node.symbol == '*':
                    expected_dim = [x + y for x, y in zip(left.dimension, right.dimension)]
                    assert node.dimension == expected_dim, (
                        f"Dimension mismatch in multiplication for node {node} (expected {expected_dim}, got {node.dimension})"
                    )
                elif node.symbol == '/':
                    expected_dim = [x - y for x, y in zip(left.dimension, right.dimension)]
                    assert node.dimension == expected_dim, (
                        f"Dimension mismatch in division for node {node} (expected {expected_dim}, got {node.dimension})"
                    )
                else:
                    assert right.dimension == null_dimension, (
                        f"Non-zero dimension found in power operation for node {node}"
                    )

    def get_features(self) -> Tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
        """
        Extracts features from the traversal of a tree, including various counts of operations,
        functions, and structural attributes.

        Returns:
            tuple: A tuple containing the following features:
                - free_scalars (int): Count of free scalar palceholders.
                - equation_len (int): Total number of nodes in the tree traversal.
                - variables (int): Count of variable nodes.
                - functions (int): Count of function nodes.
                - powers (int): Count of power operations.
                - num_trig_functions (int): Count of trigonometric functions.
                - num_exp_log_functions (int): Count of exponential and logarithmic functions.
                - additions (int): Count of addition operations.
                - minus (int): Count of subtraction operations.
                - multiplications (int): Count of multiplication operations.
                - divisions (int): Count of division operations.
                - nested_functions (int): count of nested functions in the tree, eg sin(cos(x)) has 2 nested functions
        """
        equation_len = len(self.traversal)

        # Initialize counts for special functions
        num_trig_functions = 0
        num_exp_log_functions = 0
        trig_functions = config.trig_functions
        exp_log_functions = config.exp_log_functions

        for node in self.traversal:
            if node.symbol in trig_functions:
                num_trig_functions += 1
            elif node.symbol in exp_log_functions:
                num_exp_log_functions += 1

        tree_attributes = self.traversal[-1].upward_attributes

        # Construct and return the feature tuple
        return (
            tree_attributes['free_scalars'],
            equation_len,
            tree_attributes['variables'],
            tree_attributes['functions'],
            tree_attributes['powers'],
            num_trig_functions,
            num_exp_log_functions,
            tree_attributes['additions'],
            tree_attributes['minus'],
            tree_attributes['multiplications'],
            tree_attributes['divisions'],
            get_number_of_nested_functions(self.root, self.vocabulary.arity_1_symbols_with_sqrt),
        )


def post_fix_to_tree(postfix_list : list, vocabulary : Vocabulary) -> Tree:
    """
    Converts a postfix expression, given as a list of str symbols, into a tree structure of nodes.

    This function takes a postfix expression (as a list of tokens) and constructs
    a tree of nodes where each node represents an operator or operand. The tree
    is built by iterating over the postfix tokens and handling them based on their
    arity (0, 1, or 2). The dimensional attributes of each node are also assigned
    according to the dimensional information in the provided vocabulary.

    Args:
        postfix_list (list): A list of tokens representing a postfix expression.
        vocabulary (Vocabulary): The vocabulary containing dimension information
                                  and function definitions.

    Returns:
        Tree : A tree structure representing the postfix expression.

    Raises:
        AssertionError: If the postfix expression is invalid or if dimension mismatches occur.
    """
    stack = []
    dimension_dict = vocabulary.dimensional_dict

    for token in postfix_list:
        if token in vocabulary.arity_0_symbols_with_power: # leaves
            dim = dimension_dict[token]
            stack.append(Node(token, 0, [], dim))

        elif token in vocabulary.arity_1_symbols_with_sqrt: # unary
            child_node = stack.pop()
            child_dim = child_node.dimension
            #print('happening ici', token, child_dim)
            if token == 'np.sqrt(':
                parent_dim = [x//2 for x in child_dim]
            else:
                parent_dim = child_dim
            stack.append(Node(token, 1, [child_node], parent_dim))

        elif token in vocabulary.arity_2_with_power:
            right = stack.pop()
            left = stack.pop()
            right_dim = right.dimension
            left_dim = left.dimension

            if token in ['+', '-']:
                parent_dim = right_dim
            elif token == '*':
                parent_dim = [x + y for x, y in zip(left_dim, right_dim)]
            elif token == '/':
                parent_dim = [x - y for x, y in zip(left_dim, right_dim)]
            else:
                # power operator
                if right.symbol in vocabulary.integers_for_power:
                    parent_dim = [x * int(right.symbol) for x in left_dim]
                else:
                    parent_dim = [0] * config.num_fundamental_units
            stack.append(Node(token, 2, [left, right], parent_dim))
        else:
            # must be some dimensionless scalar
            dim = [0]*config.num_fundamental_units
            stack.append(Node(token, 0, [],  dim))

    assert(len(stack) == 1), 'invalid postfix expression'
    root = stack.pop()
    return Tree(root, vocabulary)

