import numpy as np
from scipy.spatial import KDTree
import pickle

def build_and_save_kdtree(variables, file_path):
    """
    Build a k-d tree from the variables in the data and save it to a file.
    Parameters:
    - variables: A 2D numpy array where the columns are the variables defining the n-dimensional space.
    - file_path: Path to save the k-d tree.
    Returns:
    - the kdtree
    """
    print('building kdtree...')
    tree = KDTree(variables)
    with open(file_path, 'wb') as f:
        pickle.dump(tree, f)
    return tree

def denoise_with_neighbors_using_kdtree(target_function, variables, kdtree, k):
    """
    Denoise a function f using the k nearest neighbors in a multidimensional space.

    Parameters:
    - data: A 2D numpy array where the first column is the noisy function f,
            and the remaining columns are the variables defining the n-dimensional space.
    - k: The number of neighbors to consider for denoising.
    - file_path: Path to load the k-d tree.

    Returns:
    - denoised_f: A numpy array containing the denoised values of f.
    """
    print('denoising target...')
    # Find k-nearest neighbors for each point
    _, indices = kdtree.query(variables, k=k)

    # Compute the mean of f over the neighbors for denoising

    denoised_target = np.array([np.mean(target_function[neighbor_indices]) for neighbor_indices in indices])
    return denoised_target

