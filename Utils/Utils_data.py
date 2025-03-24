import pickle
import os
import pandas as pd
import numpy as np

from Utils.Denoising import denoise_with_neighbors_using_kdtree, build_and_save_kdtree
from Utils import Utils_parsing
from Utils.Utils_misc import remove_unknown_cols_in_df



def load_target_from_data(target_filepath: str) -> pd.DataFrame:
    """
    Load and validate target data from a file.
    Reads the file as a pandas DataFrame, removes invalid columns, and checks
    if the DataFrame has at least two columns (at least one variable, one target)
    Args:
        target_filepath (str): Path to the target data file.
    Returns:
        pd.DataFrame: The loaded target data as a DataFrame.
    """
    try:
        target_data = pd.read_csv(target_filepath, sep=' ')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {target_filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file '{target_filepath}': {e}")

    target_data = remove_unknown_cols_in_df(target_data)
    if target_data.shape[1] < 2:
        raise ValueError(f"Invalid data: Expected at least 2 columns, found {target_data.shape[1]} in file '{target_filepath}'")
    return target_data


def cast_variables(n_var, formula):
    """
    Replace 'x0', 'x1', ... by 'X[:, 0]', 'X[:, 1]', in the formula string.
    Must be done in reverse order to avoid replacing both 'x1' and 'x10' by 'X[:, 1]' when 'x10' is present (ie if more than 10 variables)
    """
    for k in reversed(range(n_var)):
        formula = formula.replace(f'x{k}', f'X[:, {k}]')
    return formula

def compute_target(formula, true_name_to_internal_name_dict, n_variables, X):
    """
    Compute the target function from the formula and the provided variables.
    :param (str) the formula in usual math notation, eg. 'x + y'
    :param (dict) true_name_to_internal_name_dict, eg. {'x': 'x0', 'y': 'x1'}
    :param n_variables: (int) number of variables
    :param X: (np.array) the variables values ; shape (n_points, n_variables)
    :return: the target function values ; shape (n_points,)

    """
    numpy_formula = Utils_parsing.convert_to_internal_names_and_numpy_representation(formula,true_name_to_internal_name_dict)
    numpy_formula = cast_variables(n_variables, numpy_formula) #replace 'x0', 'x1', ... by 'X[:, 0]', 'X[:, 1]', ...
    try:
        f_true = eval(numpy_formula)
    except Exception as e:
        raise ValueError(f'Error computing the target function: {e}, maybe you are using a function that is not in the list of functions,'
                         f'in which case you should add it in the config file in all_function_list ; check also for typos in the formula;')
    return f_true

def preprocess_data(equation_label, dataset_name, ground_truth, use_noise, noise_level, use_denoising, k_neighbors=10, recompute = False):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(root_dir, f'Targets/{dataset_name}/Preprocessed_data')

    target_path = f'{main_dir}/{equation_label}'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    n_variables = ground_truth['n_variables']

    if (not os.path.exists(target_path + '/noiseless_data.npy') or recompute):
        original_csv_path = f'Targets/{dataset_name}/raw_data/{equation_label}'
        dataframe = load_target_from_data(original_csv_path)
        data_numpy = dataframe.to_numpy()
        X = data_numpy[:, :-1] #original variables
        if not recompute:
            f = data_numpy[:, -1] #target function is allways the last column
        else:
            formula = ground_truth['formula']
            true_name_to_internal_name_dict = ground_truth['true_name_to_internal_name_dict']
            f = compute_target(formula, true_name_to_internal_name_dict, n_variables, X)
        # complete flags in the ground truth file
        ground_truth['is_target_everywhere_positive'] = all(f > 0)
        ground_truth['apply_global_minus_one'] = all(f < 0)
        # save it
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, f'Targets/{dataset_name}/formula_dict_with_units')

        with open(file_path, 'rb') as file:
            formula_dict = pickle.load(file)
        formula_dict[equation_label] = ground_truth
        with open(file_path, 'wb') as file:
            pickle.dump(formula_dict, file)

        # add the extra variables
        adim_variables_dict = ground_truth['adimensional_dict'] #format {'y0': '(x6)/((x3))', ...}
        norms_dict = ground_truth['norms'] #format norms {'norm0': 'np.sqrt((x0) * (x0) + (x1) * (x1))',
        new_columns = []
        for k,v in adim_variables_dict.items():
            adim_formula = v
            adim_formula = cast_variables(n_variables, adim_formula)
            new_column = eval(adim_formula)
            new_columns.append(new_column)
        for k,v in norms_dict.items():
            norm_formula = v
            norm_formula = cast_variables(n_variables, norm_formula)
            new_column = eval(norm_formula)
            new_columns.append(new_column)
        new_columns = np.array(new_columns).T
        if new_columns.shape[0] != X.shape[0]:
           extended_variables = X #when no extra variables are required
        else:
            extended_variables = np.concatenate((X, new_columns), axis=1)
        # add back the target function
        noiseless_data = np.concatenate((extended_variables, f[:, None]), axis=1)
        np.save(target_path + '/noiseless_data.npy', noiseless_data)
    else:
        noiseless_data = np.load(target_path + '/noiseless_data.npy')
        f = noiseless_data[:, -1]
        X = noiseless_data[:, :n_variables]
        extended_variables = noiseless_data[:, :-1]

    if use_noise:
        idx_noise = [0.001, 0.01, 0.1].index(noise_level) + 1
        noisy_data_path = target_path + f'/noisy_data{idx_noise}.npy'
        if os.path.exists(noisy_data_path):
            noisy_data = np.load(noisy_data_path)
            f_noise = noisy_data[:, -1]
        else:
            sigma_f = np.std(f)
            noise = noise_level * np.random.normal(loc=0.0, scale=sigma_f, size=f.size)
            f_noise = f + noise
            noisy_data = np.concatenate((extended_variables, f_noise[:, None]), axis=1)
            np.save(noisy_data_path, noisy_data)

        if use_denoising:
            denoised_data_path = target_path + f'/denoised_data{idx_noise}_{k_neighbors}.npy'
            if os.path.exists(denoised_data_path):
                pass
            else:
                kdtree_path = target_path + f'/kdtree{idx_noise}.pkl'
                if os.path.exists(kdtree_path):
                    kdtree = pickle.load(open(kdtree_path, 'rb'))
                else:
                    kdtree = build_and_save_kdtree(X, kdtree_path)

                denoised_data = denoise_with_neighbors_using_kdtree(f_noise, X, kdtree, k_neighbors)
                denoised_data = np.concatenate((extended_variables, denoised_data[:, None]), axis=1)
                np.save(denoised_data_path, denoised_data)

    #train/test split
    path_noiseless = target_path + '/noiseless_data.npy'
    noiseless_data = np.load(path_noiseless)
    L = noiseless_data.shape[0]
    indices = np.random.permutation(L)
    train_size = int(0.75 * L)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_noiseless_data = noiseless_data[train_indices]
    test_noiseless_data = noiseless_data[test_indices]
    np.save(target_path + '/train_noiseless_data.npy', train_noiseless_data)
    np.save(target_path + '/test_noiseless_data.npy', test_noiseless_data)
    if use_noise:
        # noisy_path = self.main_directory + 'Preprocessed_data/' + self.equation_label + f'/{set}_noisy_data{idx_noise}.npy'
        idx_noise = [0.001, 0.01, 0.1].index(noise_level) + 1
        noisy_data_path = target_path + f'/noisy_data{idx_noise}.npy'
        train_noisy_data = np.load(noisy_data_path)[train_indices]
        test_noisy_data = np.load(noisy_data_path)[test_indices]
        np.save(target_path + f'/train_noisy_data{idx_noise}.npy', train_noisy_data)
        np.save(target_path + f'/test_noisy_data{idx_noise}.npy', test_noisy_data)
        if use_denoising:
            denoised_data_path = target_path + f'/denoised_data{idx_noise}_{k_neighbors}.npy'
            train_denoised_data = np.load(denoised_data_path)[train_indices]
            test_denoised_data = np.load(denoised_data_path)[test_indices]
            np.save(target_path + f'/train_denoised_data{idx_noise}.npy', train_denoised_data)
            np.save(target_path + f'/test_denoised_data{idx_noise}.npy', test_denoised_data)
    # print(f'Preprocessing data done for {equation_label}')

def generate_data(ground_truth, ranges, n_points=10000, noise_level=0):
    """
    Generate data from a formula. Not used in Feynman AI, but required for custom formula with no data
    Args:
    formula (str): formula to generate data from
    dataset_name (str): name of the dataset
    range (dict): range of the data
    n_points (int): number of points to generate
    noise_level (float): level of noise to add to the data
    """
    variables_true_name = ranges.keys()
    n_variables = len(variables_true_name)
    true_name_to_internal_name_dict = ground_truth['true_name_to_internal_name_dict']
    formula = ground_truth['formula']

    X = []
    for var in variables_true_name:
        var_range = ranges[var]
        var_samples = np.random.uniform(var_range[0], var_range[1], n_points)
        X.append(var_samples)
    X = np.array(X).T
    target = compute_target(formula, true_name_to_internal_name_dict, n_variables, X)
    sigma_f = np.std(target)
    noise = noise_level * np.random.normal(loc=0.0, scale=sigma_f, size=target.size)
    target = target + noise # = target if noise_level = 0
    return np.concatenate((X, target[:, None]), axis=1)

