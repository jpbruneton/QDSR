import pickle
import os
import pandas as pd
from typing import  Dict
import shutil

def create_directory(dataset_name) -> None:
    """
    Create the necessary directories for storing results and targets.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_to_create = ['Results', f'Results/{dataset_name}', f'Targets/{dataset_name}',
                     f'Targets/{dataset_name}/raw_data',
                     f'Targets/{dataset_name}/Preprocessed_data', 'logs']

    for dir_ in dir_to_create:
        abs_path = os.path.join(root_dir, dir_)
        if not os.path.exists(abs_path):
            os.makedirs(abs_path)


def check_metadata_downloads(dataset_name) -> None:
    """
    Check if essential files and directories exist in the 'Targets' folder.
    This function verifies the presence of the following:
    - 'Targets/dataset_name/FeynmanEquations.csv': Contains the main equations.
    - 'Targets/dataset_name/units.csv': Contains associated units for the equations.
    - 'Targets/Feynman_with_units': Directory for storing targets with units.

    If any of these are missing, instructions are printed for downloading the files
    from the specified URL and placing them in the 'Targets' folder.
    Raises a ValueError if any required file or folder is missing.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(root_dir, f'Targets/{dataset_name}/FeynmanEquations.csv')
    if not os.path.exists(abs_path):
        print(
            f'Targets/{dataset_name}/FeynmanEquations.csv not found, download all files from https://space.mit.edu/home/tegmark/aifeynman.html')
        print('and place all of them in the right folder')
        raise ValueError("Missing 'FeynmanEquations.csv' in 'Targets/Feynman_with_units/'.")

    abs_path = os.path.join(root_dir, f'Targets/{dataset_name}/units.csv')
    if not os.path.exists(abs_path):
        print('Targets/units.csv not found, download all files from https://space.mit.edu/home/tegmark/aifeynman.html')
        print('and place all of them in the right folder')
        raise ValueError("Missing 'units.csv' in 'Targets/Feynman_with_units/'.")

def check_data_downloads(dataset_name, equation_label) -> None:
    """
    Check if essential files and directories exist in the 'Targets' folder.
    This function verifies the presence of the following:
    'Targets/dataset_name/{equation_label}.csv'
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(root_dir, f'Targets/{dataset_name}/raw_data/{equation_label}')
    if not os.path.exists(abs_path):
        print(f'Targets/{dataset_name}/raw_data/{equation_label} not found, download all files from https://space.mit.edu/home/tegmark/aifeynman.html')
        print('and place all of them in the right folder /Targets/dataset_name/raw_data')
        raise ValueError(f"Missing '{equation_label}' file in 'Targets/{dataset_name}/raw_data'.")


def get_unit_dict(dataset_name) -> Dict:
    """
    Get the dictionary of units for all variables.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(root_dir, f'Targets/{dataset_name}/unit_dict.pkl')
    try:
        units_dict = pickle.load(open(target_path, 'rb'))
    except Exception as e:
        raise ValueError(f"Error loading unit_dict from {target_path}: {e}")
    return units_dict

def load_ground_truth(dataset_name, equation_label: str) -> Dict:
    """
    Load the ground truth dictionary for a given equation.

    Args:
        equation_label (str): The label of the equation to load.

    Returns:
        Dict: The dictionary of metadata for the specified equation.

    Raises:
        ValueError: If the required formula dictionary file does not exist or the equation is not found.
    """

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_dir, f'Targets/{dataset_name}/formula_dict_with_units')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            formula_dict = pickle.load(f)
        if equation_label in formula_dict:
            return formula_dict[equation_label]
        else:
            error_message = f"Equation '{equation_label}' not found in the loaded dictionary."
            print(error_message)
            raise ValueError(error_message)
    else:
        error_message = f"{file_path} not found."
        print(error_message)
        raise ValueError(error_message)

def save_ground_truth(ground_truth, dataset_name):
    """
    Save the ground truth dictionary to a file.
    Args:
        ground_truth (dict): The ground truth dictionary to save.
        dataset_name (str): The name of the dataset.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root_dir, f'Targets/{dataset_name}/formula_dict_with_units')
    with open(save_path, 'wb') as f:
        pickle.dump(ground_truth, f)


def delete_preprocessed_data(dataset_name):
    """
    Delete the preprocessed_data for the given dataset.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(root_dir, f'Targets/{dataset_name}/Preprocessed_data')
    if os.path.exists(directory):
        shutil.rmtree(directory)

def delete_dictionnaries(dataset_name):
    """
    Delete the dictionaries for the given dataset.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dictionary = os.path.join(root_dir, f'Targets/{dataset_name}/formula_dict_with_units')
    if os.path.exists(dictionary):
        os.remove(dictionary)
    unit_dictionnary = os.path.join(root_dir, f'Targets/{dataset_name}/unit_dict.pkl')
    if os.path.exists(unit_dictionnary):
        os.remove(unit_dictionnary)
