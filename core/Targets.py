"""
Target Module

This module provides classes for loading and preprocessing target data, as well as representing target objects.
It includes methods for loading target data from a file, normalizing data, adding noise, and subsampling the data.

Main functionalities:
- Load and preprocess target data from a CSV file. Must be a space separated file.
- Normalize the data and calculate ranges of variation of both variables and target function.
- Add noise to the target function.
- Get a random subsample of the target data.

Classes:
- Data_Loader: Class to load and preprocess target data.
- Target: Class representing a target object with variables and objective function.
"""
import numpy as np
from config import config

class Data_Loader:
    """
    Class to load and preprocess target data.
    """
    def __init__(self,
                 apply_global_minus_one,
                 logger: 'Logger',
                 use_noise,
                 noise_level,
                 use_denoising,
                 equation_label,
                 main_directory,
                 k_neighbors) -> None:
        """
        Initialize the Data_Loader with meta_mode and logger.

        Parameters:
        meta_mode (str): Mode for data transformation ('linlin', 'loglin', 'loglog').
        logger (Logger): Logger object for logging information.
        """
        try:
            self.meta_mode = config.meta_mode
            self.logger = logger
            self.apply_global_minus_one = apply_global_minus_one
            self.use_noise = use_noise
            self.noise_level = noise_level
            self.use_denoising = use_denoising
            self.equation_label = equation_label
            self.main_directory = main_directory
            self.k_neighbors = k_neighbors

            # if use_noise is False, noisy_data is the same as noiseless_data:
            # if use noise and use_denoising, return denoised data, else return noisy data
            self.train_noisy_data= self.preprocess_data(self.load_noisy_data('train'))
            self.train_noiseless_data = self.preprocess_data(self.load_noiseless_data('train'))
            self.test_noisy_data = self.preprocess_data(self.load_noisy_data('test'))
            self.test_noiseless_data = self.preprocess_data(self.load_noiseless_data('test'))
            self.noiseless_data = np.concatenate((self.train_noiseless_data, self.test_noiseless_data), axis=0)
            self.noisy_data = np.concatenate((self.train_noisy_data, self.test_noisy_data), axis=0)

        except Exception as e:
            print('error in Data_Loader init', e)
            raise e

    def load_noiseless_data(self, set) -> np.ndarray:
        noiseless_path = self.main_directory + 'Preprocessed_data/' + self.equation_label + f'/{set}_noiseless_data.npy'
        return np.load(noiseless_path)

    def load_noisy_data(self, set) -> np.ndarray:
        if not self.use_noise:
            return self.load_noiseless_data(set)

        idx_noise = [0.001, 0.01, 0.1].index(self.noise_level)+1
        if not self.use_denoising:
            noisy_path = self.main_directory + 'Preprocessed_data/' + self.equation_label + f'/{set}_noisy_data{idx_noise}.npy'
            return np.load(noisy_path)
        else:
            denoised_path = self.main_directory + 'Preprocessed_data/' + self.equation_label + f'/{set}_denoised_data{idx_noise}_{self.k_neighbors}.npy'
            return np.load(denoised_path)

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the target data by applying log/ loglog if required by meta_mode and global minus one when target is everywhere negative.
        """
        # actually not used; runs are made in linlin mode
        if self.meta_mode == 'linlin':
            pass
        elif self.meta_mode == 'loglin':
            data[:, -1] = np.log(data[:, -1])
        elif self.meta_mode == 'loglog':
            data = np.log(data)
        else:
            print('unknown meta_mode', self.meta_mode)
            raise ValueError

        if self.apply_global_minus_one:
            data[:, -1] *= -1

        return data

    def get_subsample_target(self, size: int) -> 'Target':
        """
        Get a random subsample of the train data.
        Parameters: target_size (int): Size of the subsample.
        Returns: Target: Target object with subsampled data.
        """
        L = self.train_noiseless_data.shape[0]
        if size > L:
            indices = [i for i in range(L)]
        else:
            indices = np.random.randint(0, L, size = size)
        return Target(self.train_noiseless_data[indices, :-1],
                      self.train_noiseless_data[indices, -1],
                      self.train_noisy_data[indices, -1],
                      self.use_noise)

class Target():
    """
    Class representing a target object with variables and objective function.
    """
    def __init__(self, variables: np.ndarray, f: np.ndarray, f_noise: np.ndarray, use_noise: bool) -> None:
        """
        Initialize the Target object.

        Parameters:
        variables (array): Array of variables.
        f (array): Objective function values.
        f_noise (array): Objective function values with noise.
        n_var (int): Number of variables.
        ranges_var (list): Ranges of variables.
        ranges_f (float): Range of the objective function.
        with_noise (bool): Flag to indicate if noise is added to the objective function.
        """
        self.variables = variables
        self.f = f
        self.f_noise = f_noise
        self.use_noise = use_noise
        self.target_size = self.f.shape[0]
