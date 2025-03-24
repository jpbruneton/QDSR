from config import config
import pickle
import numpy as np


def exceeds_loss_threshold(loss, metric, threshold):
    '''Check if the loss exceeds the bad loss threshold.'''
    if metric == 'NRMSE':
        return loss > threshold
    if metric == 'R2':
        return loss < threshold

class QD():

    def __init__(self, maximal_length, qd_grid_size, use_noise, logger, initial_qd_grid = None):
        """
        This module provides the main functionalities for Quality-Diversity Genetic Programming.

        Main Functionalities:
        - **Initialize the QD_class:** Define the quality-diversity grid and binning configuration.
        - **Pool Management:** Extend the pool of equations by generating offspring (via mutations and crossovers)
          and adding random equations.
        - **Binning:** Project results into bins based on equation features, ensuring a diverse representation of solutions.
        - **Grid Updates:** Update the quality-diversity grid with new solutions, replacing previous entries if the
          new solution is superior in terms of loss.
        - **Storage and Retrieval:** Save and load the QD grid, facilitating continuity across runs.

        Parameters:
        maximal_length (int): The maximal length of the equation.
        qd_grid (dict): The quality diversity grid.
        cfg (dict): Configuration parameters.
        logger (Logger): Logger for logging information.
        """
        self.maximal_length = maximal_length
        self.qd_grid = {} if initial_qd_grid is None else initial_qd_grid
        self.use_noise = use_noise
        self.logger = logger
        self.critical_features = ["equation_length", "num_scalars"]
        self.features = ["num_functions", "num_variables"]
        self.detailed_features = ["num_nested_functions"]
        self.very_detailed_features = ["num_powers", "num_trig_functions"]
        self.all_features = self.critical_features + self.features + self.detailed_features + self.very_detailed_features
        # Initialize the quality-diversity bin attributes
        if qd_grid_size == 'small':
            self.define_small_grid()
        elif qd_grid_size == 'medium':
            self.define_medium_grid()
        elif qd_grid_size == 'large':
            self.define_large_grid()
        elif qd_grid_size == 'very_large':
            self.define_very_large_grid()

    def define_small_grid(self):
        '''Configure bin attributes for a small qd_grid.'''
        self.define_bins(self.maximal_length, 1,
                         0, 0,
                         0, 0,
                         0, 0)

    def define_medium_grid(self):
        '''Configure bin attributes for a medium qd_grid.'''
        self.define_bins(self.maximal_length, 1,
                         self.maximal_length, 2,
                         0, 0,
                         0, 0)

    def define_large_grid(self):
        '''Configure bin attributes for a medium qd_grid.'''
        self.define_bins(self.maximal_length, 1,
                         self.maximal_length, 2,
                         2, 1,
                         0, 0)

    def define_very_large_grid(self):
        '''Configure bin attributes for a large qd_grid.'''
        self.define_bins(self.maximal_length, 1,
                         self.maximal_length, 1,
                         self.maximal_length, 1,
                         self.maximal_length, 1)

    def define_bins(self, max_bins_critical, step_critical, max_bins, step, max_detailed_bins, step_detailed,
                    max_very_detailed_bins, step_very_detailed):
        """Set bin attributes based on provided maximal values and steps for binning."""
        for key in self.critical_features:
            setattr(self, f"max_bin_{key}", max_bins_critical)
            setattr(self, f"step_bin_{key}", step_critical)
        for key in self.features:
            setattr(self, f"max_bin_{key}", max_bins)
            setattr(self, f"step_bin_{key}", step)
        for key in self.detailed_features:
            setattr(self, f"max_bin_{key}", max_detailed_bins)
            setattr(self, f"step_bin_{key}", step_detailed)
        for key in self.very_detailed_features:
            setattr(self, f"max_bin_{key}", max_very_detailed_bins)
            setattr(self, f"step_bin_{key}", step_very_detailed)


    ########## qd_grid operations ##########
    def _bin(self, value : int, maximum : int, step : int) -> int:
        """
        Get the bin index for a given value.

        Parameters:
        value (int): The value to be binned.
        maximum (int): The maximum value for the bin.
        step (int): The step size for the bin.

        Returns:
        int: The bin index.
        """
        if value >= maximum:
            return maximum
        bins = [i for i in range(0, maximum + step, step)]
        for i in range(len(bins) - 1):
            if value >= bins[i] and value < bins[i + 1]:
                return i
        raise ValueError("Value out of bin range")

    def get_bin_identifier(self, result : dict) -> str:
        """
        Get the bin ID for a given result.

        Parameters:
        result (dict): The result dictionnary of an equation, containing in particular its features to be binned.

        Returns:
        str: The bin ID.
        this is a string representation of the bin identifier, of teh form
        """
        bin_identifier = []
        for feature in self.all_features:
            if feature not in result:
                raise ValueError(f"Feature {feature} not in result")
            bin_identifier.append(self._bin(result[feature], getattr(self, f"max_bin_{feature}"), getattr(self, f"step_bin_{feature}")))

        return str(bin_identifier)

    def project_results_in_bins(self, results):
        """
        Bin the results coming from the evaluation of the equations.
        If multiple results are in the same bin, keep the one with the best loss.

        Parameters:
        results (list): List of results to be binned.

        Returns:
        dict: Results binned by their characteristics.
        """
        binned_results = {}
        for result in results:
            bin_identifier = self.get_bin_identifier(result) # is a string
            if bin_identifier not in binned_results:
                binned_results[bin_identifier] = result
            else:
                previous_element = binned_results[bin_identifier]
                previous_loss = previous_element['loss_noise'] if self.use_noise else previous_element['loss_no_noise']
                new_loss = result['loss_noise'] if self.use_noise else result['loss_no_noise']
                if config.metric == 'NRMSE' and new_loss < previous_loss:
                    binned_results[bin_identifier] = result
                elif config.metric == 'R2' and new_loss > previous_loss:
                    binned_results[bin_identifier] = result
        return binned_results



    def update_qd_grid(self, new_binned_results):
        """
        Update the quality diversity grid with new equations: replace previous elements if a better loss is found
        or add new element if their characteristics are not already in the grid

        Parameters:
        new_binned_results (dict): New results binned by their characteristics, of the form {'bin_identifier': result}.
        where bin_identifier is the string representation of the features bins, and result is a dict containing:
        {'loss_noise','loss_no_noise', 'tree', 'num_scalars', 'scalar_values','equation_length','num_variables',
        'num_functions', 'num_powers','num_trig_functions',
        'num_plus_operator', 'num_minus_operator', 'num_times_operator', 'num_divide_operator',
        'num_nested_functions'}

        Returns:
        tuple: Number of new bins and replacements.
        """
        new_bins = 0
        replacements = 0
        too_bad_loss_count = 0
        new_evaluations = 0

        for bin_identifier, result in new_binned_results.items():
            if self.use_noise:
                new_loss = result['loss_noise']
            else:
                new_loss = result['loss_no_noise']

            # first case: the bin is new : add it if the loss is not too bad
            if bin_identifier not in self.qd_grid:
                if (np.isfinite(new_loss)) and (not np.iscomplex(new_loss)) and (not np.isnan(new_loss)):
                    if not exceeds_loss_threshold(new_loss, config.metric, config.max_bad_loss):
                        self.qd_grid[bin_identifier] = result
                        new_bins += 1
                else: # if you don't want too bad loss in your qd_grid
                    too_bad_loss_count += 1

            else: # this bin is already in the qd_grid
                # two cases : it is the same exact state or it is a different state
                previous_tree = self.qd_grid[bin_identifier]['tree']
                new_tree = result['tree']
                if new_tree.postfix_formula == previous_tree.postfix_formula: # it is the exact same equation
                    # replace the loss (since it changes on another subsample of the full target)
                    self.qd_grid[bin_identifier] = result  # this is required because the loss will change on another subsample of the full target
                    new_evaluations += 1
                else:
                    if self.use_noise:
                        previous_loss = self.qd_grid[bin_identifier]['loss_noise']
                    else:
                        previous_loss = self.qd_grid[bin_identifier]['loss_no_noise']
                    if config.metric == 'NRMSE' and new_loss < previous_loss: # update the bin iff the loss is better
                        self.qd_grid[bin_identifier] = result
                        replacements += 1
                    elif config.metric == 'R2' and new_loss > previous_loss:
                        self.qd_grid[bin_identifier] = result
                        replacements += 1

        self.logger.details(f'Ignoring {too_bad_loss_count} individuals with too bad losses')
        self.logger.details(f'New bins: {new_bins}, Replacements: {replacements}, New evaluations: {new_evaluations}')


    ########## misc ##########
    def return_all_trees(self):
        return [self.qd_grid[str(bin_identifier)]['tree'] for bin_identifier in self.qd_grid]

    def return_all_results(self):
        return [self.qd_grid[str(bin_identifier)] for bin_identifier in self.qd_grid]

    def save(self, output_directory, vocabulary, cfg):
        qd_object = {'configuration': cfg, 'vocabulary' : vocabulary, 'qd_grid': self.qd_grid}
        path = output_directory + '/qd_grid.pkl'
        self.logger.details(f'Saving qd_grid to {path}')
        with open(path, 'wb') as f:
            pickle.dump(qd_object, f)
            f.close()

    def load(self, output_directory):
        path = output_directory + '/qd_grid.pkl'
        with open(path, 'rb') as f:
            qd_object = pickle.load(f)
            f.close()
        return qd_object