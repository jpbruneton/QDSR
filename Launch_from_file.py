from config import config
from Utils import Utils_io, Utils_ground_truth

from config.config import get_config
import os
import pandas as pd

# Example file where you want to run the solver on other targets than Feynmann AI.

###################  CASE II : give yourself a file with data and run the solver on it;
# say the file is in Targets/example_from_file/raw_data/sample_data
# in this example, the data was generated from formula 3.01*sqrt(x**2 + y**2) -1.21*x*x/y, on 50k points)
# but lets imagine we dont know the formula, and we want to discover it
# with its units file in Targets/example_from_file/units.csv
# with units.csv :
# Variable,Units,m,s,kg,T,V,
# x,Length,1,0,0,0,0,
# y,Length,1,0,0,0,0,
# remember the target must be in the last column and variables in the first columns


########### initialize everything ###########

# Step 1 : name accordingly the dataset and the equation label
dataset_name = 'example_from_file'
equation_label = 'sample_data'

# Step 2 : get config / edit as you like in config/config.py
search_intensity = 'custom'
cfg = get_config(search_intensity)

# if you dont have a unit file, you can create it like this:
# should be a pandas file with the following format
# Variable,Units,m,s,kg,T,V,
# x,Length,1,0,0,0,0,
# y,Length,1,0,0,0,0,

units = ['m', 's', 'kg', 'T', 'V']
unit_dict = {'x': {'m': 1, 's': 0, 'kg': 0, 'T': 0, 'V': 0},
             'y': {'m': 1, 's': 0, 'kg': 0, 'T': 0, 'V': 0}}
target_dimension = [1, 0, 0, 0, 0] #you should also know this one
unit_description = {'x': 'Length', 'y': 'Length'}
true_name_variables = ['x', 'y']
n_var = len(true_name_variables)
df = pd.DataFrame(columns=['Variable', 'Units'] + units)
for var in true_name_variables:
    row = [var, unit_description[var]] + [unit_dict[var][unit] for unit in units]
    df.loc[len(df)] = row
df.to_csv(f'Targets/{dataset_name}/units.csv', index=False)

#then compute the units dictionary
if not os.path.exists(f'Targets/{dataset_name}/unit_dict.pkl'):
    Utils_ground_truth.Build_units_dict(dataset_name)

# Step 3 : create a equation descriptor:
# given our data from file we need to create a equation descriptor with Feynman Ai style:
# Filename,Formula,# variables,v1_name,v1_low,v1_high,v2_name,v2_low,v2_high,v3_name,v3_low,v3_high,v4_name,v4_low,v4_high,v5_name,v5_low,v5_high,v6_name,v6_low,v6_high,v7_name,v7_low,v7_high,v8_name,v8_low,v8_high,v9_name,v9_low,v9_high,v10_name,v10_low,v10_high
# I.6.2a,exp(-theta**2/2)/sqrt(2*pi),1,theta,1,3,,,,,,,,,,,,,,,,,,,,,,,,,,,

Formula = None #unknown
vars = list(unit_dict.keys())

with open(f'Targets/{dataset_name}/FeynmanEquations.csv', 'w') as f:
    header_line = ["Filename", "Formula", "# variables"]
    for i in range(1, n_var + 1):
        header_line.extend([f"v{i}_name", f"v{i}_low", f"v{i}_high"])
    f.write(','.join(header_line) + '\n')
    second_line = f'{equation_label},{Formula},{n_var},'
    for i in range(n_var):
        second_line += f'{vars[i]},{0},{1},' #whatever the actual range is
    f.write(second_line + '\n')
Utils_ground_truth.Build_Feynmann_Formula_Dict(dataset_name, config.fundamental_units, cfg)

# print the ground truth dictionary
ground_truth = Utils_io.load_ground_truth(dataset_name, equation_label)
ground_truth['target_dimension'] = target_dimension
import pickle
file_path = f'Targets/{dataset_name}/formula_dict_with_units'
with open(file_path, 'rb') as f:
    formula_dict = pickle.load(f)
formula_dict[equation_label] = ground_truth
with open(file_path, 'wb') as f:
    pickle.dump(formula_dict, f)

for k, v in ground_truth.items():
    print(f'{k}: {v}')
# we are all set
# should print :
# formula: None
# n_variables: 2
# variables_true_name: ['x', 'y']
# variables_internal_name: ['x0', 'x1']
# dimension_list: [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
# dimension_of_variables: {'x0': [1, 0, 0, 0, 0], 'x1': [1, 0, 0, 0, 0]}
# internal_name_to_true_name_dict: {'x0': 'x', 'x1': 'y'}
# true_name_to_internal_name_dict: {'x': 'x0', 'y': 'x1'}
# target_dimension: [1, 0, 0, 0, 0]
# is_target_everywhere_positive: None
# is_everything_dimensionless: False
# is_equation_possible: True
# is_equation_trivial: False
# eigenvectors_with_target: [array([1., 0., 0.])]
# eigenvectors_no_target: [array([ 0., -1.,  1.])]
# apply_global_minus_one: None
# adimensional_variables: ['y0', 'y1']
# adimensional_dict: {'y0': '(x0)/((x1))', 'y1': '(x1)/((x0))'}
# norms: {'norm0': '(x0)**2 + (x1)**2', 'norm1': 'x0**2 - x1**2'}
# norms_dim: {'norm0': [2, 0, 0, 0, 0], 'norm1': [2, 0, 0, 0, 0]}


# Step 4 : run the solver
from Launch_Feynman import main

multi_processing = True
n_cores = config.default_n_cores(0.9)
run_number = 0
recompute = False
task = (dataset_name, cfg, equation_label, multi_processing, n_cores, run_number, recompute)
main(task)
# the exact recovery engine is not gonna work here because the ground truth formula is unknown ; todo : add a flag

