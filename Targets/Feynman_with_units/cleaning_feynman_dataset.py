# ----------------------------------------------------------------------------------------------------
#                             this is critical for some of the targets :
# ----------------------------------------------------------------------------------------------------
# Note that our scalar placeholders is written 'A' ; no variables are allowed to have this name, or contain it
# Yet this happens in some of the targets ; we need to replace them by something else in the original files
# This concerns : A,Area,2,0,0,0,0 and A_vec,Vector potential,-1,1,0,0,1 in the units.csv file,
# Correspondingly used in the targets number I.43.43, II.2.42, II.38.3, III.21.20

# Because letter ksi is not used anywhere in the targets, we will replace A by ksi and A_vec by ksi_vec
# THIS CODE IS HIGHLY SPECIFIC TO THE FEYNMAN_WITH_UNITS DATASET FORMAT IN UNITS.CSV AND FEYNMANEQUATIONS.CSV FILES ;
# it is not meant to be reused elsewhere

# also, the FeynmanEquations.csv has some inconsistencies in the number of variables, and we need to fix it
# finally, some of the targets ar described in the FeynmanEquations.csv file, but the data is not available on the website,
# namely : ['I.15.1', 'I.48.2', 'II.11.17']

import os
import pandas as pd
from Utils.Utils_data import remove_unknown_cols_in_df
import csv
import re


def delete_cleaned_files():

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saving_path = os.path.join(root_dir, f'Feynman_with_units/units_cleaned.csv')
    if os.path.exists(saving_path):
        os.remove(saving_path)
    saving_path = os.path.join(root_dir, f'Feynman_with_units/FeynmanEquations_cleaned.csv')
    if os.path.exists(saving_path):
        os.remove(saving_path)

def handle_powers(formula):
    # our code to compute target dimension requires ** a number, but some formulas are written with **(3/2) for instance : replace with **1.5

    # Regular expression to match **(expression) where expression is a valid number
    pattern = r"\*\*\(([^)]+)\)"

    def replacer(match):
        try:
            # Evaluate the content inside the parentheses
            evaluated_value = eval(match.group(1))
            # Return the evaluated value as a string
            return f"**{evaluated_value}"
        except Exception:
            # If evaluation fails, return the original match
            return match.group(0)

    # Substitute matches in the expression with evaluated values
    return re.sub(pattern, replacer, formula)


def clean_variable_names_feynman_dataset():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loading_path = os.path.join(root_dir, f'Feynman_with_units/units.csv')
    saving_path = os.path.join(root_dir, f'Feynman_with_units/units_cleaned.csv')

    if not os.path.exists(saving_path):
        try:
            df = pd.read_csv(loading_path, sep=',')
        except Exception as e:
            raise ValueError(f"Error loading data from {loading_path}: {e}")

        df = remove_unknown_cols_in_df(df).dropna()

        # Assert that 'ksi' and 'ksi_vec' are not already in the 'Variable' column
        assert 'ksi' not in df['Variable'].values, "'ksi' already exists in the Variable column."
        assert 'ksi_vec' not in df['Variable'].values, "'ksi_vec' already exists in the Variable column."

        # Replace 'A' and 'A_vec' by 'ksi' and 'ksi_vec' respectively
        df['Variable'] = df['Variable'].replace({'A': 'ksi', 'A_vec': 'ksi_vec'})
        df.to_csv(saving_path, index=False)

def clean_equation_list_feynman_dataset():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loading_path = os.path.join(root_dir, f'Feynman_with_units/FeynmanEquations.csv')
    saving_path = os.path.join(root_dir, f'Feynman_with_units/FeynmanEquations_cleaned.csv')


    remove_targets = ['I.15.1', 'I.48.2', 'II.11.17']

    with open(loading_path, 'r') as f:
        reader = csv.reader(f)
        rows = []

        for i, line in enumerate(reader):
            # print(line)
            # ['\ufeffFilename', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', 'v1_high', 'v2_name', 'v2_low', 'v2_high', 'v3_name', 'v3_low', 'v3_high', 'v4_name', 'v4_low', 'v4_high', 'v5_name', 'v5_low', 'v5_high', 'v6_name', 'v6_low', 'v6_high', 'v7_name', 'v7_low', 'v7_high', 'v8_name', 'v8_low', 'v8_high', 'v9_name', 'v9_low', 'v9_high', 'v10_name', 'v10_low', 'v10_high']
            # lets only keep filename, Formula, # variables, v1_name, v2_name, v3_name, v4_name, v5_name, v6_name, v7_name, v8_name, v9_name, v10_name
            if i == 0:  # change the header
                # lets only keep filename, Formula, # variables, v1_name, v2_name, v3_name, v4_name, v5_name, v6_name, v7_name, v8_name, v9_name, v10_name
                nl = [line[0], line[3], line[4]] + line[5:]
                rows.append(nl)
                continue

            if not line[0]:  # Skip empty lines
                continue

            # Parse the equation details
            equation_label = line[0]  # str
            if equation_label in remove_targets:
                continue

            formula = line[3] if line[3] else None  # str or None

            # replace 'A' and 'A_vec' by 'ksi' and 'ksi_vec' respectively
            if formula:
                formula = formula.replace('A_vec', 'ksi_vec')
                formula = formula.replace('A', 'ksi')
                line[3] = formula
            formula = handle_powers(formula)

            n_var = eval(line[4])  # Convert to int
            variables_list = [line[k] for k in range(5, len(line), 3) if line[k]]
            # replace 'A' and 'A_vec' by 'ksi' and 'ksi_vec' respectively in the variables list
            variables_list = [var.replace('A_vec', 'ksi_vec').replace('A', 'ksi') for var in variables_list]
            tmp = variables_list.copy()
            for k in range(5, len(line), 3):
                if line[k]:
                    line[k] = tmp.pop(0)

            # Autocorrect variable count if necessary
            if len(variables_list) != n_var:
                print(
                    f'Bug in description file {equation_label}: num var is {n_var} but var names are {variables_list}')
                print('Autocorrecting...')
                n_var = len(variables_list)
                line[4] = str(n_var)

            # Add the modified line to the rows list
            # lets only keep filename, Formula, # variables, v1_name, v2_name, v3_name, v4_name, v5_name, v6_name, v7_name, v8_name, v9_name, v10_name
            nl = [line[0], line[3], line[4]] + line[5:]
            rows.append(nl)

    with open(saving_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(rows)

def clean_equation_list_bonus_equations():
    # same but files are a bit different, and then append to the same file
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loading_path = os.path.join(root_dir, f'Feynman_with_units/BonusEquations.csv')
    saving_path = os.path.join(root_dir, f'Feynman_with_units/FeynmanEquations_cleaned.csv')

    remove_targets = []

    with open(loading_path, 'r') as f:
        reader = csv.reader(f)
        rows = []

        for i, line in enumerate(reader):
            # print(line)
            # ['\ufeffFilename', 'Number', 'Name', 'Eqn. No.', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', 'v1_high', 'v2_name', 'v2_low', 'v2_high', 'v3_name', 'v3_low', 'v3_high', 'v4_name', 'v4_low', 'v4_high', 'v5_name', 'v5_low', 'v5_high', 'v6_name', 'v6_low', 'v6_high', 'v7_name', 'v7_low', 'v7_high', 'v8_name', 'v8_low', 'v8_high', 'v9_name', 'v9_low', 'v9_high', 'v10_name', 'v10_low', 'v10_high']
            # lets only keep filename, Formula, # variables, v1_name, v2_name, v3_name, v4_name, v5_name, v6_name, v7_name, v8_name, v9_name, v10_name
            if i == 0:  # skip  the header
                continue

            if not line[0]:  # Skip empty lines
                continue

            # Parse the equation details
            equation_label = line[0]  # str
            if equation_label in remove_targets:
                continue

            formula = line[5] if line[5] else None  # str or None
            formula = handle_powers(formula)
            # replace 'A' and 'A_vec' by 'ksi' and 'ksi_vec' respectively
            if formula:
                formula = formula.replace('A_vec', 'ksi_vec')
                formula = formula.replace('A', 'ksi')
                line[5] = formula

            n_var = eval(line[6])  # Convert to int
            variables_list = [line[k] for k in range(7, len(line), 3) if line[k]]
            # replace 'A' and 'A_vec' by 'ksi' and 'ksi_vec' respectively in the variables list
            variables_list = [var.replace('A_vec', 'ksi_vec').replace('A', 'ksi') for var in variables_list]
            tmp = variables_list.copy()
            for k in range(7, len(line), 3):
                if line[k]:
                    line[k] = tmp.pop(0)

            # Autocorrect variable count if necessary
            if len(variables_list) != n_var:
                print(
                    f'Bug in description file {equation_label}: num var is {n_var} but var names are {variables_list}')
                print('Autocorrecting...')
                n_var = len(variables_list)
                line[6] = str(n_var)

            # Add the modified line to the rows list
            # Filename,Number,Output,Formula,# variables,v1_name,v1_low,v1_high,v2_name,v2_low,v2_high,v3_name,v3_low,v3_high,v4_name,v4_low,v4_high,v5_name,v5_low,v5_high,v6_name,v6_low,v6_high,v7_name,v7_low,v7_high,v8_name,v8_low,v8_high,v9_name,v9_low,v9_high,v10_name,v10_low,v10_high
            nl = [line[0], line[5], line[6]] + line[7:]
            rows.append(nl)

    with open(saving_path, 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(rows)
