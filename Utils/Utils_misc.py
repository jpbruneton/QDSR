import pandas as pd

def generate_lists(inf, sup, size):
    """Generate all possible lists of size 'size' with values between inf and sup."""
    result = []

    def nested_loops(current_list, current_depth):
        if current_depth == size:
            result.append(current_list.copy())
            return

        for i in range(inf, sup + 1):
            current_list[current_depth] = i
            nested_loops(current_list, current_depth + 1)

    nested_loops([0] * size, 0)
    return result


def remove_unknown_cols_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with 'Unnamed' in their names from a pandas DataFrame in place.
    Args:
        df (pd.DataFrame): The input DataFrame from which to remove invalid columns.

    Returns:
        pd.DataFrame: The updated DataFrame with 'Unnamed' columns removed.
    """
    cols = df.columns
    unvalid_columns = []
    for k in range(df.shape[1]):
        if 'Unnamed' in cols[k]:
            unvalid_columns.append(k)
        else:
            pass
    df.drop(df.columns[unvalid_columns], axis=1, inplace=True)
    return df

def matsumoto_breakdown():
    easy = ["I.30.5", "I.43.16", "I.47.23", "II.2.42", "II.3.24", "II.4.23", "II.8.31", "II.10.9",
            "II.13.17", "II.15.4", "II.15.5", "II.27.16", "II.27.18", "II.34.11", "II.34.29b",
            "II.38.3", "II.38.14", "III.7.38", "III.12.43", "III.15.27"]
    medium = [
        'I.8.14', 'I.10.7', 'I.11.19', 'I.12.2', 'I.12.11', 'I.13.4', 'I.13.12', 'I.15.1',
        'I.16.6', 'I.18.4', 'I.24.6', 'I.29.4', 'I.32.5', 'I.34.8', 'I.34.1', 'I.34.27',
        'I.38.12', 'I.39.1', 'I.39.11', 'I.43.31', 'I.43.43', 'I.48.2', 'II.6.11', 'II.8.7',
        'II.11.3', 'II.21.32', 'II.34.2', 'II.34.2a', 'II.34.29a', 'II.37.1', 'III.4.32',
        'III.8.54', 'III.13.18', 'III.14.14', 'III.15.12', 'III.15.14', 'III.17.37', 'III.19.51',
        'test_8', 'test_18'
    ]
    hard = ['I.6.2', 'I.6.2a', 'I.6.2b', 'I.9.18', 'I.15.3t', 'I.15.3x', 'I.29.16', 'I.30.3', 'I.32.17', 'I.34.14', 'I.37.4', 'I.39.22', 'I.40.1', 'I.41.16', 'I.44.4', 'I.50.26', 'II.6.15a', 'II.6.15b', 'II.11.17', 'II.11.20', 'II.11.27', 'II.11.28', 'II.13.23', 'II.13.34', 'II.24.17', 'II.35.18', 'II.35.21', 'II.36.38', 'III.4.33', 'III.9.52', 'III.10.19', 'III.21.20', 'test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_9', 'test_10', 'test_11', 'test_12', 'test_13', 'test_14', 'test_15', 'test_16', 'test_17', 'test_19', 'test_20']

    #print(len(easy), len(medium), len(hard)) ; 20;40;50
    return easy, medium, hard
def load_equation_list():
    """
    Loads the list of equations to solve, excluding dimensionally trivial and found equations.
    """
    dimensionally_trivial = ['I.12.4', 'I.12.5', 'I.14.3', 'I.14.4', 'I.25.13', 'I.29.4', 'I.32.5', 'I.34.8',
                             'I.34.27', 'I.38.12', 'I.39.1', 'I.43.16', 'I.43.31', 'II.3.24', 'II.4.23', 'II.8.7',
                             'II.8.31', 'II.13.17', 'II.27.16', 'II.27.18', 'II.34.2a', 'II.34.2', 'II.34.29a',
                             'III.7.38', 'III.15.14', 'III.21.20']

    very_easy = ['I.8.14', 'I.11.19', 'I.12.1', 'I.12.2', 'I.13.4', 'I.13.12', 'I.18.12', 'I.18.14', 'I.24.6', 'I.34.1', 'I.39.11', 'I.39.22', 'I.43.43', 'I.47.23', 'II.6.11', 'II.10.9', 'II.11.20', 'II.15.4', 'II.15.5', 'II.34.11', 'II.34.29b', 'II.35.21', 'II.37.1', 'II.38.14', 'III.10.19', 'III.12.43', 'III.13.18', 'III.14.14', 'III.15.27', 'III.19.51', 'I.9.18', 'I.27.6', 'I.44.4', 'II.11.3', 'II.21.32', 'II.38.3', 'I.34.14', 'I.6.2a', 'I.12.11', 'II.2.42', 'III.15.12', 'II.13.23', 'II.13.34', 'III.17.37', 'II.6.15b', 'I.40.1', 'II.11.27', 'I.18.4', 'III.4.32']
    easy = ['II.36.38', 'I.10.7', 'II.24.17', 'III.4.33', 'I.15.3t', 'I.15.3x', 'I.41.16', 'II.35.18', 'I.50.26', 'II.11.28', 'I.16.6', 'III.8.54', 'I.32.17', 'I.37.4', 'I.6.2', 'I.30.3', 'I.30.5', 'I.26.2']
    medium = []
    hard = []
    not_found = ['III.9.52', 'I.6.2b', 'I.29.16', 'II.6.15a']
    bonus = [f'test_{i}' for i in range(1, 21)]
    missing_data = ['I.15.1', 'I.48.2', 'II.11.17']
    return dimensionally_trivial, very_easy, easy, medium, hard, not_found, bonus
