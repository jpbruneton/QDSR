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

def load_equation_list():
    """
    Loads the list of equations to solve, excluding dimensionally trivial and found equations.
    """
    dimensionally_trivial = ['I.12.4', 'I.12.5', 'I.14.3', 'I.14.4', 'I.25.13', 'I.29.4', 'I.32.5', 'I.34.8',
                             'I.34.27', 'I.38.12', 'I.39.1', 'I.43.16', 'I.43.31', 'II.3.24', 'II.4.23', 'II.8.7',
                             'II.8.31', 'II.13.17', 'II.27.16', 'II.27.18', 'II.34.2a', 'II.34.2', 'II.34.29a',
                             'III.7.38', 'III.15.14', 'III.21.20']

    very_easy = [
    "I.11.19", "I.12.1", "I.12.2", "I.13.12", "I.18.12",
    "I.18.14", "I.39.22", "II.6.11", "II.10.9", "II.11.20",
    "II.15.4", "II.15.5", "II.37.1", "II.38.14", "III.12.43",
    "III.13.18", "III.15.27", "I.27.6", "II.38.3",
    "I.12.11", "II.2.42", "III.15.12"]

    easy_table = [
    "test_5", "test_18", "I.34.14", "II.11.3", "I.43.43", "I.47.23", "I.34.1", "I.39.11",
    "III.19.51", "II.34.11", "II.34.29b", "I.40.1", "II.13.23", "II.13.34",
    "I.10.7", "II.6.15b", "II.11.27", "II.36.38", "I.50.26", "III.17.37",
    "I.18.4", "test_9", "I.8.14", "I.13.4", "I.24.6",
    "II.35.21", "III.10.19", "I.9.18", "I.44.4", "II.21.32",
    "III.14.14", "I.32.17", "II.11.28", "I.6.2a", "II.35.18"
    ]


    medium_table = [
    "III.8.54", "I.15.3x", "I.30.5", "test_19", "I.15.3t",
    "test_7", "test_17", "test_8", "I.6.2", "I.30.3",
    "I.37.4", "test_6", "I.16.6", "test_4", "I.26.2",
    "test_1", "test_2", "II.6.15a", "test_14", "test_3"
    ]
    hard_table = [
    "III.4.32", "III.4.33", "I.6.2b", "II.24.17", "I.41.16",
    "test_12", "test_15", "test_16", "test_13", "I.29.16",
    "III.9.52", "test_20", "test_10", "test_11"
    ]
    #missing_data = ['I.15.1', 'I.48.2', 'II.11.17']
    return dimensionally_trivial, very_easy, easy_table, medium_table, hard_table
