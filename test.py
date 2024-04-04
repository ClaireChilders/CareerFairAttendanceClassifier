import pandas as pd

column_names = ['test_1', 'test_2', 'test_3']
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

test_df = pd.DataFrame(data, columns=column_names)

check_values = pd.DataFrame([[1, 2, 3]], columns=column_names)

test_df['has_value'] = test_df.loc[:, column_names].isin(
    check_values).any(axis=1)

print(test_df)
