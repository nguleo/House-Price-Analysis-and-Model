
import pandas as pd
import numpy as np



def identify_inconsistencies(houseprice_df):

    '''This function will check if a column is categorical and
    print the unique values in the column
    Args:
    data: a pandas dataframe
    Prints:
    The unique values in each categorical column    
    '''

    for column in houseprice_df.columns:
        print(f'{column}: {houseprice_df[column].nunique()} unique values')
        print(houseprice_df[column].unique())
    return houseprice_df


def standardize_categories(houseprice_df):
    categorical_columns = houseprice_df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        houseprice_df[column] = houseprice_df[column].str.upper().str.strip()
    return houseprice_df


def handle_inconsistencies(houseprice_df):

    '''This function will handle inconsitencies in the categorical column and print the unique values in the colunn'''

    for column in houseprice_df.columns:
     print(f"Unique values in '{column}':")
    print(houseprice_df[column].unique())

    return houseprice_df

def check_missing_values(houseprice_df):
    '''This function will check for missing values in the data
    Args:
    data: a pandas dataframe
    Returns:
    missing_values: a pandas series with the number of missing values in each column
    '''
    print('---Checking for missing values in the data---')
    print('Number of missing values in each column:')
    print(houseprice_df.isnull().sum())

def impute_missing_values(houseprice_df):
    '''This function will impute missing values in numerical columns with -1
    and in categorical columns with 'Unknown'
    Args:
    data: a pandas dataframe
    Returns:
    data: a pandas dataframe with missing values imputed
    '''
    print('---Imputing missing values in the data---')
    for column in houseprice_df:
        if houseprice_df[column].dtype == 'object':
            houseprice_df[column] = houseprice_df[column].fillna('UNKNOWN')
        else: 
            houseprice_df[column] = houseprice_df[column].fillna(-1)
    
    return houseprice_df


def check_duplicate_columns(houseprice_df):
    '''This function will check if there are any duplicate columns in the data
    Args:
    data: a pandas dataframe
    Returns:
    duplicate_columns: a list of duplicate columns
    '''
    duplicate_columns = []
    for x in range(houseprice_df.shape[1]):
        col = houseprice_df.iloc[:, x]
        for y in range(x + 1, houseprice_df.shape[1]):
            other_col = houseprice_df.iloc[:, y]
            if col.equals(other_col):
                duplicate_columns.append(houseprice_df.columns.values[y])
    return duplicate_columns


def check_duplicate_rows(houseprice_df):
    '''This function will check if there are any duplicate rows in the data
    Args:
    data: a pandas dataframe
    Returns:
    duplicate_rows: a pandas series with True for duplicate rows and False otherwise
    '''
    print('---Checking for duplicate rows in the data---')
    print("Number of duplicate rows:", houseprice_df.duplicated().sum())

def check_partial_duplicates(houseprice_df):
    '''This function will check if there are any partial duplicates in the data
    Args:
    data: a pandas dataframe
    Returns:
    partial_duplicates: a pandas series with True for partial duplicates and False otherwise
    '''
    print('---Checking for partial duplicates in the data---')
    print("Number of partial duplicate rows:",
          houseprice_df.duplicated(subset=houseprice_df.columns.difference(['SalePrice'])).sum())

def detect_outliers(houseprice_df):
    '''This function will detect outliers using the IQR method
    Args:
    data: a pandas dataframe
    Prints:
    names of columns with outliers, if any
    the lower and upper bounds of the outliers
    '''
    print('---Detecting outliers in the data---')
    outliers = []
    for column in houseprice_df.columns:
        if houseprice_df[column].dtype != 'object':
            Q1 = houseprice_df[column].quantile(0.25)
            Q3 = houseprice_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if IQR == 0:
                continue
            elif (houseprice_df[column].min() < lower_bound) | (houseprice_df[column].max() > upper_bound):
                print(f'Outliers detected in {column}')
                print(f'Lower bound: {lower_bound}, Upper bound: {upper_bound}')
                outliers.append(column)             
    print('Columns with outliers:', outliers)
    return outliers


def handle_outliers(houseprice_df, outlier_cols):
    '''This function will handle outliers by capping them to the lower and upper bounds
    Args:
    data: a pandas dataframe
    outlier_cols: a list of columns with outliers
    Returns:
    cleaned_data: a pandas dataframe with outliers handled
    '''
    print('---Handling outliers by replacing outliers with -1---')
    for column in outlier_cols:
        print(column)
        Q1 = houseprice_df[column].quantile(0.25)
        Q3 = houseprice_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Handling outliers by replacing outliers with -1
        houseprice_df[column] = np.where(houseprice_df[column] < lower_bound, -1, houseprice_df[column])
        houseprice_df[column] = np.where(houseprice_df[column] > upper_bound, -1, houseprice_df[column])

    return houseprice_df


def identify_highly_correlated_features(houseprice_df):
    '''This function will identify highly correlated features in the data
    Args:
    data: a pandas dataframe
    Prints:
    The highly correlated features
    '''
    # numerical columns
    print('---Identifying highly correlated features in the data---')
    num_cols = houseprice_df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = houseprice_df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    print("Highly correlated columns:", to_drop)
    return to_drop


def data_cleaning(houseprice_df):
    '''This function will clean the data by removing any missing values
    Args:
    data: a pandas dataframe
    Returns:
    cleaned_data: a pandas dataframe without missing values
    '''
    # 1. Check for inconsistencies
    identify_inconsistencies(houseprice_df)

    # 2. Handle inconsistencies
    cleaned_data = standardize_categories(houseprice_df)
    cleaned_data = handle_inconsistencies(houseprice_df)

    # 3. Check and handle missing values
    check_missing_values(cleaned_data)
    cleaned_data = impute_missing_values(houseprice_df)
    

    # 4. Check and handle for duplicat columns
    duplicate_cols = check_duplicate_columns(cleaned_data)
    cleaned_data = cleaned_data.drop(columns=duplicate_cols)

    # 5. Check and handle duplicate rows
    check_duplicate_rows(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates()

    # 6. Check and handle partial duplicates
    check_partial_duplicates(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates(subset=cleaned_data.columns.difference(['SalePrice']))
    check_partial_duplicates(cleaned_data)

    # 7. Check and handle for outliers
    #outlier_cols = detect_outliers(cleaned_data)
    #cleaned_data = handle_outliers(cleaned_data, outlier_cols)

    # 7. Check for highly correlated features
    #corr_cols = identify_highly_correlated_features(cleaned_data)
    #cleaned_data = cleaned_data.drop(columns=corr_cols)

    return cleaned_data


def main():
    housepice_df = pd.read_csv('HousePricesDataSet.csv')
    cleaned_data = data_cleaning(housepice_df)
    cleaned_data.to_csv('cleaned_HousePricesDataSet2.csv', index=False)

# call the main function
main()