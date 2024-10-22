import pandas as pd
import numpy as np

def import_dataset(in_filename, norm_mode):
    """
    Import and preprocess dataset.
    """
    df = pd.read_csv(in_filename, sep=',')
    df.loc[df["month_fp_whocvd"] == 0, "month_fp_whocvd"] = 0.001
    df.reset_index(drop=True, inplace=True)
    df = df[['smoke', 'sbp', 'tc', 'hdlc', 'age_enter', 'sex', 'base_dm', 'who_cvd', 'month_fp_whocvd']]

    # Get dummies for 'smoke'
    df = pd.concat([df, pd.gein_t_dummies(df["smoke"], prefix='smoke', drop_first=True)], axis=1).drop("smoke", axis=1)

    # Split dataset
    np.random.seed(1234)
    df_test = df.sample(frac=0.2)
    df_train_val = df.drop(df_test.index)
    np.random.seed(1234)
    df_val = df_train_val.sample(frac=0.2)
    df_train = df_train_val.drop(df_val.index)
   
    # Normalize datasets
    rdf_train, rdf_val, rdf_test = center_df(df_train, df_train, norm_mode), center_df(df_val, df_train, norm_mode), center_df(df_test, df_train, norm_mode)

    return rdf_train, rdf_val, rdf_test

def center_df(df, df_train, norm_mode=None):
    """
    Normalize df using df_train's mean, based on the specified normalization mode.
    """
    continuous_vars = ['sbp', 'tc', 'hdlc', 'age_enter']
    binary_vars = ['smoke_1', 'smoke_2', 'sex', 'base_dm', 'who_cvd', 'month_fp_whocvd']

    if norm_mode == 'center':
        # Calculate mean of continuous variables from df_train
        train_mean = df_train[continuous_vars].mean()
        df[continuous_vars] = df[continuous_vars] - train_mean

    elif norm_mode == 'center_sex':
        # Normalize continuous variables using the mean by sex from df_train
        for sex in [0, 1]:
            mean_sex = df_train[df_train['sex'] == sex][continuous_vars].mean()
            df.loc[df['sex'] == sex, continuous_vars] = df[df['sex'] == sex][continuous_vars] - mean_sex

    return df

def split_var(df, method, **kwargs):
    """
    Splits the dataframe into features and targets based on the specified method.
    """
    if method not in ['discrete', 'continuous']:
        raise ValueError("Invalid method. Choose either 'discrete' or 'continuous'.")

    x = df.drop(['who_cvd', 'month_fp_whocvd'], axis=1).values.astype('float32')
    get_target = lambda df: (df['month_fp_whocvd'].values, df['who_cvd'].values)
    if method == 'discrete':
        y = labtrans.transform(*get_target(df))
    else:
        y = get_target(df)

    return x, y


