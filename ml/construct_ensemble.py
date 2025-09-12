import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
import json
import pickle

from scipy.stats import spearmanr
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

def load_csv():
    """
    there should be a folder called descriptors, where descriptors-and-labels.csv is stored (provided by Sartaaj)
    loads descriptors-and-labels.csv (i.e. a DF with CSD ref codes, RACs, geometric descriptors, labels) and returns it.
    --> also returns descriptor_names as an "optional" argument, but I use it when constructing np matrices
    """

    descriptors_and_labels_df = pd.read_csv('descriptors/descriptors-and-labels.csv')
    descriptors_and_labels_df = descriptors_and_labels_df.rename(columns = {"outputs.pbe.bandgap" : "band gap"})
    descriptor_names = descriptors_and_labels_df.columns[1:-8].tolist()

    return descriptors_and_labels_df, descriptor_names

def create_eval_demo(task = 'band gap'):
    """
    use this function if you want to demonstrate your uncertainty ensemble.. not mandatory for creating ensemble though
    """
    descriptors_and_labels_df, descriptor_names = load_csv()
    subset_df = descriptors_and_labels_df[['csd_name'] + descriptor_names + [task]]
    subset_df = subset_df.dropna()

    X, y = subset_df[descriptor_names].to_numpy(), subset_df[task].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_test, y_test

def bootstrap_samples(X, y, m=1000, random_state=None):
    """
    Generate m bootstrap samples from training data.
    refer to this: https://builtin.com/data-science/bootstrapping-statistics
    """

    rng = np.random.RandomState(random_state)
    samples = []
    for _ in range(m):
        X_res, y_res = resample(
            X, y, 
            replace=True,
            n_samples=len(X),
            random_state=rng
        )
        samples.append((X_res, y_res))
    return samples

def create_ensemble(task: str, m = 20):
    """
    performs bootstrapping, trains models on bootstrapped samples and saves models in ensemble folder
    navigate to ensemble/task, and you should see 0.pkl, 1.pkl, ..., m.pkl there
    """
    descriptors_and_labels_df, descriptor_names = load_csv()
    subset_df = descriptors_and_labels_df[['csd_name'] + descriptor_names + [task]]
    subset_df = subset_df.dropna()

    X, y = subset_df[descriptor_names].to_numpy(), subset_df[task].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(f'ensemble/{task}', exist_ok = True)
    bootstrap_sets = bootstrap_samples(X_train, y_train, m = m, random_state = 123)

    for c, i in enumerate(bootstrap_sets):
        path_to_model_save = os.path.join(f'ensemble/{task}', f'{c}.pkl')
        X_boot, y_boot = i
        model = xgb.XGBRegressor(n_estimators = 400)
        model.fit(X_boot, y_boot)

        # out of curiosity.. testing this
        y_pred = model.predict(X_test)
        srcc, mae = spearmanr(y_test, y_pred)[0], mean_absolute_error(y_test, y_pred)

        pickle.dump(model, open(path_to_model_save, "wb"))
        logger.info(f'For model {c}, SRCC = {srcc}, MAE = {mae}')
    
    print()

# IF YOU WANT TO CONSTRUCT ENSEMBLE, UNCOMMENT THIS CODE AND RUN python construct_ensemble.py
# if __name__ == "__main__":
#     properties = ['band gap', 'pure_uptake_CO2_298.00_15000', 'pure_uptake_CO2_298.00_1600000', 'pure_uptake_methane_298.00_580000', 'pure_uptake_methane_298.00_6500000',
#                           'logKH_CO2', 'logKH_CH4', 'CH4DC']
#     m = 10 #ensemble size

#     for task in properties:
#         logger.info(f'For property: {task}...')
#         create_ensemble(task = task, m = m)