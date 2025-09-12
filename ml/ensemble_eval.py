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

def ensemble_pipeline(X_test : np.array, task : str):
    """
    Accepts one prompt input at a time i.e. X_test should be (num_features, ) shape
    """
    if len(X_test) == 0:
        raise ValueError("Evaluation set is missing...")
    
    if X_test.shape[0] != 1:
        X_test = X_test.reshape(1, -1)

    path_to_ensemble = f'ensemble/{task}'
    if os.path.exists(path_to_ensemble) == False:
        raise ValueError(f"Task ensemble does not exist for {task}")
    
    list_of_models = os.listdir(path_to_ensemble)
    y_pred_ensemble = []

    for i in tqdm(list_of_models, desc = 'Evaluating property'):
        path_to_pkl = os.path.join(path_to_ensemble, i)
        model = pickle.load(open(path_to_pkl, "rb"))
        y_pred = model.predict(X_test)
        y_pred_ensemble.append(y_pred[0])

    return np.mean(y_pred_ensemble), np.std(y_pred_ensemble)