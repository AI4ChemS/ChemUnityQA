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

from .ensemble_eval import load_csv, ensemble_pipeline
from .featurizer import MOFFeaturizer

def full_pipeline(csd_code):
    """ The function uses a lookup function and ensemble of models to calculate which applications a MOF is promising for based on it's properties.
    Whenever you are interested in the properties of a MOF for an application use this function:
    These properties are used for determining the application recommendations: ['band gap', 'pure_uptake_CO2_298.00_15000', 'pure_uptake_CO2_298.00_1600000', 'pure_uptake_methane_298.00_580000', 'pure_uptake_methane_298.00_6500000', 
              'logKH_CO2', 'logKH_CH4', 'CH4DC']
    
    These are the supported applications: ["Semiconductor", "Methane Storage", "carbon capture", "direct air capture"]
    
    INPUT: 
            -csd_code (str): the CSD Ref Code for the MOF in question
    
    OUTPUT: 
            -dictionary with keys being applications and values are whether the MOF is promising or not for that application"""
    
    properties = ['band gap', 'pure_uptake_CO2_298.00_15000', 'pure_uptake_CO2_298.00_1600000', 'pure_uptake_methane_298.00_580000', 'pure_uptake_methane_298.00_6500000', 
              'logKH_CO2', 'logKH_CH4', 'CH4DC']

    descriptors_and_labels_df, descriptor_names = load_csv()
    if csd_code in descriptors_and_labels_df['csd_name'].tolist():
        # RACs and zpp descriptors exist
        subset_df = descriptors_and_labels_df[descriptors_and_labels_df['csd_name'] == csd_code]
        properties_subset_df = subset_df[properties]

        property_does_not_exist = {prop : np.isnan(subset_df[prop].values[0]) for prop in properties}
        all_properties_for_mof = {}

        for key, value in property_does_not_exist.items():
            if value == False:
                all_properties_for_mof[key] = subset_df[key].values[0]
                #logger.info(f'{key}: {all_properties_for_mof[key]}')
            
            else:
                X_eval = subset_df[descriptor_names].to_numpy()
                avg_pred, std_pred = ensemble_pipeline(X_test = X_eval, task = key)
                #all_properties_for_mof[key] = f"{avg_pred} +/- {std_pred}"
                all_properties_for_mof[key] = avg_pred

    else:
        # RACs and zpp descriptors do not exist
        logger.info(f'Featurizing MOF {csd_code}...')
        X_eval = MOFFeaturizer(csd_ref_code = csd_code).all_features(include_matminer = False)

        all_properties_for_mof = {}

        for task in properties:
            avg_pred, std_pred = ensemble_pipeline(X_test = X_eval, task = task)
            #all_properties_for_mof[task] = f"{avg_pred} +/- {std_pred}"
            all_properties_for_mof[key] = avg_pred
    
    recommendations = {}
    recommendations['semiconductors'] = "Promising" if all_properties_for_mof['band gap'] < 3 else "Not promising"
    recommendations['carbon capture'] = "Promising" if all_properties_for_mof['pure_uptake_CO2_298.00_15000'] >= 2 else "Not promising"
    recommendations['methane storage'] = "Promising" if all_properties_for_mof['pure_uptake_methane_298.00_6500000'] >= 7.93 else "Not promising"
    recommendations['DAC'] = "Promising" if all_properties_for_mof['logKH_CO2'] > -3.69 else "Not promising"
    
    return recommendations

## this is what the output looks like if I use ABAYIO or something

# full_pipeline('ABAYIO')
# Evaluating property: 100%|██████████| 10/10 [00:00<00:00, 29.34it/s]
# 2025-09-11 19:17:11.633 | INFO     | __main__:full_pipeline:23 - band gap : 1.2223217487335205 +/- 0.21109257638454437
# 2025-09-11 19:17:11.636 | INFO     | __main__:full_pipeline:17 - pure_uptake_CO2_298.00_15000: 0.4529062832
# 2025-09-11 19:17:11.640 | INFO     | __main__:full_pipeline:17 - pure_uptake_CO2_298.00_1600000: 8.8144199921
# 2025-09-11 19:17:11.644 | INFO     | __main__:full_pipeline:17 - pure_uptake_methane_298.00_580000: 3.1030846456
# 2025-09-11 19:17:11.654 | INFO     | __main__:full_pipeline:17 - pure_uptake_methane_298.00_6500000: 8.4242319463
# 2025-09-11 19:17:11.662 | INFO     | __main__:full_pipeline:17 - logKH_CO2: -4.353803176633964
# 2025-09-11 19:17:11.665 | INFO     | __main__:full_pipeline:17 - logKH_CH4: -4.770093007366181
# 2025-09-11 19:17:11.667 | INFO     | __main__:full_pipeline:17 - CH4DC: 140.90386560003927
# {'band gap': '1.2223217487335205 +/- 0.21109257638454437',
#  'pure_uptake_CO2_298.00_15000': 0.4529062832,
#  'pure_uptake_CO2_298.00_1600000': 8.8144199921,
#  'pure_uptake_methane_298.00_580000': 3.1030846456,
#  'pure_uptake_methane_298.00_6500000': 8.4242319463,
#  'logKH_CO2': -4.353803176633964,
#  'logKH_CH4': -4.770093007366181,

#  'CH4DC': 140.90386560003927}

