## write RACs and zpp featurizer, given dict of Structure

import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

from .ensemble_eval import load_csv, ensemble_pipeline
from molSimplify.Informatics.MOF.PBC_functions import overlap_removal, solvent_removal
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors, get_primitive

from pymatgen.core import Structure
import subprocess
import glob
import re
import json
from matminer.featurizers.structure import JarvisCFID

class MOFFeaturizer:
    def __init__(self, csd_ref_code):
        self.csd_ref_code = csd_ref_code
        self.path_to_cif = os.path.join('cifs/chemunity-cifs', f"{self.csd_ref_code}.cif")

        self.structure = Structure.from_file(f'cifs/chemunity-cifs/{self.csd_ref_code}.cif')
        self.primitive_folder = 'rac_calcs/primitive_structures'
        self.RACs_folder = 'rac_calcs/RAC_folder'
        self.merged_descriptors_folder = 'rac_calcs/merged_descriptors'
        self.out_dir = "rac_calcs/matminer"

        os.makedirs(self.primitive_folder, exist_ok = True)
        os.makedirs(self.RACs_folder, exist_ok = True)
        os.makedirs(self.merged_descriptors_folder, exist_ok = True)
    
    def featurize_racs(self):
        try:
            path_to_cif = os.path.join('cifs/chemunity-cifs', f'{self.csd_ref_code}.cif')
            try:
                # get_primitive removes symmetry (enforces P1 symmetry)
                get_primitive(path_to_cif, f'{self.primitive_folder}/{self.csd_ref_code}_primitive.cif')
                get_primitive_success = True
            
            except Exception as e:
                logger.info(f'The primitive cell of {self.csd_ref_code} could not be found.')
                get_primitive_success = False
            
            if get_primitive_success:
                structure_path = os.path.join(self.primitive_folder, f"{self.csd_ref_code}_primitive.cif")
            
            else:
                logger.info(f'Failed to get the primitive structure of {self.csd_ref_code}. Aborting...')
            
            full_names, full_descriptors = get_MOF_descriptors(f'{structure_path}', 3, path = self.RACs_folder, xyzpath = f'{self.RACs_folder}/xyz/{self.csd_ref_code}.xyz', wiggle_room=1,
                    max_num_atoms=6000, get_sbu_linker_bond_info=True, surrounded_sbu_file_generation=True, detect_1D_rod_sbu=True) # Allowing for very large unit cells.
            
            lc_df = pd.read_csv(f"{self.RACs_folder}/lc_descriptors.csv")
            sbu_df = pd.read_csv(f"{self.RACs_folder}/sbu_descriptors.csv")
            linker_df = pd.read_csv(f"{self.RACs_folder}/linker_descriptors.csv")

            #it is clear that lc_df, sbu_df, linker_df are all being concatenated from previous iterations. thus, we should do lc_df = lc_df[lc_df['name'] == csd_to_qmof[cif_file]_primitive], and then do mean().to_frame()...
            lc_df = lc_df[lc_df['name'] == f'{self.csd_ref_code}_primitive']
            sbu_df = sbu_df[sbu_df['name'] == f'{self.csd_ref_code}_primitive']
            linker_df = linker_df[linker_df['name'] == f'{self.csd_ref_code}_primitive']
            name_extracted = lc_df['name'].values[0]
            lc_df = lc_df.select_dtypes(include='number').mean().to_frame().transpose()
            sbu_df = sbu_df.select_dtypes(include='number').mean().to_frame().transpose()
            linker_df = linker_df.select_dtypes(include='number').mean().to_frame().transpose()
            lc_df['name'] = [name_extracted]
            sbu_df['name'] = [name_extracted]
            linker_df['name'] = [name_extracted]

            merged_df = pd.concat([lc_df, sbu_df, linker_df], axis=1)
            merged_df.to_csv(f'{self.merged_descriptors_folder}/{self.csd_ref_code}_descriptors.csv', index=False)
            #get_primitive(directory_cif, f'featurization/primitive/{csd_to_qmof[cif_file]}')
            logger.info(f'Successfully calculated RACs for {name_extracted}!')

        except:
            logger.info(f'Failed to calculate RACs for {self.csd_ref_code}.')
        
        merged_column_names = ['D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all', 
                        'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all', 'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all', 'D_func-alpha-0-all', 
                        'D_func-alpha-1-all', 'D_func-alpha-2-all', 'D_func-alpha-3-all', 'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all', 'D_func-chi-3-all', 'D_lc-I-0-all', 
                        'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all', 'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all', 
                        'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all', 'D_lc-Z-3-all', 'D_lc-alpha-0-all', 'D_lc-alpha-1-all', 'D_lc-alpha-2-all', 'D_lc-alpha-3-all', 
                        'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all', 'D_lc-chi-3-all', 'func-I-0-all', 'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all', 
                        'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all', 'func-Z-2-all', 'func-Z-3-all', 
                        'func-alpha-0-all', 'func-alpha-1-all', 'func-alpha-2-all', 'func-alpha-3-all', 'func-chi-0-all', 'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 
                        'lc-I-1-all', 'lc-I-2-all', 'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all', 'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all', 
                        'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-alpha-0-all', 'lc-alpha-1-all', 'lc-alpha-2-all', 'lc-alpha-3-all', 'lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 
                        'lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all', 'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all',
                            'D_mc-T-2-all', 'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all', 'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all', 'D_mc-chi-3-all', 
                            'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all', 'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all', 'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 
                            'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all', 'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all', 'mc-I-3-all', 'mc-S-0-all', 
                            'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all', 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all', 'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 
                            'mc-chi-1-all', 'mc-chi-2-all', 'mc-chi-3-all', 'f-lig-I-0', 'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2', 'f-lig-S-3', 
                            'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0', 'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1', 'f-lig-chi-2', 'f-lig-chi-3']

        rac_df = pd.read_csv(f'rac_calcs/merged_descriptors/{self.csd_ref_code}_descriptors.csv')
        rac_df = rac_df[["name"] + merged_column_names]
        return rac_df

    def zpp_calcs(self):
        path_to_primitive = os.path.join('rac_calcs/primitive_structures', f'{self.csd_ref_code}_primitive.cif')
        structure_base = 'rac_calcs/zpp'

        if os.path.exists(path_to_primitive):
            structure_path = path_to_primitive

        else:
            get_primitive(self.path_to_cif, path_to_primitive)
            structure_path = path_to_primitive

        volpo_output = os.path.join(structure_base, "AV", f"{self.csd_ref_code}.txt")
        volpo_command = [
            "./network",
            "-ha",
            "-volpo", "1.4", "1.4", "10000",
            volpo_output,
            structure_path
        ]

        res_output = os.path.join(structure_base, "pore", f"{self.csd_ref_code}.txt")
        res_command = [
            "./network",
            "-ha",
            "-res",
            res_output,
            structure_path
        ]

        sa_output = os.path.join(structure_base, "SA", f"{self.csd_ref_code}_sa.txt")
        sa_command = [
            "./network",
            "-ha",
            "-sa", "1.4", "1.4", "10000",
            sa_output,
            structure_path
        ]

        try:
            subprocess.run(volpo_command, check = True)
            subprocess.run(res_command, check = True)
            subprocess.run(sa_command, check = True)
        
        except subprocess.CalledProcessError as e:
            logger.info(f'Error occurred while processing {self.csd_ref_code}: {e}')

        volpo_data, sa_data, pore_data = {}, {}, {}
        full_data = []

        unitcell_volume_regex = r"Unitcell_volume:\s+([\d\.]+)"
        density_regex = r"Density:\s+([\d\.]+)"
        poav_regex = r"POAV_A\^3:\s+([\d\.]+)"
        poav_vol_frac_regex = r"POAV_Volume_fraction:\s+([\d\.]+)"
        poav_cm3g_regex = r"POAV_cm\^3/g:\s+([\d\.]+)"
        ponav_regex = r"PONAV_A\^3:\s+([\d\.]+)"
        ponav_vol_frac_regex = r"PONAV_Volume_fraction:\s+([\d\.]+)"
        ponav_cm3g_regex = r"PONAV_cm\^3/g:\s+([\d\.]+)"

        asa_a2_regex = r"ASA_A\^2:\s+([\d\.]+)"
        asa_m2cm3_regex = r"ASA_m\^2/cm\^3:\s+([\d\.]+)"
        asa_m2g_regex = r"ASA_m\^2/g:\s+([\d\.]+)"
        nasa_a2_regex = r"NASA_A\^2:\s+([\d\.]+)"
        nasa_m2cm3_regex = r"NASA_m\^2/cm\^3:\s+([\d\.]+)"
        nasa_m2g_regex = r"NASA_m\^2/g:\s+([\d\.]+)"

        with open(volpo_output, "r") as file:
            for line in file:
                # Extract Unitcell volume
                if re.search(unitcell_volume_regex, line):
                    volpo_data["Unitcell_volume"] = float(re.search(unitcell_volume_regex, line).group(1))
                # Extract Density
                if re.search(density_regex, line):
                    volpo_data["Density"] = float(re.search(density_regex, line).group(1))
                # Extract POAV_A^3
                if re.search(poav_regex, line):
                    volpo_data["POAV_A^3"] = float(re.search(poav_regex, line).group(1))
                # Extract POAV_Volume_fraction
                if re.search(poav_vol_frac_regex, line):
                    volpo_data["POAV_Volume_fraction"] = float(re.search(poav_vol_frac_regex, line).group(1))
                # Extract POAV_cm^3/g
                if re.search(poav_cm3g_regex, line):
                    volpo_data["POAV_cm^3/g"] = float(re.search(poav_cm3g_regex, line).group(1))
                # Extract PONAV_A^3
                if re.search(ponav_regex, line):
                    volpo_data["PONAV_A^3"] = float(re.search(ponav_regex, line).group(1))
                # Extract PONAV_Volume_fraction
                if re.search(ponav_vol_frac_regex, line):
                    volpo_data["PONAV_Volume_fraction"] = float(re.search(ponav_vol_frac_regex, line).group(1))
                # Extract PONAV_cm^3/g
                if re.search(ponav_cm3g_regex, line):
                    volpo_data["PONAV_cm^3/g"] = float(re.search(ponav_cm3g_regex, line).group(1))
        # Read the sa file and extract the features
        with open(sa_output, "r") as file:
            for line in file:
                # Extract ASA_A^2
                if re.search(asa_a2_regex, line):
                    sa_data["ASA_A^2"] = float(re.search(asa_a2_regex, line).group(1))
                # Extract ASA_m^2/cm^3
                if re.search(asa_m2cm3_regex, line):
                    sa_data["ASA_m^2/cm^3"] = float(re.search(asa_m2cm3_regex, line).group(1))
                # Extract ASA_m^2/g
                if re.search(asa_m2g_regex, line):
                    sa_data["ASA_m^2/g"] = float(re.search(asa_m2g_regex, line).group(1))
                # Extract NASA_A^2
                if re.search(nasa_a2_regex, line):
                    sa_data["NASA_A^2"] = float(re.search(nasa_a2_regex, line).group(1))
                # Extract NASA_m^2/cm^3
                if re.search(nasa_m2cm3_regex, line):
                    sa_data["NASA_m^2/cm^3"] = float(re.search(nasa_m2cm3_regex, line).group(1))
                # Extract NASA_m^2/g
                if re.search(nasa_m2g_regex, line):
                    sa_data["NASA_m^2/g"] = float(re.search(nasa_m2g_regex, line).group(1))

        # Regular expressions for pore file
        pore_data_regex = r"structures/pore/\S+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
        # Read the pore file and extract the features
        with open(res_output, "r") as file:
            for line in file:
                if re.search(pore_data_regex, line):
                    match = re.search(pore_data_regex, line)
                    pore_data["Largest_included_sphere"] = float(match.group(1))
                    pore_data["Largest_free_sphere"] = float(match.group(2))
                    pore_data["Largest_included_sphere_along_free_path"] = float(match.group(3))

        # Combine volpo, sa, and pore data
        combined_data = {**volpo_data, **sa_data, **pore_data}
        full_data.append(combined_data)
        full_df = pd.DataFrame(full_data)
        zpp_names = full_df.columns.tolist()
        full_df['name'] = self.csd_ref_code
        full_df = full_df[['name'] + zpp_names]

        return full_df

    def matminer_featurizer(self):
        os.makedirs(self.out_dir, exist_ok = True)
        Matminer_labels = JarvisCFID().feature_labels()
        try:
            path_to_cif = os.path.join('cifs/chemunity-cifs', f"{self.csd_ref_code}.cif")
            jarvis = JarvisCFID()
            structure = Structure.from_file(path_to_cif)
            feats = jarvis.featurize(structure)
            json_path = os.path.join(self.out_dir, f"{self.csd_ref_code}.json")
            with open(json_path, "w") as f:
                json.dump(dict(zip(Matminer_labels, feats)), f, indent = 2)
            
            return dict(zip(Matminer_labels, feats))
        
        except Exception as e:
            logger.info(f"Failed on {self.csd_ref_code}: {e}")
    
    def all_features(self, include_matminer = False):
        rac_df = self.featurize_racs()
        zpp_df = self.zpp_calcs()

        if include_matminer:
            matminer_json = self.matminer_featurizer()
            matminer_data = {'name' : self.csd_ref_code}
            for key, value in matminer_json.items():
                matminer_data[key] = value
            
            matminer_df = pd.DataFrame(matminer_data)
            matminer_names = matminer_df.columns[1::].tolist()
            matminer_df = matminer_df[['name'] + matminer_names]

            return matminer_df[matminer_names].to_numpy()
        
        else:
            rac_zpp_df = rac_df.merge(zpp_df, how = 'inner', on = 'name')
            rac_zpp_names = rac_zpp_df.columns[1::].tolist()

            return rac_zpp_df[rac_zpp_names].to_numpy()
        
