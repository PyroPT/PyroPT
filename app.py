#!/usr/bin/env python3
"""
Machine Learning for Garnet Pressure-Temperature Prediction

This module implements a KNN algorithm on a database of garnet compositions from 
peridotite xenoliths in kimberlites and lamproites. It takes unknown input data 
and yields predictions for pressure and temperature.

The code includes:
- Fe3+ calculations using the Droop (1987) method
- Data preprocessing and filtering
- KNN model training and evaluation
- Unknown sample prediction
- Visualization of results
"""

import pandas as pd
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sympy import symbols, Eq, solve
from typing import Tuple, List, Dict, Optional, Union, IO
import warnings


class MissingRequiredColumnsError(Exception):
    """Raised when uploaded data is missing required oxide columns."""
    pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DATA
# =============================================================================

# Mineral configuration for Fe3+ calculations
MINERAL_CONFIG = pd.DataFrame([('Garnet', 'Garnet', 12, 8, 'yes', 0.014, 0.055)])
MINERAL_CONFIG.columns = ['Mineral', 'Abbreviation', 'Oxygens', 'Cations', 'Droop1987', 'MinFe3', 'MaxFe3']
MINERAL_CONFIG = MINERAL_CONFIG.sort_values('Mineral').reset_index(drop=True)

# Oxide configuration for conversions
OXIDE_CONFIG = pd.DataFrame([
    ('SiO2', 60.08, 2, 1, 4), ('TiAll', 79.87, 2, 1, 4), ('TiO2', 79.87, 2, 1, 4), 
    ('Al2O3', 101.96, 3, 2, 3), ('Cr2O3', 151.99, 3, 2, 3), ('FeO', 71.84, 1, 1, 2), 
    ('Fe2O3', 159.69, 3, 2, 3), ('MnO', 70.94, 1, 1, 2), ('MgO', 40.30, 1, 1, 2), 
    ('NiO', 74.69, 1, 1, 2), ('CaO', 56.08, 1, 1, 2), ('Na2O', 61.98, 1, 2, 1), 
    ('K2O', 94.20, 1, 2, 1), ('H2O', 18.01, 1, 2, 1), ('P2O5', 283.89, 5, 2, 5), 
    ('V2O3', 149.88, 3, 2, 3), ('SrO', 103.62, 1, 2, 2), ('CoO', 74.93, 1, 2, 2),
], columns=['Oxide', 'MolecularWeight', 'Oxygens', 'Cations', 'CationCharge'])

# Required oxides for Fe3+ calculations
REQUIRED_OXIDES = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'FeO', 'MnO', 'TiO2', 'Cr2O3', 'Na2O']


def dataframe_signature(df: pd.DataFrame) -> str:
    """
    Generate a stable signature for a DataFrame irrespective of column order.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        Hex digest representing the DataFrame contents
    """
    normalized = df.sort_index(axis=1)
    csv_bytes = normalized.to_csv(index=False).encode('utf-8')
    return hashlib.md5(csv_bytes).hexdigest()

# Model parameters
DEFAULT_N_NEIGHBORS = 12
DEFAULT_RANDOM_STATE_P = 286
DEFAULT_RANDOM_STATE_T = 154

# Feature sets for P and T prediction
FEATURES_P = ['Al2O3', 'TiO2', 'CaO', 'MgO']
FEATURES_T = ['MnO', 'TiO2', 'MgO', 'CaO']

# Data paths
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "required" / "training_data.csv"

# Garnet types to exclude from training
EXCLUDED_GARNET_TYPES = ['G1', 'G3', 'G4', 'G5', 'G12', 'Exp']

# Plotting configuration
GARNET_COLOR_MAP = {
    'G10': 'blue',
    'G9': 'green', 
    'G11': 'orange',
    'Cpx': 'black'
}

# Geotherm parameters
MOHO_TEMPERATURE = 480  # °C
MOHO_PRESSURE = 13      # kbar
MOHO_UNCERTAINTY = 5
ADIABAT_TEMPERATURE = 1300  # °C

# Garnet types to include (G9, G10 are default)
INCLUDE_G11 = True  # Default to include G11 - allow user to unselect in UI sidebar
FIT_GEOTHERM = True

# =============================================================================
# FE3+ CALCULATION FUNCTIONS
# =============================================================================

def convert_weight_to_moles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert oxide weight percentages to moles.
    
    Args:
        df: DataFrame with oxide columns
        
    Returns:
        DataFrame with oxide values converted to moles
    """
    oxides_from_data = df.columns
    for oxide in oxides_from_data:
        if oxide in OXIDE_CONFIG['Oxide'].values:
            mw = OXIDE_CONFIG.loc[OXIDE_CONFIG['Oxide'] == oxide, 'MolecularWeight'].iloc[0]
            df[oxide] = df[oxide].div(mw).round(5)
    return df


def convert_moles_to_elements(df: pd.DataFrame, mineral: str) -> pd.DataFrame:
    """
    Convert moles to elements and normalize to mineral stoichiometry.
    
    Args:
        df: DataFrame with oxide data in moles
        mineral: Mineral type for normalization
        
    Returns:
        DataFrame with element data normalized to mineral stoichiometry
    """
    elements = df.columns
    
    # Step 1: Calculate oxygen number of moles
    for element in elements:
        if element in OXIDE_CONFIG['Oxide'].values:
            ox_number = OXIDE_CONFIG.loc[OXIDE_CONFIG['Oxide'] == element, 'Oxygens'].iloc[0]
            df[element] = df[element].mul(ox_number).round(4)
    
    # Step 2: Normalize to required number of oxygens
    mineral_clean = mineral.replace('/', '').replace('.xls', '')
    ox_to_norm = MINERAL_CONFIG.loc[MINERAL_CONFIG['Mineral'] == mineral_clean, 'Oxygens'].iloc[0]
    df = df.div(df.sum(axis=1), axis=0)
    df = df.mul(ox_to_norm, axis=0)
    
    # Step 3: Calculate number of cations
    for element in elements:
        if element in OXIDE_CONFIG['Oxide'].values:
            cat = OXIDE_CONFIG.loc[OXIDE_CONFIG['Oxide'] == element, 'Cations'].iloc[0]
            ox = OXIDE_CONFIG.loc[OXIDE_CONFIG['Oxide'] == element, 'Oxygens'].iloc[0]
            ratio = cat / ox
            df[element] = df[element].mul(ratio).round(4)
    
    return df.fillna(0)


def calculate_fe3_droop1987(df: pd.DataFrame, mineral: str, sum_charge: float, 
                           min_fe3: float, max_fe3: float) -> pd.DataFrame:
    """
    Calculate Fe3+ using the Droop (1987) approach.
    
    Args:
        df: DataFrame with element data
        mineral: Mineral type
        sum_charge: Sum of cation charges
        min_fe3: Minimum Fe3+ value
        max_fe3: Maximum Fe3+ value
        
    Returns:
        DataFrame with Fe3+ calculated
    """
    cat_number_t = MINERAL_CONFIG.loc[MINERAL_CONFIG['Mineral'] == mineral, 'Cations'].iloc[0]
    ox_number_norm = MINERAL_CONFIG.loc[MINERAL_CONFIG['Mineral'] == mineral, 'Oxygens'].iloc[0]
    a = cat_number_t / sum_charge
    
    # Calculate Fe3+
    iron3 = 2 * ox_number_norm * (1 - a)
    
    # Normalize formula to T cations
    df = df.mul(a)
    
    # Calculate total iron and check bounds
    total_iron = df['FeO']
    
    # Apply min/max constraints
    iron3 = max(min_fe3, min(iron3, max_fe3))
    
    # Recalculate Fe2+ and Fe3+
    check_df = total_iron - iron3
    check = float(check_df.iloc[0])
    
    if check < 0:
        df['Fe2O3'] = total_iron
        df['FeO'] = 0
    else:
        df['Fe2O3'] = iron3
        df['FeO'] = total_iron - iron3
    
    return df


def convert_elements_to_weight(df_elements: pd.DataFrame, df_weight: pd.DataFrame) -> pd.DataFrame:
    """
    Convert element data back to weight percentages.
    
    Args:
        df_elements: DataFrame with element data
        df_weight: Original weight percentage DataFrame
        
    Returns:
        DataFrame with Fe2O3 and FeO recalculated in weight percentages
    """
    fe2_ratio = df_elements['FeO'] / (df_elements['FeO'] + df_elements['Fe2O3'])
    fe3_ratio = df_elements['Fe2O3'] / (df_elements['FeO'] + df_elements['Fe2O3'])
    df_weight['Fe2O3'] = 1.1113 * df_weight['FeO'] * fe3_ratio
    df_weight['FeO'] = df_weight['FeO'] * fe2_ratio
    return df_weight


def recalculate_fe3_droop(df: pd.Series) -> pd.Series:
    """
    Main function to calculate Fe2O3 in garnet using the Droop (1987) method.
    
    Args:
        df: Series containing garnet composition data
        
    Returns:
        Series with Fe3+ calculated and added
    """
    mineral = df.get('Mineral', 'Garnet')
    
    # Extract oxide data by name to remain agnostic to column order
    missing_oxides = [oxide for oxide in REQUIRED_OXIDES if oxide not in df.index]
    if missing_oxides:
        raise MissingRequiredColumnsError(
            f"Your file is missing required columns: {', '.join(missing_oxides)}."
        )
    
    ser_ox = df.loc[REQUIRED_OXIDES]
    df_ox = pd.DataFrame([ser_ox])
    
    # Convert to moles and elements
    moles = convert_weight_to_moles(df_ox.copy())
    elements = convert_moles_to_elements(moles, mineral)
    
    # Check if Droop 1987 is needed
    sum_values = float(elements.sum(axis=1).round(5))
    cat_number_t = float(MINERAL_CONFIG.loc[MINERAL_CONFIG['Mineral'] == mineral, 'Cations'].iloc[0])
    
    # Get Fe3+ constraints
    min_fe3 = float(MINERAL_CONFIG.loc[MINERAL_CONFIG['Mineral'] == mineral, 'MinFe3'].iloc[0])
    max_fe3 = float(MINERAL_CONFIG.loc[MINERAL_CONFIG['Mineral'] == mineral, 'MaxFe3'].iloc[0])
    
    if sum_values > cat_number_t:
        # Apply Droop 1987
        elements = calculate_fe3_droop1987(elements, mineral, sum_values, min_fe3, max_fe3)
    else:
        # Apply minimum Fe3+
        elements['Fe2O3'] = float(min_fe3)
        elements['FeO'] = elements['FeO'] - float(min_fe3)
    
    # Convert back to oxides
    new_ox = convert_elements_to_weight(elements, df_ox)
    
    # Update the original series with recalculated oxide values while preserving metadata
    updated = df.copy()
    ox_ser_new = new_ox.iloc[0]
    for column, value in ox_ser_new.items():
        updated[column] = value
    
    return updated


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_training_data(df: pd.DataFrame, target_variable: str, features: List[str], 
                         exclude_garnets: List[str], random_state: int) -> Tuple:
    """
    Prepare training data by filtering and splitting.
    
    Args:
        df: Input DataFrame
        target_variable: Target variable name ('P' or 'T')
        features: List of feature column names
        exclude_garnets: List of garnet types to exclude
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, garnet_types_train, garnet_types_test)
    """
    # Filter out specified garnet types
    df_filtered = df[~df['Garnet_type'].isin(exclude_garnets)]
    df_filtered['Target'] = df_filtered[target_variable]
    
    # Select top and bottom instances for each garnet type
    top_bottom_indices = []
    for garnet in df_filtered['Garnet_type'].unique():
        garnet_group = df_filtered[df_filtered['Garnet_type'] == garnet]
        sorted_group = garnet_group.sort_values(by='Target', ascending=True)
        top_indices = sorted_group.head(12).index
        bottom_indices = sorted_group.tail(12).index
        top_bottom_indices.extend(top_indices)
        top_bottom_indices.extend(bottom_indices)
    
    # Separate selected instances
    df_top_bottom = df_filtered.loc[top_bottom_indices]
    df_remaining = df_filtered.drop(top_bottom_indices)
    
    # Prepare remaining data
    X_remaining = df_remaining[features]
    y_remaining = df_remaining['Target']
    garnet_types_remaining = df_remaining['Garnet_type']
    
    # Create stratified split
    num_bins = 5
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.quantile(y_remaining, quantiles)
    y_bin = pd.cut(y_remaining, bins=bins, include_lowest=True)
    
    # Split the data
    X_train, X_test, y_train, y_test, garnet_types_train, garnet_types_test = train_test_split(
        X_remaining, y_remaining, garnet_types_remaining, 
        test_size=0.2, random_state=random_state, shuffle=True, stratify=y_bin
    )
    
    # Add selected instances to training set
    X_train = pd.concat([X_train, df_top_bottom[features]], axis=0)
    y_train = pd.concat([y_train, df_top_bottom['Target']], axis=0)
    garnet_types_train = pd.concat([garnet_types_train, df_top_bottom['Garnet_type']], axis=0)
    
    return X_train, X_test, y_train, y_test, garnet_types_train, garnet_types_test


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series, 
                           garnet_types_test: pd.Series, target_variable: str) -> Tuple:
    """
    Train and evaluate a KNN model.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        garnet_types_test: Test garnet types
        target_variable: Name of target variable
        
    Returns:
        Tuple of (X_test, y_test, y_pred, scaler, knn_model)
    """
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN model
    knn_model = KNeighborsRegressor(n_neighbors=DEFAULT_N_NEIGHBORS)
    knn_model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = knn_model.predict(X_test_scaled)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error for {target_variable} prediction: {mae:.2f}")
    
    # Calculate MAE per garnet type
    X_test_copy = X_test.copy()
    X_test_copy['Predicted'] = y_pred
    X_test_copy['Actual'] = y_test.values
    
    mae_per_garnet = X_test_copy.groupby(garnet_types_test).apply(
        lambda group: mean_absolute_error(group['Actual'], group['Predicted'])
    )
    print(f"\nMean Absolute Error per Garnet_Type:")
    print(mae_per_garnet)
    
    return X_test, y_test, y_pred, scaler, knn_model


# =============================================================================
# UNKNOWN SAMPLE PROCESSING
# =============================================================================

def load_unknown_samples(file_source: Union[str, IO]) -> pd.DataFrame:
    """
    Load unknown samples from CSV file.
    
    Args:
        file_source: Path to CSV file or file-like object
        
    Returns:
        DataFrame with unknown samples
    """
    if hasattr(file_source, 'seek'):
        file_source.seek(0)
    df = pd.read_csv(file_source)
    
    # Add 'Mineral' column if missing
    if 'Mineral' not in df.columns:
        df['Mineral'] = 'Garnet'
    
    return df


def load_training_data(path: Union[str, Path] = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load the reference training dataset."""
    return pd.read_csv(path)


def preprocess_unknown_samples(unknowns: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess unknown samples including Fe3+ calculation and validation.
    
    Args:
        unknowns: DataFrame with unknown samples
        training_df: Training DataFrame for range validation
        
    Returns:
        Preprocessed DataFrame with validation flags
    """
    missing_columns = [col for col in REQUIRED_OXIDES if col not in unknowns.columns]
    if missing_columns:
        raise MissingRequiredColumnsError(
            "Your file must include the following columns: "
            f"{', '.join(sorted(REQUIRED_OXIDES))}. "
            f"Missing: {', '.join(missing_columns)}."
        )
    
    old_unknowns = unknowns.copy()
    new_rows = []
    
    # Process each sample
    for idx, row in old_unknowns.iterrows():
        if row['SiO2'] == 0.000001:
            # No data
            new_rows.append(row)
        else:
            # Calculate Fe3+
            new = recalculate_fe3_droop(row)
            new_rows.append(new)
    
    # Create new DataFrame
    unknowns = pd.DataFrame(new_rows)
    
    # Calculate totals
    oxides = ['SiO2', 'Al2O3', 'Cr2O3', 'TiO2', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'Fe2O3']
    unknowns['Total'] = unknowns[oxides].sum(axis=1)
    
    # Initialize validation
    unknowns['Valid'] = True
    unknowns['Validation_Failures'] = ''
    
    # Check total range
    total_invalid = (unknowns['Total'] < 99) | (unknowns['Total'] > 102)
    unknowns.loc[total_invalid, 'Valid'] = False
    unknowns.loc[total_invalid, 'Validation_Failures'] += 'Total out of range; '
    
    # Check feature ranges
    for feature in oxides:
        if feature in training_df.columns:
            min_val = training_df[feature].min()
            max_val = training_df[feature].max()
            out_of_range = (unknowns[feature] < min_val) | (unknowns[feature] > max_val)
            unknowns.loc[out_of_range, 'Valid'] = False
            unknowns.loc[out_of_range, 'Validation_Failures'] += f'{feature} out of range; '
    
    # Calculate derived parameters
    unknowns['Ca_int'] = np.where(
        unknowns['CaO'] <= 3.375 + 0.25 * unknowns['Cr2O3'],
        13.5 * unknowns['CaO'] / (unknowns['Cr2O3'] + 13.5),
        unknowns['CaO'] - 0.25 * unknowns['Cr2O3']
    )
    unknowns['Mg#'] = (unknowns['MgO'] / 40.3) / ((unknowns['MgO'] / 40.3) + (unknowns['FeO'] / 71.85))
    
    # Classify garnet types
    unknowns['Garnet_type'] = 'G0'  # Default
    
    # Define classification conditions
    conditions = [
        # G1
        ((0 <= unknowns['Cr2O3']) & (unknowns['Cr2O3'] <= 4) &
         (3.375 <= unknowns['Ca_int']) & (unknowns['Ca_int'] <= 6.0) &
         (0.65 <= unknowns['Mg#']) & (unknowns['Mg#'] <= 0.85) &
         (unknowns['TiO2'] > (2.13 - 2.1 * unknowns['Mg#'])) & (unknowns['TiO2'] < 4)),
        # G11
        ((1 < unknowns['Cr2O3']) & (unknowns['Cr2O3'] < 20) &
         (unknowns['Ca_int'] > 3) & (unknowns['CaO'] < 28) &
         (0.65 < unknowns['Mg#']) & (unknowns['Mg#'] < 0.9) &
         (unknowns['TiO2'] > (2.13 - 2.1 * unknowns['Mg#'])) & (unknowns['TiO2'] < 4)),
        # G10
        ((1 < unknowns['Cr2O3']) & (unknowns['Cr2O3'] < 22) &
         (0 <= unknowns['Ca_int']) & (unknowns['Ca_int'] <= 3.375) &
         (0.75 <= unknowns['Mg#']) & (unknowns['Mg#'] <= 0.95)),
        # G9
        ((1 < unknowns['Cr2O3']) & (unknowns['Cr2O3'] <= 20) &
         (3.375 <= unknowns['Ca_int']) & (unknowns['Ca_int'] <= 5.4) &
         (0.7 <= unknowns['Mg#']) & (unknowns['Mg#'] <= 0.9)),
        # G12
        ((1 <= unknowns['Cr2O3']) & (unknowns['Cr2O3'] <= 20) &
         (unknowns['Ca_int'] > 5.4) & (unknowns['CaO'] < 28) & (unknowns['MgO'] > 5)),
        # G5
        ((unknowns['TiO2'] < (2.13 - 2.1 * unknowns['Mg#'])) &
         (1 <= unknowns['Cr2O3']) & (unknowns['Cr2O3'] <= 4) &
         (3.375 <= unknowns['Ca_int']) & (unknowns['Ca_int'] <= 5.4) &
         (0.3 <= unknowns['Mg#']) & (unknowns['Mg#'] <= 0.7)),
        # G4
        ((unknowns['TiO2'] < (2.13 - 2.1 * unknowns['Mg#'])) &
         (unknowns['Cr2O3'] < 1) & (2 <= unknowns['CaO']) & (unknowns['CaO'] <= 6) &
         (0.3 <= unknowns['Mg#']) & (unknowns['Mg#'] <= 0.9)),
        # G3
        ((unknowns['Cr2O3'] < 1) & (6 <= unknowns['CaO']) & (unknowns['CaO'] <= 32) &
         (0.17 <= unknowns['Mg#']) & (unknowns['Mg#'] <= 0.86) &
         (unknowns['TiO2'] < (2.13 - 2.1 * unknowns['Mg#'])) & (unknowns['TiO2'] < 2))
    ]
    
    choices = ['G1', 'G11', 'G10', 'G9', 'G12', 'G5', 'G4', 'G3']
    
    # Apply conditions
    assigned = np.zeros(len(unknowns), dtype=bool)
    for condition, choice in zip(conditions, choices):
        unassigned_condition = condition & ~assigned
        unknowns.loc[unassigned_condition, 'Garnet_type'] = choice
        assigned = assigned | unassigned_condition
    
    # Mark invalid garnet types
    invalid_garnet_type = ~unknowns['Garnet_type'].isin(['G9', 'G10', 'G11'])
    unknowns.loc[invalid_garnet_type, 'Valid'] = False
    unknowns.loc[invalid_garnet_type, 'Validation_Failures'] += 'Invalid Garnet type; '
    
    # Clean up validation failures
    unknowns['Validation_Failures'] = unknowns['Validation_Failures'].str.rstrip('; ')
    
    return unknowns


def predict_unknown_samples(unknowns: pd.DataFrame, model_type: str, training_df: pd.DataFrame,
                          features_p: List[str], features_t: List[str], 
                          random_state: int, n_neighbors: int = DEFAULT_N_NEIGHBORS) -> pd.DataFrame:
    """
    Predict pressure or temperature for unknown samples.
    
    Args:
        unknowns: DataFrame with unknown samples
        model_type: 'P' for pressure or 'T' for temperature
        training_df: Training DataFrame
        features_p: Features for pressure prediction
        features_t: Features for temperature prediction
        random_state: Random state for reproducibility
        n_neighbors: Number of neighbors for KNN
        
    Returns:
        DataFrame with predictions and neighbor information
    """
    # Preprocess unknowns
    unknowns = preprocess_unknown_samples(unknowns, training_df)
    
    # Prepare training data
    target_variable = 'P' if model_type == 'P' else 'T'
    features = features_p if model_type == 'P' else features_t
    
    X_train, X_test, y_train, y_test, garnet_types_train, garnet_types_test = prepare_training_data(
        training_df, target_variable, features, EXCLUDED_GARNET_TYPES, random_state
    )
    
    # Fill missing values
    X_train = X_train.fillna(12)
    X_test = X_test.fillna(12)
    
    # Train model
    _, _, _, scaler, knn_model = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, garnet_types_test, target_variable
    )
    
    # Prepare unknown samples for prediction
    X_unknowns = unknowns[features].copy()
    X_unknowns = X_unknowns.replace([np.inf, -np.inf], np.nan)
    X_unknowns = X_unknowns.fillna(0.000001)
    X_unknowns_scaled = scaler.transform(X_unknowns)
    
    # Make predictions
    predictions = np.full(unknowns.shape[0], np.nan)
    neighbors = np.full((unknowns.shape[0], n_neighbors), np.nan)
    distances = np.full((unknowns.shape[0], n_neighbors), np.nan)
    
    valid_unknowns = unknowns[unknowns['Valid']]
    if not valid_unknowns.empty:
        X_valid_unknowns_scaled = scaler.transform(valid_unknowns[features])
        predictions[unknowns['Valid']] = knn_model.predict(X_valid_unknowns_scaled)
        distances[unknowns['Valid']], neighbors[unknowns['Valid']] = knn_model.kneighbors(
            X_valid_unknowns_scaled, n_neighbors=n_neighbors, return_distance=True
        )
    
    # Calculate average distances
    avg_distances = np.mean(distances, axis=1)
    predictions = np.round(predictions, 1)
    
    # Prepare output
    output = unknowns.copy()
    output['Predicted_' + target_variable] = predictions
    output['Average_Distance'] = avg_distances
    
    # Add neighbor information
    known_samples = X_train.copy()
    known_samples['Sample_ID'] = training_df.loc[known_samples.index, 'Sample_ID']
    known_samples['Actual_' + target_variable] = y_train.values
    known_samples['Reference_Short'] = training_df.loc[known_samples.index, 'Reference Short']
    known_samples['Reference_Doi'] = training_df.loc[known_samples.index, 'Reference Doi']
    known_samples['Specific_Location'] = training_df.loc[known_samples.index, 'Specific Location']
    known_samples['Rocktype'] = training_df.loc[known_samples.index, 'Rocktype']
    
    # Add neighbor details
    for i, neighbor_indices in enumerate(neighbors):
        if not np.isnan(neighbor_indices[0]):
            for j, idx in enumerate(neighbor_indices, start=1):
                if idx < len(known_samples):
                    output.loc[unknowns.index[i], f'Neighbor_{j}_Sample_ID'] = known_samples.iloc[int(idx)]['Sample_ID']
                    output.loc[unknowns.index[i], f'Neighbor_{j}_Actual_{target_variable}'] = known_samples.iloc[int(idx)]['Actual_' + target_variable]
                    output.loc[unknowns.index[i], f'Neighbor_{j}_Reference_Short'] = known_samples.iloc[int(idx)]['Reference_Short']
                    output.loc[unknowns.index[i], f'Neighbor_{j}_Reference_Doi'] = known_samples.iloc[int(idx)]['Reference_Doi']
                    output.loc[unknowns.index[i], f'Neighbor_{j}_Specific_Location'] = known_samples.iloc[int(idx)]['Specific_Location']
                    output.loc[unknowns.index[i], f'Neighbor_{j}_Rocktype'] = known_samples.iloc[int(idx)]['Rocktype']
    
    # Round numerical columns
    output = output.round({col: 3 for col in output.columns 
                          if col not in ['Predicted_P', 'Predicted_T', 'Sample_ID', 'Garnet_type', 
                                       'Valid', 'Validation_Failures', 'Average_Distance']})
    
    if 'Fe2O3' not in output.columns and 'Fe2O3' in unknowns.columns:
        output['Fe2O3'] = unknowns['Fe2O3']
    
    return output


# =============================================================================
# VISUALISATION FUNCTIONS
# =============================================================================

def plot_pt_results(df: pd.DataFrame, pipe: Optional[str] = None,
                   anchor_weight: Optional[float] = None, include_cpx: bool = False,
                   include_g11: bool = INCLUDE_G11,
                   fit_geotherm: bool = FIT_GEOTHERM,
                   moho_temperature: float = MOHO_TEMPERATURE,
                   moho_pressure: float = MOHO_PRESSURE,
                   moho_uncertainty: float = MOHO_UNCERTAINTY,
                   adiabat_temperature: float = ADIABAT_TEMPERATURE
                   ) -> Tuple[plt.Figure, Dict[str, Optional[float]], bytes]:
    """
    Create P-T plot with geotherm fitting.
    
    Args:
        df: DataFrame with P-T predictions
        pipe: Optional pipe name to filter data
        anchor_weight: Weight for Moho anchor point
        include_cpx: Whether to include Cpx data in fitting
        include_g11: Whether to include G11 garnets in fitting
        moho_temperature: Temperature anchor (°C) at the Moho
        moho_pressure: Pressure anchor (kbar) at the Moho
        moho_uncertainty: Uncertainty weighting for the Moho anchor
        adiabat_temperature: Mantle adiabat temperature baseline (°C)
    """
    plt.figure(figsize=(6, 8))

    # Geotherm calculation functions
    def geotherm(z_km: float, q0: float, Ts: float = 273.0, k: float = 4.5, H: float = 1e-7) -> float:
        z_m = z_km * 1000
        return Ts + (q0 / k) * z_m - (H / (2 * k)) * z_m**2
    
    def kbar_to_km(P_kbar: float) -> float:
        return P_kbar * 3.1
    
    def km_to_kbar(km: float) -> float:
        return km / 3.1
    
    def gd_transition_p_gpa(T_C: float) -> float:
        return 1.94 + 0.0025 * T_C
    
    # Background geotherms
    pressure_kbar = np.linspace(0, 90, 500)
    depth_km = kbar_to_km(pressure_kbar)
    Tp = adiabat_temperature
    gamma = 0.3
    adiabat_T = Tp + gamma * pressure_kbar
    
    heat_flows = [0.036, 0.040, 0.044, 0.048, 0.052]
    labels = ['36 mW/m²', '40 mW/m²', '44 mW/m²', '48 mW/m²', '52 mW/m²']
    colors = ['saddlebrown', 'darkred', 'red', 'orange', 'gold']
    
    for q0, label_q, color_q in zip(heat_flows, labels, colors):
        T_K = geotherm(depth_km, q0)
        T_C = T_K - 273.15
        crossing_idx = np.where(T_C >= adiabat_T)[0]
        idx_cut = crossing_idx[0] if crossing_idx.size > 0 else len(depth_km) - 1
        moho_index = np.searchsorted(pressure_kbar, moho_pressure)
        start_idx = max(moho_index, 0)
        end_idx = min(idx_cut, len(depth_km) - 1)
        if start_idx <= end_idx:
            T_C_plot = T_C[start_idx:end_idx + 1]
            P_plot = pressure_kbar[start_idx:end_idx + 1]
            plt.plot(T_C_plot, P_plot, color=color_q, label=label_q, 
                    linewidth=1.5, alpha=0.6, zorder=1)
    
    # Filter data
    if pipe:
        df_filtered = df[df['Pipe'] == pipe]
        title = f'P vs T for Pipe: {pipe}'
    else:
        df_filtered = df
        title = 'P vs T'
    
    # Prepare defaults for LAB summary
    T_lab = None
    P_lab = None
    T_lab_lower = None
    P_lab_lower = None
    T_lab_upper = None
    P_lab_upper = None
    slope = None
    intercept = None

    if fit_geotherm:
        T_anchor = moho_temperature
        P_anchor = moho_pressure

        fit_types = ['G9', 'G10']
        if include_g11:
            fit_types.append('G11')
        if include_cpx:
            fit_types.append('Cpx')
        df_fit = df_filtered[df_filtered['Garnet_type'].isin(fit_types)]

        T_vals = df_fit['Predicted_T'].values
        P_vals = df_fit['Predicted_P'].values
        mask_fit = ~np.isnan(T_vals) & ~np.isnan(P_vals)
        T_vals = T_vals[mask_fit]
        P_vals = P_vals[mask_fit]

        T_all = np.append(T_vals, T_anchor)
        P_all = np.append(P_vals, P_anchor)
        if anchor_weight is None:
            anchor_weight = max(1.0, 20.0 / max(moho_uncertainty, 1e-6))

        weights = np.append(np.ones_like(T_vals), anchor_weight)

        # Weighted linear regression
        X = np.vstack([T_all - T_anchor, np.ones_like(T_all)]).T
        W = np.diag(weights)
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ P_all)
        m, c = beta
        slope, intercept = float(m), float(c)

        residuals = P_all - (m * (T_all - T_anchor) + c)
        weighted_variance = np.sum(weights * residuals**2) / (len(P_all) - 2)
        cov_beta = weighted_variance * np.linalg.inv(X.T @ W @ X)
        std_err_m = np.sqrt(cov_beta[0, 0])
        std_err_c = np.sqrt(cov_beta[1, 1])
        m_upper, m_lower = m + std_err_m, m - std_err_m
        c_upper, c_lower = c + std_err_c, c - std_err_c

        # Calculate LAB intersection
        def calc_lab(m_fit: float, c_fit: float) -> Tuple[Optional[float], Optional[float]]:
            T_sym = symbols('Predicted_T')
            eq = Eq(m_fit * (T_sym - T_anchor) + c_fit, 0.5 * (T_sym - adiabat_temperature) + moho_pressure)
            sol = solve(eq, T_sym)
            sol = [float(s.evalf()) for s in sol if s.is_real and 300 <= float(s.evalf()) <= 1700]
            if not sol:
                return None, None
            T_lab_val = sol[0]
            return T_lab_val, m_fit * (T_lab_val - T_anchor) + c_fit

        T_lab, P_lab = calc_lab(m, c)
        T_lab_lower, P_lab_lower = calc_lab(m_lower, c_lower)
        T_lab_upper, P_lab_upper = calc_lab(m_upper, c_upper)

        # Plot fitted line and uncertainty if available
        T_line = np.linspace(300, 1700, 300)
        P_line = m * (T_line - T_anchor) + c
        P_line_upper = m_upper * (T_line - T_anchor) + c_upper
        P_line_lower = m_lower * (T_line - T_anchor) + c_lower
        moho_p_kbar = moho_pressure
        mask_line = P_line >= moho_p_kbar

        if np.any(mask_line):
            plt.fill_between(
                T_line[mask_line],
                P_line_lower[mask_line],
                P_line_upper[mask_line],
                color='grey',
                alpha=0.3,
                label='±2σ',
                zorder=0
            )
            plt.plot(T_line[mask_line], P_line[mask_line], 'k--', label='Fit Geotherm', zorder=2)
    
    # Plot data points
    for garnet_type, color in GARNET_COLOR_MAP.items():
        if garnet_type == 'G11' and not include_g11:
            continue
        if garnet_type == 'Cpx' and not include_cpx:
            continue
        subset = df_filtered[df_filtered['Garnet_type'] == garnet_type]
        if garnet_type == 'Cpx':
            plt.scatter(subset['Predicted_T'], subset['Predicted_P'], 
                       edgecolors='black', facecolors='none', label='Cpx', 
                       alpha=0.8, zorder=3)
        else:
            plt.scatter(subset['Predicted_T'], subset['Predicted_P'], 
                       color=color, label=garnet_type, alpha=0.7, zorder=3)
    
    # Plot mantle adiabat
    T_adiabat = np.linspace(1000, 1600, 200)
    P_adiabat = 0.5 * (T_adiabat - adiabat_temperature) + moho_pressure
    plt.plot(T_adiabat, P_adiabat, 'r-.', label='Mantle adiabat', zorder=1.5)
    
    # Plot LAB point
    if T_lab is not None:
        plt.annotate(f'LAB\n{P_lab:.1f} kbar\n{int(kbar_to_km(P_lab))} km',
                    xy=(T_lab, P_lab), xytext=(T_lab + 50, P_lab + 3),
                    arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=10, zorder=4)
    
    # Plot Garnet-Diamond transition
    T_line_gd = np.linspace(300, 1600, 200)
    P_GPa = gd_transition_p_gpa(T_line_gd)
    plt.plot(T_line_gd, P_GPa * 10, '-', color='purple', label='G–D transition', zorder=2)
    
    # Final plot configuration
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (kbar)')
    plt.title(title)
    plt.xlim(400, 1600)
    plt.ylim(20, 70)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left', fontsize='small', framealpha=0.8, ncol=1, 
              handlelength=1.5, borderaxespad=0.5)
    plt.tight_layout()
    fig = plt.gcf()

    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format='pdf', dpi=300, bbox_inches='tight')
    pdf_buffer.seek(0)

    lab_results = {
        'lab_temperature_c': T_lab,
        'lab_pressure_kbar': P_lab,
        'lab_depth_km': kbar_to_km(P_lab) if P_lab is not None else None,
        'lab_temperature_lower_c': T_lab_lower,
        'lab_pressure_lower_kbar': P_lab_lower,
        'lab_depth_lower_km': kbar_to_km(P_lab_lower) if P_lab_lower is not None else None,
        'lab_temperature_upper_c': T_lab_upper,
        'lab_pressure_upper_kbar': P_lab_upper,
        'lab_depth_upper_km': kbar_to_km(P_lab_upper) if P_lab_upper is not None else None,
        'slope': slope,
        'intercept': intercept
    }

    return fig, lab_results, pdf_buffer.getvalue()


# =============================================================================
# MAIN EXECUTION - Renders the interaction as a Streamlit app
# =============================================================================

# Logo for the tool - will be displayed in the header
LOGO = BASE_DIR / "assets" / "logo.png"



def run_app() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="PyroPT Garnet P-T Predictor", layout="wide")

    header_section = st.container()
    with header_section:
        if LOGO.exists():
            st.image(str(LOGO), width=250)
        st.title("Garnet Pressure–Temperature Predictor")
    st.write(
        "Upload garnet composition data in CSV format to estimate pressure and temperature using the trained KNN models."
    )

    @st.cache_data(show_spinner=False)
    def get_training_data(path: Union[str, Path]) -> pd.DataFrame:
        return load_training_data(path)

    try:
        training_df = get_training_data(TRAINING_DATA_PATH)
    except FileNotFoundError as exc:
        st.error(
            "Training dataset not found. Ensure `required/training_data.csv` is present in the app directory."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load training data: {exc}")
        st.stop()

    def render_footer() -> None:
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.75rem;color:#555;">
                <p><small><strong>Data Privacy:</strong> If you are using this app via <a href="https://pyropt.streamlit.app">pyropt.streamlit.app</a>, note that uploaded .CSV files are processed on a <em>Streamlit Community Cloud</em> server. Your files/data are not saved or stored, and data transfered to this service is subject to strict and high standards. See <a href="https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security">Streamlit documentation</a> for more detail.
                If you prefer to run this app on your own machine for greater data privacy, you can <a href="https://docs.streamlit.io/get-started/installation">install streamlit</a> on your computer, and download the code for this app from the <a href="https://github.com/PyroPT/PyroPT">PyroPT GitHub repository</a>.</small></p>
                <p><small>This tool was developed from research funded by the European Union (<a href='https://cordis.europa.eu/project/id/101044276'>ERC-CoG-2020 LITHO3</a>, 101044276 to ELT).</small></p>
                <p>Gary J. O'Sullivan<sup>1</sup>, Emma L. Tomlinson<sup>1</sup>, Dónal Mulligan<sup>2</sup>, Michele Rinaldi<sup>1</sup>, Oliver Higgins<sup>1,3</sup>, Phillip E. Janney<sup>4</sup>, Brendan C. Hoare<sup>5</sup></p>
                <p><small> <sup>1</sup> Department of Geology, <a href='http://www.tcd.ie'>Trinity College Dublin</a>, Ireland<br/>
                    <sup>2</sup> School of Communications, <a href='http://www.dcu.ie'>Dublin City University</a>, Ireland<br/>
                    <sup>3</sup> <a href='https://www.st-andrews.ac.uk'>University of St Andrews</a>, United Kingdom<br/>
                    <sup>4</sup> <a href='https://www.uct.ac.za'>University of Cape Town</a>, South Africa<br/>
                    <sup>5</sup> National High Magnetic Field Laboratory, <a href='https://www.fsu.edu'>Florida State University</a>, United States of America
                </small></p>
                <p><small>Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.</small></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.sidebar:
        st.header("Plotting Settings")
        st.markdown("These options adjust how the illustrative plot based on your predictions is rendered.")
        include_g11 = st.checkbox(
            "Include G11",
            value=INCLUDE_G11,
            help="Include G11 garnet in the fitted geotherm, or not. It may be desirable to exclude G11 garnet from the linear-fit geotherm where the lithosphere is in disequilibrium."
        )

        fit_geotherm = st.checkbox(
            "Fit Geotherm?",
            value=FIT_GEOTHERM,
            help="Toggle fitting of the dashed geotherm line and LAB estimate"
        )
        moho_temperature = st.slider(
            "Moho Temperature (°C)", min_value=400, max_value=700, value=int(MOHO_TEMPERATURE), step=10
        )
        moho_pressure = st.slider(
            "Moho Pressure (kbar)", min_value=5, max_value=20, value=int(MOHO_PRESSURE), step=1
        )
        moho_uncertainty = st.slider(
            "Qualitative Moho Uncertainty",
            min_value=1,
            max_value=10,
            value=int(MOHO_UNCERTAINTY),
            step=1,
            help="Affects the certainty with which the points are anchored to the Moho pressure and temperature. Lower values indicate increased certainty. Default is 5."
        )
        adiabat_temperature = st.slider(
            "Mantle Adiabat Temperature (°C)", min_value=1250, max_value=1500,
            value=int(ADIABAT_TEMPERATURE), 
            step=10,
            help="Mantle potential temperature"
        )
        st.markdown("*Changing these settings alters your plot but does not re-compute the predictions.*")
        st.markdown("---")
        st.markdown(":gray[PyroPT v.0.9.3 2025]")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing unknown garnet analyses", type="csv", accept_multiple_files=False
    )

    if not uploaded_file:
        st.info('''Your **.CSV** file must have at least the following column headings (in any order):  
        `Sample_ID`, `SiO2`, `TiO2`, `Al2O3`, `Cr2O3`, `MnO`, `MgO`, `FeO`, `CaO`, `Na2O`'''
        '''\n\nEmpty values for cells are assumed to be zero, when your file is processed. Values for `Na2O` can always be empty, if no data. You can include additional columns in your file - *these are ignored and do not affect the models*.
        ''')
        render_footer()
        return

    try:
        unknowns = load_unknown_samples(uploaded_file).fillna(0.000001)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV file: {exc}")
        render_footer()
        return

    st.success(f"Loaded {len(unknowns)} samples from `{uploaded_file.name}`.")

    data_signature = dataframe_signature(unknowns)
    cache = st.session_state.setdefault('prediction_cache', {})

    recompute_requested = st.button("🔄 Recompute predictions", type="secondary")
    if recompute_requested:
        cache.pop(data_signature, None)
        st.session_state['prediction_cache'] = cache

    cached_result = cache.get(data_signature)

    if cached_result is None:
        try:
            with st.spinner("Running pressure model..."):
                output_p = predict_unknown_samples(
                    unknowns.copy(), 'P', training_df, FEATURES_P, FEATURES_T, DEFAULT_RANDOM_STATE_P
                )

            with st.spinner("Running temperature model..."):
                output_t = predict_unknown_samples(
                    unknowns.copy(), 'T', training_df, FEATURES_P, FEATURES_T, DEFAULT_RANDOM_STATE_T
                )
        except MissingRequiredColumnsError as exc:
            st.error(str(exc))
            render_footer()
            return

        output_t_trimmed = output_t[['Sample_ID', 'Predicted_T']]
        final_output = output_p.merge(output_t_trimmed, on='Sample_ID', how='left')

        allowed_columns = [
            "Sample_ID", "SiO2", "Al2O3", "MgO", "CaO", "FeO", "MnO", "TiO2", "Cr2O3",
            "Na2O", "Fe2O3", "Mg#", "Garnet_type", "Total", "Valid",
            "Validation_Failures", "Ca_int", "Predicted_P", "Average_Distance", "Predicted_T"
        ]

        for i in range(1, 13):
            allowed_columns.extend([
                f"Neighbor_{i}_Sample_ID", f"Neighbor_{i}_Actual_P", f"Neighbor_{i}_Reference_Short",
                f"Neighbor_{i}_Reference_Doi", f"Neighbor_{i}_Specific_Location", f"Neighbor_{i}_Rocktype"
            ])

        final_output_filtered = final_output[[col for col in allowed_columns if col in final_output.columns]]

        cache[data_signature] = {
            "output_p": output_p,
            "output_t": output_t,
            "final_output": final_output,
            "final_output_filtered": final_output_filtered
        }
        st.session_state['prediction_cache'] = cache
    else:
        output_p = cached_result["output_p"]
        output_t = cached_result["output_t"]
        final_output = cached_result["final_output"]
        final_output_filtered = cached_result["final_output_filtered"]

    if 'Valid' in final_output_filtered.columns:
        reordered_columns = ['Valid'] + [col for col in final_output_filtered.columns if col != 'Valid']
        final_output_filtered = final_output_filtered[reordered_columns]
        if data_signature in cache:
            cache[data_signature]["final_output_filtered"] = final_output_filtered

    uploaded_stem = Path(uploaded_file.name).stem
    safe_input_name = re.sub(r'[^A-Za-z0-9]+', '_', uploaded_stem).strip('_').lower() or "input"
    export_timestamp = datetime.now().strftime("%d%m%y-%H%M")
    csv_download_name = f"PyroPT_predictions_{safe_input_name}_{export_timestamp}.csv"
    pdf_download_name = f"PyroPT_plot_{safe_input_name}_{export_timestamp}.pdf"

    csv_bytes = final_output_filtered.to_csv(index=False).encode('utf-8')

    valid_column = final_output_filtered.get('Valid')
    if valid_column is not None:
        valid_mask = valid_column.fillna(False)
        valid_count = int(valid_mask.sum())
        invalid_count = int(len(valid_column) - valid_count)
    else:
        valid_mask = None
        valid_count = 0
        invalid_count = 0

    has_valid_rows = bool(valid_mask is not None and valid_mask.any())
    has_predicted_pressure = (
        'Predicted_P' in final_output_filtered.columns and final_output_filtered['Predicted_P'].notna().any()
    )
    has_predicted_temperature = (
        'Predicted_T' in final_output_filtered.columns and final_output_filtered['Predicted_T'].notna().any()
    )

    has_predictions = has_valid_rows and (has_predicted_pressure or has_predicted_temperature)

    if has_predictions:
        success_msg = f"Predictions complete. {valid_count} valid row"
        success_msg += "s." if valid_count != 1 else "."
        if invalid_count:
            invalid_label = "rows" if invalid_count != 1 else "row"
            success_msg = success_msg.rstrip(".")
            success_msg += f" and {invalid_count} invalid {invalid_label} - check validation failures field for details."
        st.success(success_msg)
    else:
        st.error(
            "No valid rows were identified in the uploaded file, so no pressure or temperature predictions "
            "were generated. Please review the validation failures for each row."
        )
    st.dataframe(final_output_filtered)
    st.download_button(
        label="📄 Download predictions as CSV",
        data=csv_bytes,
        file_name=csv_download_name,
        mime="text/csv"
    )

    if not has_predictions:
        render_footer()
        return

    anchor_weight = max(1.0, 20.0 / float(moho_uncertainty))

    fig, lab_summary, plot_pdf = plot_pt_results(
        final_output_filtered,
        anchor_weight=anchor_weight,
        include_g11=include_g11,
        fit_geotherm=fit_geotherm,
        moho_temperature=float(moho_temperature),
        moho_pressure=float(moho_pressure),
        moho_uncertainty=float(moho_uncertainty),
        adiabat_temperature=float(adiabat_temperature)
    )

    _, plot_col, _ = st.columns([1, 2.5, 1])
    with plot_col:
        st.pyplot(fig, use_container_width=False)
        st.download_button(
            label="⬇️ Download plot as PDF",
            data=plot_pdf,
            file_name=pdf_download_name,
            mime="application/pdf"
        )
        if lab_summary.get('lab_temperature_c') is not None:
            st.markdown(
                (
                    "**Estimated LAB:** "
                    f"{lab_summary['lab_pressure_kbar']:.2f} kbar "
                    f"({lab_summary['lab_depth_km']:.1f} km) at "
                    f"{lab_summary['lab_temperature_c']:.0f} °C"
                )
            )

    render_footer()


def main() -> None:
    """Entry point for both Streamlit and direct execution."""
    run_app()


if __name__ == "__main__":
    main()
