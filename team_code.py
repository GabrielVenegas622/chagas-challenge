#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # Iterate over the records to extract the features and labels.
    features = list()
    labels = list()
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        age, sex, source, signal_mean, signal_std = extract_features(record)
        label = load_label(record)

        # Store the features and labels, but skip most of the CODE-15% data because there are many records, and the labels are weak.
        # You can treat the different data sources however you might like in your training code.
        if source != 'CODE-15%' or (i % 10) == 0:
            features.append(np.concatenate((age, sex, signal_mean, signal_std)))
            labels.append(label)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)

    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')

    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    age, sex, source, signal_mean, signal_std = extract_features(record)
    features = np.concatenate((age, sex, signal_mean, signal_std)).reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)