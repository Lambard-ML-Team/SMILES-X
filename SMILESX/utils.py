import os
import sys
import math
import logging
import datetime
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf

from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import RobustScaler

from rdkit import Chem
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

np.set_printoptions(precision=3)

class StopExecution(Exception):
    """Clean execution termination (no warnings).
    """
    def _render_traceback_(self):
        pass
##

def set_gpuoptions(n_gpus = 1,
                   gpus_list = None,
                   gpus_debug = False,
                   print_fn=logging.info):
    """Setup GPU usage and memory growth.
    
    Parameters
    ----------
    ngpus: int
        Number of GPUs to be used. (Default: 1)
    gpus_list: list, optional
        List of GPU IDs to be used, e.g. [0, 1, 2]. If `gpus_list` and `ngpus` 
        are both provided, `gpus_list` prevails. (Default: None)
    gpus_debug: bool
        Print out the GPUs ongoing usage. (Default: False)
    print_fn: {logging.info, print}
        Print out function. Either logging.info or print options are accepted.
        (Default: logging.info)

    Returns
    -------
    strategy:
        Memory growth strategy.
    logical_gpus: list
        List of logical GPUs.
    """
    
    # To find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(gpus_debug)
    if gpus_list is not None:
        gpu_ids = [int(iid) for iid in gpus_list]
    elif n_gpus>0:
        gpu_ids = [int(iid) for iid in range(n_gpus)]
    else:
        print_fn("Number of GPUs to be used is set to 0. Proceed with CPU.")
        print_fn("")
        device = "/cpu:0"
        strategy = tf.distribute.OneDeviceStrategy(device=device)
        devices = tf.config.list_logical_devices('CPU')
        return strategy, devices
        
    gpus = tf.config.experimental.list_physical_devices('GPU')    
    if gpus:
        try:
            # Keep only requested GPUs
            gpus = [gpus[i] for i in gpu_ids]
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            devices = tf.config.list_logical_devices('GPU')
            print_fn("{} Physical GPU(s), {} Logical GPU(s) detected and configured.".format(len(gpus), len(devices)))
        except RuntimeError as e: 
            print_fn(e)
                
        gpus_list_len = len(devices)
        if gpus_list_len > 0:
            if gpus_list_len > 1: 
                strategy = tf.distribute.MirroredStrategy()
            else:
                # Important! The command list_logical_devices renumerates the gpus starting from 0
                # The number here will be 0 regardless the requested GPU number
                device = "/gpu:0"
                strategy = tf.distribute.OneDeviceStrategy(device=device)
            print_fn('{} GPU device(s) will be used.'.format(strategy.num_replicas_in_sync))
            print_fn("")
            return strategy, devices
    else:
        device = "/cpu:0"
        strategy = tf.distribute.OneDeviceStrategy(device=device)
        devices = tf.config.list_logical_devices('CPU')
        print_fn("No GPU is detected in the system. Proceed with CPU.")
        print_fn("")
        return strategy, devices
#         print_fn("No GPU is detected in the system. SMILES-X needs at least one GPU to proceed.")
#         raise StopExecution
##

def log_setup(save_dir, name, verbose):
    """Setting up the logging format and files.

    Parameters
    ----------
    save_dir: str
        The directory where the logfile will be saved.
    name: str
        The name of the operation (train, inference, interpretation).
    verbose: bool
        Whether of now to printout the logs into console.

    Returns
    -------
    logger: logger
        Logger instance.
    logfile: str
        File to save the logs to.
    """
    
    # Setting up logging
    currentDT = datetime.datetime.now()
    strDT = currentDT.strftime("%Y-%m-%d_%H-%M-%S")
       
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)    
    formatter = logging.Formatter(fmt='%(asctime)s:   %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    
    # Remove existing handlers if any
    logger.handlers.clear()
    
    # Logging to the file
    logfile = '{}/{}_{}.log'.format(save_dir, name, strDT)
    handler_file = logging.FileHandler(filename=logfile, mode='w')
    handler_file.setLevel(logging.INFO)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    
    # Logging to console
    if verbose:
        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setLevel(logging.INFO)
        handler_stdout.setFormatter(formatter)
        logger.addHandler(handler_stdout)
    
    return logger, logfile
##

def rand_split(smiles_input, prop_input, extra_input, err_input, train_val_idx, test_idx, bayopt = False):
    """Splits into train, valid, test sets, and standardizes the target property (mean 0, std 1).

    Parameters
    ----------
    smiles_input: np.array
        Numpy array of SMILES.
    prop_input: np.array
        Numpy array of property to split.
    extra_input: np.array
        Numpy array of additional input data to split.
    err_input: np.array
        Numpy array of property error to split.
    train_val_idx: list(int)
        Indices of the data to be used for training and validation returned by KFold.
    valid_test_index: list(int)
        Indices of the data to be used for testing returned by KFold.
    bayopt: bool
        Whether or not Bayesian optimization is used. If True, a test set isn't returned.
        
    Returns
    -------
    x_train: np.array
        SMILES np.array for training.
    x_valid: np.array
        SMILES np.array for validation.
    x_test: np.array
        SMILES np.array for test.
    extra_train: np.array
        np.array of additional input data for training.
    extra_valid: np.array
        np.array of additional input data for validation.
    extra_test: np.array
        np.array of additional input data for test.
    y_train: np.array
        np.array of output data for training.
    y_valid: np.array
        np.array of output data for validation.
    y_test: np.array
        np.array of output data for test.
    err_train: np.array
        np.array of errors on output data for training.
    err_valid: np.array
        np.array of errors on output data for validation.
    err_test: np.array
        np.array of errors on output data for training.
    """
    
    # Assure random training/validation split (test set is unchanged)
    np.random.seed(42)
    np.random.shuffle(train_val_idx)
    
    # How many samples goes to training
    # We perform 7:2:1 split for train:val:test sets
    train_smpls = math.ceil(train_val_idx.shape[0]*6/9)
    
    train_idx = train_val_idx[:train_smpls]
    valid_idx = train_val_idx[train_smpls:]
    
    x_train = smiles_input[train_idx]
    y_train = prop_input[train_idx]
    extra_train = extra_input[train_idx] if extra_input is not None else None
    
    x_valid = smiles_input[valid_idx]
    y_valid = prop_input[valid_idx]
    extra_valid = extra_input[valid_idx] if extra_input is not None else None

    if bayopt:
        # No need of test set for Bayesian optimisation
        return x_train, x_valid, extra_train, extra_valid, y_train, y_valid
    
    x_test = smiles_input[test_idx]
    y_test = prop_input[test_idx]
    extra_test = extra_input[test_idx] if extra_input is not None else None
    
    # Only split when errors are provided
    err_test = err_input[test_idx] if err_input is not None else None
    err_train = err_input[train_idx] if err_input is not None else None
    err_valid = err_input[valid_idx] if err_input is not None else None

    logger.info("Train/valid/test splits: {0:0.2f}/{1:0.2f}/{2:0.2f}".format(\
                                          x_train.shape[0]/smiles_input.shape[0],\
                                          x_valid.shape[0]/smiles_input.shape[0],\
                                          x_test.shape[0]/smiles_input.shape[0]))
    logger.info("")
    
    return x_train, x_valid, x_test, extra_train, extra_valid, extra_test, y_train, y_valid, y_test, err_train, err_valid, err_test
##

def robust_scaler(train, valid, test, file_name, ifold):
    """Scale the output data and optionally saves scalers.
    
    Parameters
    ----------
    train: np.array
        Train set output property or extra input values.
    valid: np.array
        Validation set output property or extra input values.
    test: np.array
        Test set output property or extra input values.
    save_dir: str
        Path to the directory where the scalers should be saved (if `save`=True).
    data_name: str
        Data name (used for naming).
    ifold: int
        Current k-fold cross-validation fold index.
        
    Returns
    -------
    train_scaled: np.array
        Scaled train set output property or extra input values.
    valid_scaled: np.array
        Scaled validation set output property or extra input values.
    test_scaled: np.array
        Scaled test set output property or extra input values.
    scaler: sklearn.preprocessing.RobustScaler object
        Scaler to be used during the prediciton phase to unscale the outputs.
    """
    
    if ifold is not None:
        scaler_file = '{}_Fold_{}.pkl'.format(file_name, ifold)
        try:
            # If the scaler exists, load and make no changes
            scaler = pkl.load(open(scaler_file, "rb"))
        except (OSError, IOError) as e:
            # If doens't exist, create and fit to training data
            scaler = RobustScaler(with_centering=True, 
                                  with_scaling=True, 
                                  quantile_range=(5.0, 95.0), 
                                  copy=True)
            scaler_fit = scaler.fit(train)
            # Save scaler for future usage (e.g. inference)
            pkl.dump(scaler, open(scaler_file, "wb"))
            logger = logging.getLogger()
            logger.info("Scaler: {}".format(scaler_fit))
    else: # The scalers are not saved during Bayesian optimization
        scaler = RobustScaler(with_centering=True, 
                              with_scaling=True, 
                              quantile_range=(5.0, 95.0), 
                              copy=True)
        scaler_fit = scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    valid_scaled = scaler.transform(valid)
    if test is not None:
        test_scaled = scaler.transform(test)
    else:
        test_scaled = None
    
    return train_scaled, valid_scaled, test_scaled, scaler
##

def smiles_concat(smiles_list):
    """ Concatenate multiple SMILES in one via 'j'
    
    Parameters
    ----------
    smiles_list: array
        Array of SMILES to be concatenated along axis=0 to form a single SMILES.
    
    Returns
    -------
    concat_smiles_list
        List of concatenated SMILES, one per data point.
    """
    concat_smiles_list = []
    for smiles in smiles_list:
        concat_smiles_list.append('j'.join([ismiles for ismiles in smiles if ismiles != '']))
    return concat_smiles_list
##

def mean_result(smiles_enum_card, preds_enum, model_type = 'regression'):
    """Compute mean and median of predictions
    
    Parameters
    ----------
    smiles_enum_card: list(int)
        List of indices that are the same for the augmented SMILES originating from the same original SMILES
    preds_enum: np.array
        Predictions for every augmented SMILES for every predictive model
    model_type: str
        Type of the predictive model ('regression', 'binary_classification', 'multiclass_classification')

    Returns
    -------
        preds_mean: float
            Mean over predictions augmentations and models
        preds_std: float
            Standard deviation over predictions augmentations and models
    """
    
    if model_type == 'multiclass_classification' and len(preds_enum.shape) == 3:
        n_tile = preds_enum.shape[2]
        preds_enum = preds_enum.reshape(preds_enum.shape[0]*preds_enum.shape[2], preds_enum.shape[1])
        smiles_enum_card = np.tile(smiles_enum_card, n_tile)

    preds_ind = pd.DataFrame(preds_enum, index = smiles_enum_card)
    if model_type != 'multiclass_classification':
        preds_mean = preds_ind.groupby(preds_ind.index).apply(lambda x: np.mean(x.values)).values.flatten()
        preds_std = preds_ind.groupby(preds_ind.index).apply(lambda x: np.std(x.values)).values.flatten()
    else:
        preds_mean = preds_ind.groupby(preds_ind.index).apply(lambda x: np.mean(x.values, axis=0)).values.flatten()
        preds_std = preds_ind.groupby(preds_ind.index).apply(lambda x: np.std(x.values, axis=0)).values.flatten()
        preds_mean = np.stack(preds_mean)
        preds_std = np.stack(preds_std)

    return preds_mean, preds_std
##
