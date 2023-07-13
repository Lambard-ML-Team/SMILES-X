__version__ = '2.1'
__author__ = 'Guillaume Lambard, Ekaterina Gracheva'

"""Add main docstring discription
TODO(kathya): update the description
This script allows the user to ...
Ex. This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.
Ex. This script requires that `pandas` be installed within the Python
environment you are running this script in.
Ex. This file can also be imported as a module and contains the following
functions:
    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script
"""

# TODO (Guillaume) : check redundant imports against main.py
import os
import sys
import glob
import math
import time
import logging
import datetime
import collections
import pickle as pkl

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import List, Optional

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from sklearn.model_selection import GroupKFold, StratifiedKFold

from SMILESX import utils, token, augm
from SMILESX import model, bayopt, geomopt
from SMILESX import visutils, trainutils
from SMILESX import loadmodel

np.random.seed(seed=123)
np.set_printoptions(precision=3)
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def main(data_smiles,
         data_prop,
         data_err = None,
         data_extra = None,
         data_name: str = 'Test',
         data_units: str = '',
         data_label: str  = '',
         smiles_concat: bool = False,
         outdir: str = './outputs',
         geomopt_mode: str ='off',
         bayopt_mode: str = 'off',
         train_mode: str = 'on',
         pretrained_data_name: str = '',
         pretrained_augm: str = False,
         model_type = 'regression', 
         scale_output = True, 
         embed_bounds: Optional[List[int]] = None,
         lstm_bounds: Optional[List[int]] = None,
         tdense_bounds: Optional[List[int]] = None,
         nonlin_bounds: Optional[List[int]] = None,
         bs_bounds: Optional[List[int]] = None,
         lr_bounds: Optional[List[float]] = None,
         embed_ref: Optional[int] = 512,
         lstm_ref: Optional[int] = 128,
         tdense_ref: Optional[int] = 128,
         dense_depth: Optional[int] = 0,
         bs_ref: int = 16,
         lr_ref: float = 3.9,
         k_fold_number: Optional[int] = 5,
         k_fold_index: Optional[List[int]] = None,
         run_index: Optional[List[int]] = None,
         n_runs: Optional[int] = None,
         check_smiles: bool = True,
         augmentation: bool = False,
         geom_sample_size: int = 32,
         bayopt_n_rounds: int = 25,
         bayopt_n_epochs: int = 30,
         bayopt_n_runs: int = 3,
         n_gpus: int = 1,
         gpus_list: Optional[List[int]] = None,
         gpus_debug: bool = False,
         patience: int = 25,
         n_epochs: int = 100,
         batchsize_pergpu: Optional[int] = None,
         lr_schedule: Optional[str] = None,
         bs_increase: bool = False,
         ignore_first_epochs: int = 0,
         lr_min: float = 1e-5,
         lr_max: float = 1e-2,
         prec: int = 4,
         log_verbose: bool = True,
         train_verbose: bool = True) -> None:
    
    #TODO(Guillaume): Do not mention copolymers yet in smiles_concat but will be added later
    #TODO(Guillaume): Check utils.smiles_concat function to prevent misusage when one SMILES/entry
    #TODO(Guillaume): Review data_name docstring if a timestamp is incorporated in the files path

    #TODO(Guillaume): Add a caution message if already existing output results already exist
    #TODO(Guillaume): batchsize_pergpu option isn't used. Check it. 
    
    '''SMILESX main pipeline
    Parameters
    ----------data_extra
    data_smiles: single or multi columns pandas dataframe
        A single or multiple columns pandas dataframe of SMILES as inputs.
    data_prop: single column pandas dataframe
        A single column pandas dataframe of true property values as outputs, e.g. experimental measurements, etc. 
        It should have the same length than the provided set of input SMILES.
    data_err: single or multi columns pandas dataframe
        A single column dataframe of standard deviations, or a two columns dataframe of minimum and maximum values, 
        related to the property values. It should have the same length than the provided set of input SMILES. 
        If standard deviations are provided, symmetric errorbars will be plotted in the output figures. 
        And if [min, max] ranges are provided, asymmetric error bars will be plotted in accordance 
        with the min and max values.
        (Default: None)
    data_extra: single or multi columns pandas dataframe, optional
        A single or multi columns pandas dataframe of additional input data intended to be used as extra descriptors 
        joined to the SMILES alone. It should have the same length than the provided set of input SMILES.
        (Default: None)
    data_name: str
        Dataset name used for naming directories and files related to an intended study with the SMILES-X. 
        A good practice is to use the name of the dataset as a prefix for the output files and directories. 
        (Default: 'Test')
    data_units: str
        Desired output property's unit to be displayed in plots.
        (Default: '')
    data_label: str
        Desired output property's label to be displayed in plots. 
        The TeX formmat is supported according to the matplotlib's documentation (see 
        https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html).
        (Default: '')
    smiles_concat: bool
        Whether to concatenate multiple SMILES per entry with the token 'j' standing for 'join'. 
        This comes in handy if multiple SMILES are provided for a single entry, e.g. blends, etc.
        (Default: False)
    outdir: str
        Name of outputs directory. 
        (Default './outputs')
    geomopt_mode: {'on', 'off'}, str
        Whether to apply a trainless geometry optimisation for the LSTM, time-distributed
        dense, and embedding layers. 
        (Default 'off')
    bayopt_mode: {'on', 'off'}, str
        Whether to perform a Bayesian optimisation for the LSTM, time-distributed
        dense, and embedding layers, batch size and learning rate. If requested together with geometry
        optimisation, only batch size and learning rate will be optimized. 
        (Default: 'on')
    train_mode: {'on', 'finetune', 'off'}, str
        'on' for training from scratch on a given dataset.
        'finetune' for fine-tuning a pretrained model. Requires `pretrained_data_name` and `pretrained_augm`.
        'off' for just retrieving an existing trained model. 
        (Default: 'train')
    pretrained_data_name: str
        The name of the data which was usd for pretraining
        (Default: '')
    pretrained_augm: bool
        Whether augmentation was used or not during pretraining. It is used to build the path to the 
        pretrained model. 
        (Default: False)
    model_type: {'regression', 'classification'}, str 
        Requests if the SMILES-X architecture should perform a regression, binary classification, or multiclass classification task.  
        Basically, the activation function of the last layer will be set to 'linear', 'sigmoid', or 'softmax' respectively.
        (Default: 'regression') 
    scale_output: bool
        Whether to scale the output property values or not. For binary classification tasks, it is recommended not to scale 
        the categorical (e.g. 0, 1) output values. For regression tasks, this is preferable to guarantee quicker 
        training convergence.
        (Default: True)
    embed_bounds: list(int)
        Bounds constraining the Bayesian search for optimal embedding dimensions. 
        (Default: None)
    lstm_bounds: list(int)
        Bounds constraining the Bayesian search for optimal LSTM layer dimensions. 
        (Default: None)
    tdense_bounds: list(int), optional
        Bounds constraining the Bayesian search for optimal time-distributed dense layer dimensions. 
        (Default: None)
    bs_bounds: list(int), optional
        Bounds constraining the Bayesian search for optimal batch size. 
        (Default: None)
    lr_bounds: list(float), optional
        Bounds constraining the Bayesian search for optimal learning rate. 
        (Default: None)
    embed_ref: int
        User defined number of dimensions of the dense embedding. 
        (Default: 512)
    lstm_ref: int
        User defined number of LSTM units. 
        (Default: 128)
    tdense_ref: int
        User defined number of units of the time-distributed dense layer. 
        (Default: 128)
    dense_depth: int, optional
        Number of dense layers added to the architecture between the attention layer and the last dense layer. 
        The size of every consecutive layer is set to half the size of the previous one. 
        (Default: None)
    bs_ref: int
        User defined batch size (no Bayesian optimisation). 
        (Default: 16)
    lr_ref: int
        User defined learning rate (no Bayesian optimisation) translated to the Adam optimizer as 
        10**(-lr_ref). 
        (Default: 3.9)
    k_fold_number: int
        Number of folds used for a k-fold cross-validation. 
        (Default: 5)
    k_fold_index: list(int), optional
        List of indices of the folds of interest (e.g., [3, 5, 7]), other folds being skipped. 
        (Default: None)
    run_index: list(int), optional
        List of indices of the runs of interest (e.g., [3, 5, 7]), other runs being skipped. 
        (Default: None)
    n_runs: int, optional
        Number of training repetitions with random train/val split performed during Bayesian optimisation.
        (Default: None)
    check_smiles: bool
        Whether to check the validity of the SMILES strings with RDKit.
        (Default: True)
    augmentation: bool
        Whether to apply SMILES augmentation (i.e., de-canonicalization and exhaustive enumeration) or not. 
        This is recommended for small datasets and models' performance improvement, even if this will increase 
        the training time. 
        (Default: False)
    geom_sample_size: int
        Number of data samples used for trainless geometry evaluation.
        (Default: 32)
    bayopt_n_rounds: int
        Number of architectures to be sampled during Bayesian architecture search 
        (initialization + optimisation). 
        (Default: 25)    
    bayopt_n_epochs: int
        Number of epochs used for training during hyperparameters Bayesian optimisation. 
        (Default: 30)
    bayopt_n_runs: int
        Number of trainings performed for sampled architectures during hyperparameters 
        Bayesian optimisation to average performance per architecture. 
        (Default: 3)
    n_gpus: int
        Number of GPUs to be used in parallel. 
        (Default: 1)
    gpus_list: list[str], optional
        List of GPU IDs to be used, e.g. ['0','1','2']. 
        (Default: None)convert a list to a da
    gpus_debug: bool
        Print out the GPUs ongoing usage. 
        (Default: False)
    patience: int
        Used for early stopping. Patience is the number of epochs before stopping training when 
        the validation error has stopped improving. 
        (Default: 25)
    n_epochs: int
        Maximum number of epochs for training. 
        (Default: 100)
    batchsize_pergpu: int, optional
        Batch size used per GPU.
        If None, it is set in accordance with the augmentation statistics. 
        (Default: None)
    lr_schedule: {'decay', 'clr', 'cosine'}, str, optional
        Learning rate schedule
            'decay': step decay (see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay)
            'clr': cyclical (see https://arxiv.org/abs/1506.01186)
            'cosine': cosine annealing (see https://arxiv.org/abs/1608.03983)
             None: No learning rate schedule applied 
        (Default: None)
    bs_increase: bool
        Increase batch size econvert a list to a davery N steps (see https://arxiv.org/abs/1711.00489). 
        (Default: False)
    lr_min: float
        Minimum learning rate used during learning rate scheduling. 
        (Default: 1e-5)
    lr_max: float
        Maximum learning rate used during learning rate scheduling. 
        (Default: 1e-2)
    prec: int
        Precision of numerical values for printouts and plots. 
        (Default: 4)
    log_verbose: bool
        Whether or not to printout the logs into console.
        (Default: True)
    train_verbose: {0, 1, 2}
        Verbosity during training. 0 = silent, 1 = progress bar, 2 = one line per epoch 
        (see https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        (Default: 0)
    Returns
    -------
    For each fold and run the following outputs will be saved in outdir:
        -- tokens list (vocabulary) -> *.txt
        -- scaler -> *.pkl
        -- geometry optimisation scores -> Scores.csv
        -- list of optimized hyperparameters -> Optimized_Hyperparameters.csv
        -- best architecture -> *.hdf5
        -- training plot (loss vs epoch) ->convert a list to a da History_*.png
        For regression tasks:
            -- predictions vs observations plot -> TrainValidTest_*.png
        For classification tasks:
            -- Confusion matrix (CM) plot -> Train_CM_*.png, Valid_CM_*.png, Test_CM_*.png
            -- Precision-Recall curve (PRC) plot -> Train_PRC_*.png, Valid_PRC_*.png, Test_PRC_*.png
    '''

    # TODO(katia): update Returns list above
    start_time = time.time()

    # Define and create output directories
    if train_mode=='finetune':
        save_dir = '{}/{}/{}/Transfer'.format(outdir, data_name, 'Augm' if augmentation else 'Can')
    else:
        save_dir = '{}/{}/{}/Train'.format(outdir, data_name, 'Augm' if augmentation else 'Can')
    scaler_dir = save_dir + '/Other/Scalers'
    model_dir = save_dir + '/Models'
    pred_plot_run_dir = save_dir + '/Figures/Pred_vs_True/Runs'
    pred_plot_fold_dir = save_dir + '/Figures/Pred_vs_True/Folds'
    lcurve_dir = save_dir + '/Figures/Learning_Curves'
    create_dirs = [scaler_dir, model_dir, pred_plot_run_dir, pred_plot_fold_dir, lcurve_dir]
    for create_dir in create_dirs:
        if not os.path.exists(create_dir):
            os.makedirs(create_dir)

    # Setting up logger
    logger, logfile = utils.log_setup(save_dir, "Train", log_verbose)

    logging.info("************************")
    logging.info("***SMILES-X starts...***")
    logging.info("************************")
    logging.info("")
    logging.info("")
    logging.info("The SMILES-X logs can be found in the " + logfile + " file.")
    logging.info("")

    # Reading the data
    header = []
    data_smiles = data_smiles.replace([np.nan, None], ["", ""]).values
    if data_smiles.ndim==1:
        data_smiles = data_smiles.reshape(-1,1)
        header.extend(["SMILES"])
    elif data_smiles.shape[1]==1:
        data_smiles = data_smiles.reshape(-1,1)
        header.extend(["SMILES"])
    else:
        for i in range(data_smiles.shape[1]):
            header.extend(["SMILES_{}".format(i+1)])
    data_prop = data_prop.values

    # n_class: number of classes allocated as the number of output nodes in the last layer of the model for classification tasks
    unique_classes = np.unique(data_prop).tolist()
    if model_type == 'classification':
        n_class = len(unique_classes) 
        if n_class == 2:
            model_type = 'binary_classification'
            n_class = 1 # for binary_classification tasks with a sigmoid activation function at the output layer, output_n_nodes = n_class = 1
        elif n_class > 2:
            model_type = 'multiclass_classification' 
    else:
        n_class = 1 # for regression tasks with a linear activation function at the output layer, output_n_nodes = n_class = 1

    # save model_type and n_class to a file
    with open(save_dir + '/Other/'+ data_name +'_model_type.txt', 'w') as f:
        f.write(model_type)
        f.write('\n')
        f.write(str(n_class))

    header.extend([data_label])
    if data_err is not None:
        if data_err.ndim==1:
            data_err = data_err.reshape(-1,1)
        if data_err.shape[1] == 1:
            header.extend(["Standard deviation"])
            err_bars = 'std'
        elif data_err.shape[1] == 2:
            header.extend(["Minimum", "Maximum"])
            err_bars = 'minmax'
        data_err = data_err.values
    else:
        err_bars = None
    if data_extra is not None:
        header.extend(data_extra.columns)
        data_extra = data_extra.values
        extra_dim = data_extra.shape[1]
    else:
        extra_dim = None
    if data_label=='':
        data_label = data_name    

    # Initialize Predictions.txt and Scores.csv files
    predictions = np.concatenate([arr for arr in (data_smiles, data_prop.reshape(-1,1), data_err, data_extra) if arr is not None], axis=1)
    predictions = pd.DataFrame(predictions)
    predictions.columns = header
    scores_folds = []

    logging.info("***Configuration parameters:***")
    logging.info("")
    logging.info("data =\n" + tabulate(predictions.head(), header))
    logging.info("data_name = \'{}\'".format(data_name))
    logging.info("data_units = \'{}\'".format(data_units))
    logging.info("data_label = \'{}\'".format(data_label))
    logging.info("smiles_concat = \'{}\'".format(smiles_concat))
    logging.info("outdir = \'{}\'".format(outdir))
    logging.info("pretrained_data_name = \'{}\'".format(pretrained_data_name))
    logging.info("pretrained_augm = \'{}\'".format(pretrained_augm))
    logging.info("model_type = \'{}\'".format(model_type))
    if model_type == 'multiclass_classification':
        logging.info("n_class = {}".format(n_class))
    logging.info("scale_output = \'{}\'".format(scale_output))
    logging.info("geomopt_mode = \'{}\'".format(geomopt_mode))
    logging.info("bayopt_mode = \'{}\'".format(bayopt_mode))
    logging.info("train_mode = \'{}\'".format(bayopt_mode))
    logging.info("embed_bounds = {}".format(embed_bounds))
    logging.info("lstm_bounds = {}".format(lstm_bounds))
    logging.info("tdense_bounds = {}".format(tdense_bounds))
    logging.info("bs_bounds = {}".format(bs_bounds))
    logging.info("lr_bounds = {}".format(lr_bounds))
    logging.info("embed_ref = {}".format(embed_ref))
    logging.info("lstm_ref = {}".format(lstm_ref))
    logging.info("tdense_ref = {}".format(tdense_ref))
    logging.info("dense_depth = {}".format(dense_depth))
    logging.info("bs_ref = {}".format(bs_ref))
    logging.info("lr_ref = {}".format(lr_ref))
    logging.info("k_fold_number = {}".format(k_fold_number))
    logging.info("k_fold_index = {}".format(k_fold_index))
    logging.info("run_index = {}".format(run_index))
    logging.info("n_runs = {}".format(n_runs))
    logging.info("augmentation = {}".format(augmentation))
    logging.info("geom_sample_size = {}".format(geom_sample_size))
    logging.info("bayopt_n_rounds = {}".format(bayopt_n_rounds))
    logging.info("bayopt_n_epochs = {}".format(bayopt_n_epochs))
    logging.info("bayopt_n_runs = {}".format(bayopt_n_runs))
    logging.info("n_gpus = {}".format(n_gpus))
    logging.info("gpus_list = {}".format(gpus_list))
    logging.info("gpus_debug = {}".format(gpus_debug))
    logging.info("patience = {}".format(patience))
    logging.info("n_epochs = {}".format(n_epochs))
    logging.info("batchsize_pergpu = {}".format(batchsize_pergpu))
    logging.info("lr_schedule = {}".format(lr_schedule))
    logging.info("bs_increase = {}".format(bs_increase))
    logging.info("ignore_first_epochs = {}".format(ignore_first_epochs))
    logging.info("lr_min = {}".format(lr_min))
    logging.info("lr_max = {}".format(lr_max))
    logging.info("prec = {}".format(prec))
    logging.info("log_verbose = {}".format(log_verbose))
    logging.info("train_verbose = {}".format(train_verbose))
    logging.info("******")
    logging.info("")

    # Setting up GPUs
    strategy, gpus = utils.set_gpuoptions(n_gpus=n_gpus,
                                          gpus_list=gpus_list,
                                          gpus_debug=gpus_debug)
    if strategy is None:
        raise utils.StopExecution
    logging.info("{} Logical GPU(s) detected and configured.".format(len(gpus)))
    logging.info("")

    # Setting up the scores summary
    scores_summary = {'train': [],
                      'valid': [],
                      'test': []}

    if ignore_first_epochs >= n_epochs:
            logging.error("ERROR:")
            logging.error("The number of ignored epochs `ignore_first_epochs` should be less than")
            logging.error("the total number of training epochs `n_epochs`.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise utils.StopExecution

    # Retrieve the models for training in case of transfer learning
    if train_mode == 'finetune':
        if len(pretrained_data_name) == 0:
            logging.error("ERROR:")
            logging.error("Cannot determine the pretrained model path.")
            logging.error("Please, specify the name of the data used for the pretraining (`pretrained_data_name`)")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise utils.StopExecution
        if k_fold_number is None:
            # If the dataset is too small to transfer the number of kfolds
            if model.k_fold_number > data.shape[0]:
                k_fold_number = data.shape[0]
                logging.info("The number of cross-validation folds (`k_fold_number`) is not defined.")
                logging.info("Borrowing it from the pretrained model...")
                logging.info("Number of folds `k_fold_number` is set to {}". format(k_fold_number))
            else:
                k_fold_number = model.k_fold_number
                logging.info("The number of cross-validation folds (`k_fold_number`)")
                logging.info("used for the pretrained model is too large to be used with current data:")
                logging.info("size of the data is too small ({} > {})".format(model.k_fold_number, data.shape[0]))
                logging.info("The number of folds is set to the length of the data ({})". format(k_fold_number))
        if n_runs is None:
            logging.info("The number of runs per fold (`n_runs`) is not defined.")
            logging.info("Borrowing it from the pretrained model...")
            logging.info("Number of runs `n_runs` is set to {}". format(model.n_runs))
        logging.info("Fine tuning has been requested, loading pretrained model...")
        pretrained_model = loadmodel.LoadModel(data_name = pretrained_data_name,
                                               outdir = outdir,
                                               augmentation = pretrained_augm,
                                               gpu_name = gpus[0].name,
                                               strategy = strategy, 
                                               return_attention=False, # no need to return attention for transfer learning
                                               model_type = model_type,
                                               extra = (data_extra!=None),
                                               scale_output=scale_output, 
                                               k_fold_number = k_fold_number)
    else:
        if k_fold_number is None:
            logging.error("ERROR:")
            logging.error("The number of cross-validation folds (`k_fold_number`) is not defined.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise utils.StopExecution
        if n_runs is None:
            logging.error("ERROR:")
            logging.error("The number of runs per fold (`n_runs`) is not defined.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise utils.StopExecution
        pretrained_model = None

    # Setting up the cross-validation according to the model_type
    # For regression
    # Splitting is done based on groups of the provided SMILES data
    # This is done for the cases where the same SMILES has multiple entries with
    # varying additional parameters (molecular weight, proportion, processing time, etc.)
    # For classification
    # Splitting is done based on the provided property (e.g. class) data
    if model_type == 'regression':
        groups = pd.DataFrame(data_smiles).groupby(by=0).ngroup().values.tolist()
        kf = GroupKFold(n_splits=k_fold_number)
        kf.get_n_splits(X=data_smiles, groups=groups)
        kf_splits = kf.split(X=data_smiles, groups=groups)
        model_loss = 'mse'
        model_metrics = [metrics.mae, metrics.mse]
    else:
        kf = StratifiedKFold(n_splits=k_fold_number, shuffle=True, random_state=42)
        kf.get_n_splits(X=data_smiles, y=data_prop)
        kf_splits = kf.split(X=data_smiles, y=data_prop)
        if model_type == 'binary_classification':
            model_loss = 'binary_crossentropy'
        elif model_type == 'multiclass_classification':
            model_loss = 'sparse_categorical_crossentropy'
        model_metrics = ['accuracy']
     
    # Individual counter for the folds of interest in case of k_fold_index
    nfold = 0
    for ifold, (train_val_idx, test_idx) in enumerate(kf_splits):
        start_fold = time.time()

        # In case only some of the folds are requested for training
        if k_fold_index is not None:
            k_fold_number = len(k_fold_index)
            if ifold not in k_fold_index:
                continue
        
        # Keep track of the fold number for every data point
        predictions.loc[test_idx, 'Fold'] = ifold

        # Estimate remaining training duration based on the first fold duration
        if nfold > 0:
            if nfold == 1:
                onefold_time = time.time() - start_time # First fold's duration
            elif nfold < (k_fold_number - 1):
                logging.info("Remaining time: {:.2f} h. Processing fold #{} of data..."\
                             .format((k_fold_number - nfold) * onefold_time/3600., ifold))
            elif nfold == (k_fold_number - 1):
                logging.info("Remaining time: {:.2f} h. Processing the last fold of data..."\
                             .format(onefold_time/3600.))

        logging.info("")
        logging.info("***Fold #{} initiated...***".format(ifold))
        logging.info("")
        
        logging.info("***Splitting and standardization of the dataset.***")
        logging.info("")
        x_train, x_valid, x_test, \
        extra_train, extra_valid, extra_test, \
        y_train, y_valid, y_test, \
        y_err_train, y_err_valid, y_err_test = utils.rand_split(smiles_input = data_smiles,
                                                                prop_input = data_prop,
                                                                extra_input = data_extra,
                                                                err_input = data_err,
                                                                train_val_idx = train_val_idx,
                                                                test_idx = test_idx)
        # Scale the outputs
        if scale_output:
            scaler_out_file = '{}/{}_Scaler_Outputs'.format(scaler_dir, data_name)
            y_train_scaled, y_valid_scaled, y_test_scaled, scaler = utils.robust_scaler(train=y_train,
                                                                                        valid=y_valid,
                                                                                        test=y_test,
                                                                                        file_name=scaler_out_file,
                                                                                        ifold=ifold)
        else:
            y_train_scaled, y_valid_scaled, y_test_scaled, scaler = y_train, y_valid, y_test, None
        # Scale the auxiliary numeric inputs (if given) 
        if data_extra is not None:
            scaler_extra_file = '{}/{}_Scaler_Extra'.format(scaler_dir, data_name)
            extra_train, extra_valid, extra_test, scaler_extra = utils.robust_scaler(train=extra_train,
                                                                                     valid=extra_valid,
                                                                                     test=extra_test,
                                                                                     file_name=scaler_extra_file,
                                                                                     ifold=ifold)

        # Check/augment the data if requested
        train_augm = augm.augmentation(x_train,
                                       train_val_idx,
                                       extra_train,
                                       y_train_scaled,
                                       check_smiles,
                                       augmentation)

        valid_augm = augm.augmentation(x_valid,
                                       train_val_idx,
                                       extra_valid,
                                       y_valid_scaled,
                                       check_smiles,
                                       augmentation)

        test_augm = augm.augmentation(x_test,
                                      test_idx,
                                      extra_test,
                                      y_test_scaled,
                                      check_smiles,
                                      augmentation)
        
        x_train_enum, extra_train_enum, y_train_enum, y_train_clean, x_train_enum_card, _ = train_augm
        x_valid_enum, extra_valid_enum, y_valid_enum, y_valid_clean, x_valid_enum_card, _ = valid_augm
        x_test_enum, extra_test_enum, y_test_enum, y_test_clean, x_test_enum_card, test_idx_clean = test_augm
                
        # Concatenate multiple SMILES into one via 'j' joint
        if smiles_concat:
            x_train_enum = utils.smiles_concat(x_train_enum)
            x_valid_enum = utils.smiles_concat(x_valid_enum)
            x_test_enum = utils.smiles_concat(x_test_enum)
        
        logging.info("Enumerated SMILES:")
        logging.info("\tTraining set: {}".format(len(x_train_enum)))
        logging.info("\tValidation set: {}".format(len(x_valid_enum)))
        logging.info("\tTest set: {}".format(len(x_test_enum)))
        logging.info("")

        logging.info("***Tokenization of SMILES.***")
        logging.info("")

        # Tokenize SMILES per dataset
        x_train_enum_tokens = token.get_tokens(x_train_enum)
        x_valid_enum_tokens = token.get_tokens(x_valid_enum)
        x_test_enum_tokens = token.get_tokens(x_test_enum)

        logging.info("Examples of tokenized SMILES from a training set:")
        logging.info("{}".format(x_train_enum_tokens[:5]))
        logging.info("")

        # Vocabulary size computation
        all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens

        # Check if the vocabulary for current dataset exists already
        vocab_file = '{}/Other/{}_Vocabulary.txt'.format(save_dir, data_name)
        if os.path.exists(vocab_file):
            tokens = token.get_vocab(vocab_file)
        else:
            tokens = token.extract_vocab(all_smiles_tokens)
            token.save_vocab(tokens, vocab_file)
            tokens = token.get_vocab(vocab_file)

        # TODO(kathya): add info on how much previous model vocabs differ from the current data train/val/test vocabs
        #               (for transfer learning)
        train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
        logging.info("Number of tokens only present in training set: {}".format(len(train_unique_tokens)))
        logging.info("")

        valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
        logging.info("Number of tokens only present in validation set: {}".format(len(valid_unique_tokens)))
        if valid_unique_tokens.issubset(train_unique_tokens):
            logging.info("Validation set contains no new tokens comparing to training set tokens")
        else:
            logging.info("Validation set contains the following new tokens comparing to training set tokens:")
            logging.info(valid_unique_tokens.difference(train_unique_tokens))
            logging.info("")

        test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
        logging.info("Number of tokens only present in a test set: {}".format(len(test_unique_tokens)))
        if test_unique_tokens.issubset(train_unique_tokens):
            logging.info("Test set contains no new tokens comparing to the training set tokens")
        else:
            logging.info("Test set contains the following new tokens comparing to the training set tokens:")
            logging.info(test_unique_tokens.difference(train_unique_tokens))

        if test_unique_tokens.issubset(valid_unique_tokens):
            logging.info("Test set contains no new tokens comparing to the validation set tokens")
        else:
            logging.info("Test set contains the following new tokens comparing to the validation set tokens:")
            logging.info(test_unique_tokens.difference(valid_unique_tokens))
            logging.info("")

        # Add 'pad' (padding), 'unk' (unknown) tokens to the existing list
        tokens.insert(0,'unk')
        tokens.insert(0,'pad')

        logging.info("Full vocabulary: {}".format(tokens))
        logging.info("Vocabulary size: {}".format(len(tokens)))
        logging.info("")

        # Maximum of length of SMILES to process
        max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
        logging.info("Maximum length of tokenized SMILES: {} tokens (termination spaces included)".format(max_length))
        logging.info("")

        # predict and compare for the training, validation and test sets
        x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_train_enum_tokens,
                                                            max_length=max_length + 1,
                                                            vocab=tokens)
        x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_valid_enum_tokens,
                                                            max_length=max_length + 1,
                                                            vocab=tokens)
        x_test_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_test_enum_tokens,
                                                           max_length=max_length + 1,
                                                           vocab=tokens)
        # Hyperparameters optimisation
        if nfold==0:
            logging.info("*** HYPERPARAMETERS OPTIMISATION ***")
            logging.info("")

        # Dictionary to store optimized hyperparameters
        # Initialize at reference values, update gradually
            hyper_opt = {'Embedding': embed_ref,
                         'LSTM': lstm_ref,
                         'TD dense': tdense_ref,
                         'Batch size': bs_ref,
                         'Learning rate': lr_ref}
            hyper_bounds = {'Embedding': embed_bounds,
                            'LSTM': lstm_bounds,
                            'TD dense': tdense_bounds,
                            'Batch size': bs_bounds,
                            'Learning rate': lr_bounds}
            
            # Geometry optimisation
            if geomopt_mode == 'on':
                geom_file = '{}/Other/{}_GeomScores.csv'.format(save_dir, data_name)
                # Do not optimize the architecture in case of transfer learning
                if train_mode=='finetune':
                    logging.info("Transfer learning is requested together with geometry optimisation,")
                    logging.info("but the architecture is already fixed in the original model.")
                    logging.info("Only batch size and learning rate can be tuned.")
                    logging.info("Skipping geometry optimisation...")
                    logging.info("")
                else:
                    hyper_opt, hyper_bounds = \
                    geomopt.geom_search(data_token=x_train_enum_tokens_tointvec,
                                        data_extra=extra_train_enum,
                                        subsample_size=geom_sample_size,
                                        hyper_bounds=hyper_bounds,
                                        hyper_opt=hyper_opt,
                                        dense_depth=dense_depth,
                                        vocab_size=len(tokens),
                                        max_length=max_length,
                                        geom_file=geom_file,
                                        strategy=strategy, 
                                        model_type='regression') # model_type for regression is by default fitted to the geom_search function
            else:
                logging.info("Trainless geometry optimisation is not requested.")
                logging.info("")

             # Bayesian optimisationmodel_type
            if bayopt_mode == 'on':
                if geomopt_mode == 'on':
                    logging.info("*Note: Geometry-related hyperparameters will not be updated during the Bayesian optimisation.")
                    logging.info("")
                    if not any([bs_bounds, lr_bounds]):
                        logging.info("Batch size bounds and learning rate bounds are not defined.")
                        logging.info("Bayesian optimisation has no parameters to optimize.")
                        logging.info("Skipping...")
                        logging.info("")
                hyper_opt = bayopt.bayopt_run(smiles=data_smiles,
                                              prop=data_prop,
                                              extra=data_extra,
                                              train_val_idx=train_val_idx,
                                              smiles_concat=smiles_concat,
                                              tokens=tokens,
                                              max_length=max_length,
                                              check_smiles=check_smiles,
                                              augmentation=augmentation,
                                              hyper_bounds=hyper_bounds,
                                              hyper_opt=hyper_opt,
                                              dense_depth=dense_depth,
                                              bo_rounds=bayopt_n_rounds,
                                              bo_epochs=bayopt_n_epochs,
                                              bo_runs=bayopt_n_runs,
                                              strategy=strategy,
                                              model_type=model_type, 
                                              output_n_nodes=n_class, 
                                              scale_output=scale_output, 
                                              pretrained_model=pretrained_model)
            else:
                logging.info("Bayesian optimisation is not requested.")
                logging.info("")
                if geomopt == 'off':
                    logging.info("Using reference values for training.")
                    logging.info("")

            hyper_df = pd.DataFrame([hyper_opt.values()], columns = hyper_opt.keys())
            hyper_file = "{}/Other/{}_Hyperparameters.csv".format(save_dir, data_name)
            hyper_df.to_csv(hyper_file, index=False)

            logging.info("*** HYPERPARAMETERS OPTIMISATION COMPLETED ***")
            logging.info("")
            
            logging.info("The following hyperparameters will be used for training:")
            for key in hyper_opt.keys():
                if key == "Learning rate":
                    logging.info("    - {}: 10^-{}".format(key, hyper_opt[key]))
                else:
                    logging.info("    - {}: {}".format(key, hyper_opt[key]))
            logging.info("")
            logging.info("File containing the list of used hyperparameters:")
            logging.info("    {}".format(hyper_file))
            logging.info("")

            logging.info("*** TRAINING ***")
            logging.info("")
        start_train = time.time()
        if model_type != 'multiclass_classification':
            prediction_train_bag = np.zeros((y_train_enum.shape[0], n_runs))
            prediction_valid_bag = np.zeros((y_valid_enum.shape[0], n_runs))
            prediction_test_bag = np.zeros((y_test_enum.shape[0], n_runs))
        else:
            prediction_train_bag = np.zeros((y_train_enum.shape[0], n_class, n_runs))
            prediction_valid_bag = np.zeros((y_valid_enum.shape[0], n_class, n_runs))
            prediction_test_bag = np.zeros((y_test_enum.shape[0], n_class, n_runs))
        
        for run in range(n_runs):
            start_run = time.time()

            # In case only some of the runs are requested for training
            if run_index is not None:
                if run not in run_index:
                    continue

            logging.info("*** Run #{} ***".format(run))
            logging.info(time.strftime("%m/%d/%Y %H:%M:%S", time.localtime()))

            # Checkpoint, Early stopping and callbacks definition
            filepath = '{}/{}_Model_Fold_{}_Run_{}.hdf5'.format(model_dir, data_name, ifold, run)
                
            if train_mode == 'off' or os.path.exists(filepath):
                logging.info("Training was set to `off`.")
                logging.info("Evaluating performance based on the previously trained models...")
                logging.info("")
            else:
                # Create and compile the model
                K.clear_session()
                # Freeze the first half of the network in case of transfer learning
                if train_mode == 'finetune':
                    model_train = model.model_dic['Fold_{}'.format(ifold)][run]
                    # Freeze encoding layers
                    #TODO(Guillaume): Check if this is the best way to freeze the layers as layers' name may differ
                    for layer in mod.layers:
                        if layer.name in ['embedding', 'bidirectional', 'time_distributed']:
                            layer.trainable = False
                    if (nfold==0 and run==0):
                        logging.info("Retrieved model summary:")
                        model_train.summary(print_fn=logging.info)
                        logging.info("")
                elif (train_mode == 'train' or train_mode == 'on'):
                    with strategy.scope():
                        model_train = model.LSTMAttModel.create(input_tokens=max_length+1,
                                                                extra_dim=extra_dim,
                                                                vocab_size=len(tokens),
                                                                embed_units=hyper_opt["Embedding"],
                                                                lstm_units=hyper_opt["LSTM"],
                                                                tdense_units=hyper_opt["TD dense"],
                                                                dense_depth=dense_depth,
                                                                model_type=model_type, 
                                                                output_n_nodes=n_class)
                        custom_adam = Adam(lr=math.pow(10,-float(hyper_opt["Learning rate"])))
                        model_train.compile(loss=model_loss, optimizer=custom_adam, metrics=model_metrics)
                    if (nfold==0 and run==0):
                        logging.info("Model summary:")
                        model_train.summary(print_fn=logging.info)
                        logging.info("\n")

                batch_size = hyper_opt["Batch size"]
                if bs_increase:
                    # Apply batch increments schedule in accordance with the paper by S.Smith, Q.Le,
                    # "Don't decay the learning rate, increase the batch size"
                    # https://arxiv.org/abs/1711.00489
                    logging.info("Batch size increment option is selected,")
                    logging.info("learning rate schedule will NOT be applied.")
                    logging.info("")
                    if ignore_first_epochs >= n_epochs:
                        logging.info("The number of epochs to be ignored, `ignore_first_epochs`,")
                        logging.info("should be strictly less than the total number of epochs to train, `n_epochs`.")
                        raise utils.StopExecution

                    # Setting up the batch size schedule
                    # Increment batch twofold every 1/3 of the total of epochs (heuristic)
                    batch_size_schedule = [int(batch_size), int(batch_size*2), int(batch_size*4)]
                    n_epochs_schedule = [int(n_epochs/3), int(n_epochs/3), n_epochs - 2*int(n_epochs/3)]

                    # Fit the model applying the batch size schedule:

                    # Keeping track of history
                    # During BS increments model is trained 3 times, histories should be stitched manually
                    history_train_loss = []
                    history_val_loss = []

                    # Define callbacks
                    n_epochs_done = 0
                    best_loss = np.Inf
                    best_epoch = 0
                    logging.info("Training:")
                    for i, batch_size in enumerate(batch_size_schedule):
                        if i == (len(batch_size_schedule) - 1):
                            last = True
                        else:
                            last = False
                        n_epochs_part = n_epochs_schedule[i]
                        # Ignores noise fluctuations of the beginning of the training
                        # Avoids picking up undertrained model
                        # TODO: add early stopping to ignorebeginning
                        ignorebeginning = trainutils.IgnoreBeginningSaveBest(filepath=filepath,
                                                                             n_epochs=n_epochs_part,
                                                                             best_loss=best_loss,
                                                                             best_epoch=best_epoch,
                                                                             initial_epoch=n_epochs_done,
                                                                             ignore_first_epochs=ignore_first_epochs,
                                                                             last=last)
                        logcallback = trainutils.LoggingCallback(print_fcn=logging.info,verbose=train_verbose)
                        # Default callback list
                        callbacks_list = [ignorebeginning, logcallback]
                        with strategy.scope():
                            if i == 0:
                                logging.info("The batch size is initialized at {}".format(batch_size))
                                logging.info("")
                            else:
                                logging.info("")
                                logging.info("The batch size is changed to {}".format(batch_size))
                                logging.info("")
                            history = model_train.fit(\
                                      trainutils.DataSequence(x_train_enum_tokens_tointvec,
                                                              extra_train_enum,
                                                              props=y_train_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      validation_data = \
                                      trainutils.DataSequence(x_valid_enum_tokens_tointvec,
                                                              extra_valid_enum,
                                                              props=y_valid_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      shuffle=True,
                                      initial_epoch=n_epochs_done,
                                      epochs=n_epochs_done + n_epochs_part,
                                      callbacks=callbacks_list,
                                      verbose=train_verbose,
                                      max_queue_size=batch_size,
                                      use_multiprocessing=False,
                                      workers=1)
                        history_train_loss += history.history['loss']
                        history_val_loss += history.history['val_loss']
                        best_loss = ignorebeginning.best_loss
                        best_epoch = ignorebeginning.best_epoch
                        n_epochs_done += n_epochs_part
                else:
                    ignorebeginning = trainutils.IgnoreBeginningSaveBest(filepath=filepath,
                                                                         n_epochs=n_epochs,
                                                                         best_loss=np.Inf,
                                                                         initial_epoch=0,
                                                                         ignore_first_epochs=ignore_first_epochs)
                    logcallback = trainutils.LoggingCallback(print_fcn=logging.info,verbose=train_verbose)
                    # Default callback list
                    callbacks_list = [ignorebeginning, logcallback]
                    # Additional callbacks
                    if lr_schedule == 'decay':
                        schedule = trainutils.StepDecay(initAlpha=lr_max,
                                                        finalAlpha=lr_min,
                                                        gamma=0.95,
                                                        epochs=n_epochs)
                        callbacks_list.append(LearningRateScheduler(schedule))
                    elif lr_schedule == 'clr':
                        clr = trainutils.CyclicLR(base_lr=lr_min,
                                                  max_lr=lr_max,
                                                  step_size=8*(x_train_enum_tokens_tointvec.shape[0] // \
                                                              (batch_size//strategy.num_replicas_in_sync)),
                                                  mode='triangular')
                        callbacks_list.append(clr)
                    elif lr_schedule == 'cosine':
                        cosine_anneal = trainutils.CosineAnneal(initial_learning_rate=lr_max,
                                                                final_learning_rate=lr_min,
                                                                epochs=n_epochs)
                        callbacks_list.append(cosine_anneal)

                    # Fit the model
                    with strategy.scope():
                        history = model_train.fit(\
                                      trainutils.DataSequence(x_train_enum_tokens_tointvec,
                                                              extra_train_enum,
                                                              props=y_train_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      validation_data = \
                                      trainutils.DataSequence(x_valid_enum_tokens_tointvec,
                                                              extra_valid_enum,
                                                              props=y_valid_enum,
                                                              batch_size=batch_size * strategy.num_replicas_in_sync),
                                      shuffle=True,
                                      initial_epoch=0,
                                      epochs=n_epochs,
                                      callbacks=callbacks_list,
                                      verbose=train_verbose,
                                      max_queue_size=batch_size,
                                      use_multiprocessing=False,
                                      workers=1)
                    history_train_loss = history.history['loss']
                    history_val_loss = history.history['val_loss']

                # Summarize history for losses per epoch
                visutils.learning_curve(history_train_loss, history_val_loss, lcurve_dir, data_name, ifold, run, model_type)

                logging.info("Evaluating performance of the trained model...")
                logging.info("")

            with tf.device(gpus[0].name):
                K.clear_session()
                model_train = load_model(filepath, custom_objects={'SoftAttention': model.SoftAttention()})
                if data_extra is not None:
                    y_pred_train = model_train.predict({"smiles": x_train_enum_tokens_tointvec, "extra": extra_train_enum})
                    y_pred_valid = model_train.predict({"smiles": x_valid_enum_tokens_tointvec, "extra": extra_valid_enum})
                    y_pred_test = model_train.predict({"smiles": x_test_enum_tokens_tointvec, "extra": extra_test_enum})
                else:
                    y_pred_train = model_train.predict({"smiles": x_train_enum_tokens_tointvec})
                    y_pred_valid = model_train.predict({"smiles": x_valid_enum_tokens_tointvec})
                    y_pred_test = model_train.predict({"smiles": x_test_enum_tokens_tointvec})

            # Unscale prediction outcomes
            if scale_output:
                y_pred_train_unscaled = scaler.inverse_transform(y_pred_train.reshape(-1,1)).ravel()
                y_pred_valid_unscaled = scaler.inverse_transform(y_pred_valid.reshape(-1,1)).ravel()
                y_pred_test_unscaled = scaler.inverse_transform(y_pred_test.reshape(-1,1)).ravel()
                
                y_train_clean_unscaled = scaler.inverse_transform(y_train_clean.reshape(-1,1)).ravel()
                y_valid_clean_unscaled = scaler.inverse_transform(y_valid_clean.reshape(-1,1)).ravel()
                y_test_clean_unscaled = scaler.inverse_transform(y_test_clean.reshape(-1,1)).ravel()
            else:
                y_pred_train_unscaled = y_pred_train.ravel() if model_type != 'multiclass_classification' else y_pred_train
                y_pred_valid_unscaled = y_pred_valid.ravel() if model_type != 'multiclass_classification' else y_pred_valid
                y_pred_test_unscaled = y_pred_test.ravel() if model_type != 'multiclass_classification' else y_pred_test

                y_train_clean_unscaled = y_train_clean.ravel()
                y_valid_clean_unscaled = y_valid_clean.ravel()
                y_test_clean_unscaled = y_test_clean.ravel()

            if model_type != 'multiclass_classification':
                prediction_train_bag[:, run] = y_pred_train_unscaled
                prediction_valid_bag[:, run] = y_pred_valid_unscaled
                prediction_test_bag[:, run]  = y_pred_test_unscaled
            else:
                prediction_train_bag[:,:, run] = y_pred_train_unscaled
                prediction_valid_bag[:,:, run] = y_pred_valid_unscaled
                prediction_test_bag[:,:, run]  = y_pred_test_unscaled

            # Compute average per set of augmented SMILES for the plots per run
            y_pred_train_mean_augm, y_pred_train_std_augm = utils.mean_result(x_train_enum_card, y_pred_train_unscaled, model_type)
            y_pred_valid_mean_augm, y_pred_valid_std_augm = utils.mean_result(x_valid_enum_card, y_pred_valid_unscaled, model_type)
            y_pred_test_mean_augm, y_pred_test_std_augm = utils.mean_result(x_test_enum_card, y_pred_test_unscaled, model_type)

            # Print the stats for the run
            visutils.print_stats(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                                 preds=[y_pred_train_mean_augm, y_pred_valid_mean_augm, y_pred_test_mean_augm],
                                 errs_pred=[y_pred_train_std_augm, y_pred_valid_std_augm, y_pred_test_std_augm],
                                 prec=prec, 
                                 model_type=model_type, 
                                 labels = unique_classes)

            # Plot prediction vs observation plots per run
            visutils.plot_fit(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                              preds=[y_pred_train_mean_augm, y_pred_valid_mean_augm, y_pred_test_mean_augm],
                              errs_true=[y_err_train, y_err_valid, y_err_test],
                              errs_pred=[y_pred_train_std_augm, y_pred_valid_std_augm, y_pred_test_std_augm],
                              err_bars=err_bars,
                              save_dir=save_dir,
                              dname=data_name,
                              dlabel=data_label,
                              units=data_units,
                              fold=ifold,
                              run=run, 
                              model_type=model_type)

            end_run = time.time()
            elapsed_run = end_run - start_run
            logging.info("Fold {}, run {} duration: {}".format(ifold, run, str(datetime.timedelta(seconds=elapsed_run))))
            logging.info("")

        # Averaging predictions over augmentations and runs
        pred_train_mean, pred_train_sigma = utils.mean_result(x_train_enum_card, prediction_train_bag, model_type)
        pred_valid_mean, pred_valid_sigma = utils.mean_result(x_valid_enum_card, prediction_valid_bag, model_type)
        pred_test_mean, pred_test_sigma = utils.mean_result(x_test_enum_card, prediction_test_bag, model_type)

        #Save the predictions to the final table
        if model_type == 'multiclass_classification':
            pred_test_mean_argmax = np.argmax(pred_test_mean, axis=1).ravel()
            predictions.loc[test_idx_clean, 'Mean'] = pred_test_mean_argmax
            predictions.loc[test_idx_clean, 'Standard deviation'] = pred_test_sigma[np.arange(len(pred_test_sigma)), pred_test_mean_argmax.tolist()].ravel()
        else:
            predictions.loc[test_idx_clean, 'Mean'] = pred_test_mean.ravel()
            predictions.loc[test_idx_clean, 'Standard deviation'] = pred_test_sigma.ravel()
        predictions.to_csv('{}/{}_Predictions.csv'.format(save_dir, data_name), index=False)
        
        logging.info("Fold {}, overall performance:".format(ifold))

        # Print the stats for the fold
        fold_scores = visutils.print_stats(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                                           preds=[pred_train_mean, pred_valid_mean, pred_test_mean],
                                           errs_pred=[pred_train_sigma, pred_valid_sigma, pred_test_sigma],
                                           prec=prec, 
                                           model_type=model_type, 
                                           labels = unique_classes)        
        
        scores_folds.append([err for set_name in fold_scores for err in set_name])

        # Plot prediction vs observation plots for the fold
        visutils.plot_fit(trues=[y_train_clean_unscaled, y_valid_clean_unscaled, y_test_clean_unscaled],
                          preds=[pred_train_mean, pred_valid_mean, pred_test_mean],
                          errs_true=[y_err_train, y_err_valid, y_err_test],
                          errs_pred=[pred_train_sigma, pred_valid_sigma, pred_test_sigma],
                          err_bars=err_bars,
                          save_dir=save_dir,
                          dname=data_name,
                          dlabel=data_label,
                          units=data_units,
                          fold=ifold,
                          run=None, 
                          model_type=model_type)

        end_fold = time.time()
        elapsed_fold = end_fold - start_fold
        logging.info("Fold {} duration: {}".format(ifold, str(datetime.timedelta(seconds=elapsed_fold))))
        logging.info("")

        if ifold == (k_fold_number-1) and not k_fold_index:
            logging.info("*******************************")
            logging.info("***Predictions score summary***")
            logging.info("*******************************")
            logging.info("")

            logging.info("***Preparing the final out-of-sample prediction.***")
            logging.info("")
            
            data_prop_clean = data_prop[predictions['Mean'].notna()]
            predictions = predictions.dropna()

            print(predictions['Mean'].values)
            # Print the stats for the whole data
            final_scores = visutils.print_stats(trues=[data_prop_clean],
                                                preds=[predictions['Mean'].values],
                                                errs_pred=[predictions['Standard deviation'].values],
                                                prec=prec, 
                                                model_type=model_type, 
                                                labels = unique_classes)
            
            scores_final = [err for set_name in final_scores for err in set_name]
            
            # Final plot for prediction vs observation
            visutils.plot_fit(trues=[data_prop_clean.reshape(-1,1)],
                              preds=[predictions['Mean'].values],
                              errs_true=[data_err],
                              errs_pred=[predictions['Standard deviation'].values],
                              err_bars=err_bars,
                              save_dir=save_dir,
                              dname=data_name,
                              dlabel=data_label,
                              units=data_units,
                              final=True, 
                              model_type=model_type)
            
            if model_type == 'regression':
                scores_list = ['RMSE', 'MAE', 'R2-score']
            elif model_type.split('_')[1] == 'classification':
                scores_list = ['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PRC-AUC']

            scores_folds = pd.DataFrame(scores_folds)
            if model_type == 'regression':
                scores_folds.columns = pd.MultiIndex.from_product([['Train', 'Valid', 'Test'],\
                                                                   scores_list,\
                                                                   ['Mean', 'Sigma']])
            elif model_type.split('_')[1] == 'classification':
                scores_folds.columns = pd.MultiIndex.from_product([['Train', 'Valid', 'Test'],\
                                                                   ['Micro avg', 'Macro avg', 'Weighted avg'],\
                                                                   scores_list,\
                                                                   ['Mean', 'Sigma']])

            scores_folds.index.name = 'Fold'
            scores_folds.to_csv('{}/{}_Scores_Folds.csv'.format(save_dir, data_name))
            
            scores_final = pd.DataFrame(scores_final).T
            if model_type == 'regression':
                scores_final.columns = pd.MultiIndex.from_product([scores_list,\
                                                                ['Mean', 'Sigma']])
            else:
                scores_final.columns = pd.MultiIndex.from_product([['Micro avg', 'Macro avg', 'Weighted avg'],\
                                                                   scores_list,\
                                                                   ['Mean', 'Sigma']])
                                                                   
            scores_final.to_csv('{}/{}_Scores_Final.csv'.format(save_dir, data_name), index=False)
        
        nfold += 1
        
    logging.info("*******************************************")
    logging.info("***SMILES_X has terminated successfully.***")
    logging.info("*******************************************")

    end_all = time.time()
    elapsed_tot = end_all - start_time
    logging.info("Total elapsed time: {}".format(str(datetime.timedelta(seconds=elapsed_tot))))
    logging.shutdown()