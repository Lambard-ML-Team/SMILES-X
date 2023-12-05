__version__ = '2.1'
__author__ = 'Guillaume Lambard, Ekaterina Gracheva'

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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
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

def generative_main(data_smiles,
                    data_name: str = 'Test',
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
        
    '''Generative SMILESX main pipeline
    
    Parameters
    ----------
    data_smiles: pd.DataFrame
        A single or multiple columns pandas dataframe of SMILES as inputs.
    data_name: str
        Dataset name used for naming directories and files related to an intended study with the SMILES-X. 
        A good practice is to use the name of the dataset as a prefix for the output files and directories. 
        (Default: 'Test')
    smiles_concat: bool
        Whether to concatenate multiple SMILES per entry with the token 'j' standing for 'join'. 
        This comes in handy if multiple SMILES are provided for a single entry, e.g. blends, etc.
        (Default: False)
    outdir: str
        Name of outputs directory. 
        (Default './outputs')
    geomopt_mode: {'on', 'off'}
        Whether to apply a trainless geometry optimisation for the LSTM, time-distributed
        dense, and embedding layers. 
        (Default 'off')
    bayopt_mode: {'on', 'off'}
        Whether to perform a Bayesian optimisation for the LSTM, time-distributed
        dense, and embedding layers, batch size and learning rate. If requested together with geometry
        optimisation, only batch size and learning rate will be optimized. 
        (Default: 'on')
    train_mode: {'on', 'finetune', 'off'}
        - 'on' for training from scratch on a given dataset.
        - 'finetune' for fine-tuning a pretrained model. Requires 'pretrained_data_name' and 'pretrained_augm'.
        - 'off' for just retrieving an existing trained model.
        
        (Default: 'train')
    pretrained_data_name: str
        The name of the data which was usd for pretraining.
        (Default: '')
    pretrained_augm: bool
        Whether augmentation was used or not during pretraining. It is used to build the path to the 
        pretrained model. 
        (Default: False)
    model_type: {'regression', 'classification'}
        Requests if the SMILES-X architecture should perform a regression, binary classification,
        or multiclass classification task. Basically, the activation function of the last layer
        will be set to 'linear', 'sigmoid', or 'softmax' respectively.
        (Default: 'regression') 
    scale_output: bool
        Whether to scale the output property values or not. For binary classification tasks,
        it is recommended not to scale the categorical (e.g. 0, 1) output values.
        For regression tasks, this is preferable to guarantee quicker training convergence.
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
        10^(-lr_ref). 
        (Default: 3.9)
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
    lr_schedule: {'decay', 'clr', 'cosine'}, optional
        Learning rate schedule
            - 'decay': step decay (see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay)
            - 'clr': cyclical (see https://arxiv.org/abs/1506.01186)
            - 'cosine': cosine annealing (see https://arxiv.org/abs/1608.03983)
            - None: No learning rate schedule applied
            
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
    For each run the following outputs will be saved in outdir:
        - tokens list (vocabulary, .txt)
        - geometry optimisation scores (.csv)
        - list of optimized hyperparameters (.csv)
        - best architecture (.hdf5)
        - learning curves (.png)

        For classification tasks:
            - confusion matrix (CM) plots for train, valid and test sets (.png)
            - precision-recall curve (PRC) plots for train, valid and test sets (.png)
            
    '''

    start_time = time.time()

    # Define and create output directories
    if train_mode=='finetune':
        main_save_dir = '{}/{}/{}/Transfer'.format(outdir, data_name, 'Augm' if augmentation else 'Can')
        save_dir = '{}/{}/{}/{}/Transfer'.format(outdir, data_name, 'LM', 'Augm' if augmentation else 'Can')
    else:
        main_save_dir = '{}/{}/{}/Train'.format(outdir, data_name, 'Augm' if augmentation else 'Can')
        save_dir = '{}/{}/{}/{}/Train'.format(outdir, data_name, 'LM', 'Augm' if augmentation else 'Can')
    other_dir = save_dir + '/Other'
    model_dir = save_dir + '/Models'
    res_plot_run_dir = save_dir + '/Figures/Results/Runs'
    lcurve_dir = save_dir + '/Figures/Learning_Curves'
    create_dirs = [model_dir, res_plot_run_dir, lcurve_dir, other_dir]
    for create_dir in create_dirs:
        if not os.path.exists(create_dir):
            os.makedirs(create_dir)

    # Setting up logger
    logger, logfile = utils.log_setup(save_dir, "Train", log_verbose)

    logging.info("***********************************")
    logging.info("***Generative SMILES-X starts...***")
    logging.info("***********************************")
    logging.info("")
    logging.info("")
    logging.info("The Generative SMILES-X logs can be found in the " + logfile + " file.")
    logging.info("")

    # Reading the data
    header = []
    data_smiles = data_smiles.replace([np.nan, None], ["", ""]).values
    data_smiles = data_smiles.reshape(-1,1)
    header.extend(["SMILES"])

    # Default model type for the generative SMILES-X
    model_type = 'multiclass_classification' 

    err_bars = None
    extra_dim = None
    
    # Initialize Predictions.txt and Scores.csv files
    predictions = pd.DataFrame(data_smiles)
    predictions.columns = header
    scores_folds = []

    logging.info("***Configuration parameters:***")
    logging.info("")
    logging.info("data =\n" + tabulate(predictions.head(), header))
    logging.info("data_name = \'{}\'".format(data_name))
    logging.info("smiles_concat = \'{}\'".format(smiles_concat))
    logging.info("outdir = \'{}\'".format(outdir))
    logging.info("pretrained_data_name = \'{}\'".format(pretrained_data_name))
    logging.info("pretrained_augm = \'{}\'".format(pretrained_augm))
    logging.info("model_type = \'{}\'".format(model_type))
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
        if n_runs is None:
            logging.info("The number of runs per fold (`n_runs`) is not defined.")
            logging.info("Borrowing it from the pretrained model...")
            logging.info("Number of runs `n_runs` is set to {}". format(model.n_runs))
        logging.info("Fine tuning has been requested, loading pretrained model...")
        # TODO (Guillaume): check if the pretrained model is loaded correctly 
        # pretrained_model = loadmodel.LoadModel(data_name = pretrained_data_name,
        #                                        outdir = outdir,
        #                                        augment = pretrained_augm,
        #                                        gpu_name = gpus[0].name,
        #                                        strategy = strategy, 
        #                                        return_attention=False, # no need to return attention for transfer learning
        #                                        extra = (data_extra!=None),
        #                                        scale_output=scale_output, 
        #                                        k_fold_number = k_fold_number)
        logging.info("Fine-tuning a pretrained model isn't implemented yet.")
        return
    else:
        if n_runs is None:
            logging.error("ERROR:")
            logging.error("The number of runs (`n_runs`) is not defined.")
            logging.error("")
            logging.error("*** SMILES-X EXECUTION ABORTED ***")
            raise utils.StopExecution
        pretrained_model = None

    # Setting up the loss and metrics according to the model_type
    model_loss = 'categorical_crossentropy'
    model_metrics = [metrics.categorical_accuracy, 
                     metrics.top_k_categorical_accuracy]
    
    # Keep track of the fold number for every data point ???
    # predictions.loc[test_idx, 'Fold'] = ifold

    # Check/augment the data if requested
    train_augm = augm.augmentation(data_smiles = data_smiles,
                                   indices = np.arange(data_smiles.shape[0]),
                                   check_smiles = check_smiles,
                                   augment = augmentation)
    
    x_train_enum, _, y_train_enum, y_train_clean, x_train_enum_card, _ = train_augm
    
    logging.info("Enumerated SMILES:")
    logging.info("\tTraining set: {}".format(len(x_train_enum)))
    logging.info("")

    logging.info("***Tokenization of SMILES.***")
    logging.info("")

    # Tokenize SMILES per dataset
    x_train_enum_tokens = token.get_tokens(x_train_enum)
    # Compute the length of each tokenized SMILES
    x_train_enum_tokens_len = [len(ismiles) for ismiles in x_train_enum_tokens]
    # Generate a list of tuples (index, token, length) for each tokenized SMILES
    x_train_enum_tokens_hash = [(ic, itoken, ilen-1) for ic, ilen in enumerate(x_train_enum_tokens_len) for itoken in range(1,ilen)]

    logging.info("Examples of tokenized SMILES from a training set:")
    logging.info("{}".format(x_train_enum_tokens[:5]))
    logging.info("")

    # Vocabulary size computation
    all_smiles_tokens = x_train_enum_tokens

    # Check if the vocabulary for current dataset exists already
    vocab_file = '{}/Other/{}_Vocabulary.txt'.format(main_save_dir, data_name)
    if os.path.exists(vocab_file):
        tokens = token.get_vocab(vocab_file)
    else:
        logging.info("No vocabulary file found for the current dataset.")
        logging.info("Extracting the vocabulary from the current dataset...")
        logging.info("")
        tokens = token.extract_vocab(all_smiles_tokens)
        vocab_file = '{}/{}_Vocabulary.txt'.format(other_dir, data_name)
        token.save_vocab(tokens, vocab_file)
        tokens = token.get_vocab(vocab_file)

    # TODO(kathya): add info on how much previous model vocabs differ from the current data train/val/test vocabs
    #               (for transfer learning)
    train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
    logging.info("Number of tokens only present in training set: {}".format(len(train_unique_tokens)))
    logging.info("")

    # Add 'pad' (padding), 'unk' (unknown) tokens to the existing list
    tokens.insert(0,'unk')
    tokens.insert(0,'pad')

    # n_class: number of classes allocated as the number of output nodes in the last layer of the generative SMILES-X model
    n_class = len(tokens)

    logging.info("Full vocabulary: {}".format(tokens))
    logging.info("Vocabulary size: {}".format(len(tokens)))
    logging.info("")

    # Maximum of length of SMILES to process
    all_smiles_tokens_len = [len(ismiles) for ismiles in all_smiles_tokens]
    max_length = np.max(all_smiles_tokens_len)
    median_length = np.median(all_smiles_tokens_len)
    logging.info("Maximum length of tokenized SMILES: {} tokens (termination spaces included)".format(max_length))
    logging.info("")

    # Convert tokenized SMILES to integer vectors
    x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_train_enum_tokens,
                                                        max_length=max_length + 1,
                                                        vocab=tokens)
                                                        
    # Hyperparameters retrieval from the geometry optimisation
    logging.info("*** HYPERPARAMETERS RETRIEVAL ***")
    logging.info("")

    # Dictionary to store optimized hyperparameters
    # Initialize at reference values, update gradually
    hyper_opt = {'Embedding': embed_ref,
                 'LSTM': lstm_ref,
                 'TD dense': tdense_ref,
                 'Batch size': bs_ref,
                 'Learning rate': lr_ref}
    hyper_opt_file = '{}/Other/{}_Hyperparameters.csv'.format(main_save_dir, data_name)
    if os.path.exists(hyper_opt_file):
        hyper_opt_df = pd.read_csv(hyper_opt_file)
        hyper_opt = hyper_opt_df.iloc[0].to_dict()
        # Batch size and learning rate optimised from inference task are ignored
        hyper_opt['Batch size'] = bs_ref  
        hyper_opt['Learning rate'] = lr_ref
        logging.info("File containing the list of optimised hyperparameters:")
        logging.info("    {}".format(hyper_opt_file))
        logging.info("")
        logging.info("The following hyperparameters will be used for training:")
        for key in hyper_opt.keys():
            if key == "Learning rate":
                logging.info("    - {}: 10^-{}".format(key, hyper_opt[key]))
            else:
                logging.info("    - {}: {}".format(key, hyper_opt[key]))
        logging.info("")
    else:
        logging.info("No geometry optimisation file found for the current dataset.")
        logging.info("Using reference values for hyperparameters.")
        logging.info("")

    logging.info("*** HYPERPARAMETERS RETRIEVAL COMPLETED ***")
    logging.info("")
    
    logging.info("*** TRAINING ***")
    logging.info("")

    for run in range(n_runs):
        start_run = time.time()
        
        # Estimate remaining training duration based on the first run duration
        if run > 0:
            if run == 1:
                onerun_time = time.time() - start_time # First run's duration
            elif run < (n_runs - 1):
                logging.info("Remaining time: {:.2f} h. Processing run #{} of data..."\
                                .format((n_runs - run) * onerun_time/3600., run))
            elif run == (n_runs - 1):
                logging.info("Remaining time: {:.2f} h. Processing the last run of data..."\
                                .format(onerun_time/3600.))

        # In case only some of the runs are requested for training
        if run_index is not None:
            if run not in run_index:
                continue

        logging.info("*** Run #{} ***".format(run))
        logging.info(time.strftime("%m/%d/%Y %H:%M:%S", time.localtime()))

        # Checkpoint, Early stopping and callbacks definition
        filepath = '{}/{}_Model_Run_{}_Best_Epoch.hdf5'.format(model_dir, data_name, run)
            
        if train_mode == 'off' or os.path.exists(filepath):
            logging.info("Training was set to `off`.")
            logging.info("Evaluating performance based on the previously trained models...")
            logging.info("")
        else:
            # Create and compile the model
            K.clear_session()
            # Freeze the first half of the network in case of transfer learning
            if train_mode == 'finetune':
                # model_train = model.model_dic['Fold_{}'.format(ifold)][run]
                # # Freeze encoding layers
                # #TODO(Guillaume): Check if this is the best way to freeze the layers as layers' name may differ
                # for layer in mod.layers:
                #     if layer.name in ['embedding', 'bidirectional', 'time_distributed']:
                #         layer.trainable = False
                # if run==0:
                #     logging.info("Retrieved model summary:")
                #     model_train.summary(print_fn=logging.info)
                #     logging.info("")
                logging.info("Fine-tuning a pretrained model isn't implemented yet.")
                return
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
                    custom_adam = Adam(lr=hyper_opt["Learning rate"])
                    model_train.compile(loss=model_loss, optimizer=custom_adam, metrics=model_metrics)
                if run==0:
                    logging.info("Model summary:")
                    model_train.summary(print_fn=logging.info)
                    logging.info("\n")

            batch_size = hyper_opt["Batch size"]
            if batchsize_pergpu is None:
                batch_size_list = np.array([int(2**itn) for itn in range(3,11)])
                batchsize_pergpu = batch_size_list[np.argmax((batch_size_list // median_length) == 1.)]
            batch_size = batchsize_pergpu * strategy.num_replicas_in_sync
            batchsize_pergpu = batch_size // strategy.num_replicas_in_sync
            logging.info("Total fixed batch size: {} ({} / gpu)\n".format(batch_size, batchsize_pergpu))
            logging.info("")

            # ignorebeginning = trainutils.IgnoreBeginningSaveBest(filepath=filepath,
            #                                                      n_epochs=n_epochs,
            #                                                      best_loss=np.Inf,
            #                                                      initial_epoch=0,
            #                                                      ignore_first_epochs=ignore_first_epochs)
            logcallback = trainutils.LoggingCallback(print_fcn=logging.info,verbose=train_verbose)
            
            filepath_tmp = model_dir+'/'+data_name+'_Model_Run_'+str(run)+'_Epoch_{epoch:02d}.hdf5'
            checkpoint = ModelCheckpoint(filepath_tmp, 
                                         monitor='loss', 
                                         verbose=0, 
                                         save_best_only=False, 
                                         mode='min')

            earlystopping = EarlyStopping(monitor='loss', 
                                          min_delta=0, 
                                          patience=patience, 
                                          verbose=0, 
                                          mode='min')
            # Default callback list
            #callbacks_list = [ignorebeginning, logcallback]
            callbacks_list = [checkpoint, earlystopping, logcallback]
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
                                          step_size=8*(x_train_enum_tokens_tointvec.shape[0] // batchsize_pergpu),
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
                                trainutils.LM_DataSequence(hash_set = x_train_enum_tokens_hash, 
                                                           smiles_set = x_train_enum_tokens_tointvec,
                                                           vocab_size = len(tokens),
                                                           max_length = max_length + 1,
                                                           batch_size=batch_size),
                                shuffle=False,
                                epochs=n_epochs,
                                callbacks=callbacks_list,
                                verbose=train_verbose,
                                max_queue_size=batch_size)

            # Summarize history for losses per epoch
            metrics_list = [m for m in history.history.keys() if not m.startswith("val_") and m != 'lr']
            for imetrics in metrics_list[:-1]:
                history_train_loss_tmp = history.history[imetrics]
                if imetrics == 'loss':
                    visutils.learning_curve(history_train_loss_tmp, None, lcurve_dir, data_name, None, run, model_type)
                else:
                    imetrics_p = metrics_list[-1]
                    history_train_metric_tmp = history.history[imetrics_p]
                    visutils.lm_metric_curve(history_train_metric_tmp, imetrics, imetrics_p, lcurve_dir, data_name, run)

            logging.info("\n\nBest loss @ Epoch #{}\n".format(np.argmin(history.history['loss'])))
            logging.info("")

            logging.info("***CxUxN scores history during the training.***\n")
            cun_metric = trainutils.CxUxN(init_data = data_smiles.flatten().tolist(), 
                                          data_name = data_name, 
                                          vocab = tokens, 
                                          gen_max_length = max_length+1, 
                                          gpus = gpus,
                                          model_init = model_train, 
                                          n_generate = 1000,
                                          warm_up = 0, 
                                          batch_size = 8096, 
                                          print_fcn = logging.info, 
                                          model_dir = model_dir, 
                                          run = run, 
                                          results_dir = res_plot_run_dir, 
                                          verbose = True)
            filepath_tmp_trunc = '{}/{}_Model_Run_{}_Epoch_*.hdf5'.format(model_dir, data_name, run)
            evaluated_epochs_list = glob.glob(filepath_tmp_trunc)
            for iepoch in range(len(evaluated_epochs_list)):
                cun_metric.evaluation(iepoch)
            cun_metric.on_evaluation_end()
            logging.info("")

        logging.info("Evaluating performance of the trained model...")
        logging.info("")

        # with tf.device(gpus[0].name):
        #     K.clear_session()
        #     model_train = load_model(filepath, custom_objects={'SoftAttention': model.SoftAttention()})
        #     model_train.compile(loss = model_loss, optimizer=Adam(), metrics=model_metrics)

        #     model_eval = model_train.evaluate(trainutils.LM_DataSequence(hash_set = x_train_enum_tokens_hash, 
        #                                                       smiles_set = x_train_enum_tokens_tointvec, 
        #                                                       vocab_size = len(tokens), 
        #                                                       max_length = max_length+1, 
        #                                                       batch_size = batchsize_pergpu), 
        #                                       verbose = 0)

        # logging.info("From the whole dataset:\n\t categorical_crossentropy loss: {0:0.4f} \n\t categorical accuracy: {1:0.4f} \n\t top_k_categorical accuracy (k=5): {2:0.4f}\n".\
        #     format(model_eval[0], model_eval[1], model_eval[2]))

        end_run = time.time()
        elapsed_run = end_run - start_run
        logging.info("Run {} duration: {}".format(run, str(datetime.timedelta(seconds=elapsed_run))))
        logging.info("")
            
    logging.info("******************************************************")
    logging.info("***Generative SMILES_X has terminated successfully.***")
    logging.info("******************************************************")

    end_all = time.time()
    elapsed_tot = end_all - start_time
    logging.info("Total elapsed time: {}".format(str(datetime.timedelta(seconds=elapsed_tot))))
    logging.shutdown()