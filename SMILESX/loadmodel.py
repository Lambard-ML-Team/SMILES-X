"""Add main docstring discription

"""

import os
import glob
from pickle import load

import numpy as np
import pandas as pd

from typing import Optional
from typing import List

from rdkit import Chem

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# Ignore shape mismatch warnings related to the attention layer
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from SMILESX import utils, model, token, augm

class LoadModel:
    """
    Load an ensemble of trained models.

    Collects all the available trained models for a given dataset.
    Allows to keep the collected models in memory for multiple utilisation.
    Full models are collected for inference, truncated models are collected for interpretation
    (models are truncated at the level of attention, with attention vectors output).

    Verifies the availability of all the necessary files in the requested directory at initilisation.
    The following conditions must be fulfilled:
    -- Vocabulary file must exist.
       Without vocabulary file SMILES cannot be properly tokenized to be fed into a model.
    -- The number of scalers for the output data and additional input data (if used) should match.

    The number of folds `k_fold_number` and/or `n_runs` are automatically set based on the number of scalers 
    in the `Train/Other/Scalers/` directory.

    Parameters
    ----------
    data_name: str
        The name of the data used for training.
    augment: bool
        Whether augmentation has been used for training.
    outdir: str
        The path to the directory SMILES-X output directory. (Default: "./outputs")
    gpu_ind: int
        The index of the GPU to be used. (single GPU usage only). (Default: 0)
    gpu_name: str, optional
        The name of the GPU to be used. (Default: None)
    strategy:
        Memory growth strategy. (Default: None)
    log_verbose: bool
        Whether to output the logs to the console. If set to False, outputs will be
        directed towards the logfile only. (Default: True)
    return_attention: bool
        If `True`, collect full models (e.g., for inference). Otherwise collect 
        truncated models (e.g., for interpretation). (Default: True)

    Returns
    -------
        A class containing a dictionary of trained models per fold per run,
        output scalers, additional input data scalers if applicable, information
        on the number of k-folds and runs, as well as training data path.
    """

    def __init__(self,
                 data_name: str,
                 augment: bool,
                 outdir: str = "./outputs",
                 use_cpu: bool = False,
                 gpu_ind: int = 0,
                 gpu_name: str = None,
                 strategy = None,
                 log_verbose: bool = True,
                 return_attention: bool = True):
        self.data_name = data_name
        self.augment = augment
        self.outdir = outdir
        self.gpu_ind = gpu_ind
        self.log_verbose = log_verbose
        self.return_attention = return_attention

        self.train_dir = "{}/{}/{}/Train".format(self.outdir, self.data_name, 'Augm' if self.augment else 'Can')
        if gpu_name is not None:
            self.gpus = gpu_name
            self.strategy = strategy
        elif use_cpu:
            # CPUs options
            self.strategy, self.gpus = utils.set_gpuoptions(n_gpus=0,
                                                            gpus_debug=False,
                                                            print_fn=print)
        else:
            # GPUs options
            self.strategy, self.gpus = utils.set_gpuoptions(n_gpus=1,
                                                            gpus_list=[gpu_ind],
                                                            gpus_debug=False,
                                                            print_fn=print)
        # Verify path existance
        if not os.path.exists(self.train_dir):
            print("ERROR:")
            print("Path {} does not exist.\n".format(self.train_dir))
            print("HINT: check the data name and the augmentation flag.\n")
            print("")
            print("*** LOADING ABORTED ***")
            raise utils.StopExecution
            
        # Verify existance of vocabulary file
        vocab_file = '{}/Other/{}_Vocabulary.txt'.format(self.train_dir, self.data_name)
        if not os.path.exists(vocab_file):
            print("ERROR:")
            print("The input directory does not contain any vocabulary (*_Vocabulary.txt file).\n")
            print("")
            print("*** LOADING ABORTED ***")
            raise utils.StopExecution
        else:
            self.vocab_file = vocab_file
        
        # Load the model_type and the n_class
        with open('{}/Other/{}_model_type.txt'.format(self.train_dir, self.data_name), 'r') as f:
            self.model_type = f.readline().strip()
            self.n_class = int(f.readline().strip())
        
        scaler_dir = self.train_dir + '/Other/Scalers'
        model_dir = self.train_dir + '/Models'
        
        # Check whether additional data have been used for training
        self.scale_output = False
        n_output_scalers = len(glob.glob(scaler_dir + "/*Outputs*"))
        if n_output_scalers != 0:
            self.scale_output = True
        self.extra = False
        n_extra_scalers = len(glob.glob(scaler_dir + "/*Extra*"))
        if n_extra_scalers != 0:
            self.extra = True
        
        n_models = len(glob.glob(model_dir + "/*"))
        self.k_fold_number = len(glob.glob(model_dir + "/*Model_Fold_*_Run_0.hdf5"))
        self.n_runs = len(glob.glob(model_dir + "/*Model_Fold_0_Run_*.hdf5"))

        print("\nAll the required model files have been found.")
        
        # Load tokens from vocabulary file
        self.tokens = token.get_vocab(self.vocab_file)
        # Add 'pad', 'unk' tokens to the existing list
        self.tokens.insert(0, 'unk')
        self.tokens.insert(0, 'pad')
        
        # Setting up the scalers, trained models, and vocabulary
        self.att_dic = {}
        self.model_dic = {}
        if self.scale_output:
            self.output_scaler_dic = {}
        if self.extra:
            self.extra_scaler_dic = {}
        
        # Start loading models
        for ifold in range(self.k_fold_number):
            fold_model_list = []
            fold_att_list = []
            for run in range(self.n_runs):
                K.clear_session()
                model_file = '{}/{}_Model_Fold_{}_Run_{}.hdf5'.format(model_dir, self.data_name, ifold, run)
                model_tmp = load_model(model_file, custom_objects={'SoftAttention': model.SoftAttention()})
                fold_model_list.append(model_tmp)

                # Retrieve max_length
                if ifold == 0 and run == 0:
                    # Maximum of length of SMILES to process
                    self.max_length = model_tmp.layers[0].output_shape[-1][1]

                # For the attention, collect truncated
                if self.return_attention:
                    # Retrieve the geometry based on the trained model
                    embed_att = model_tmp.get_layer('embedding').output_shape[-1]
                    lstm_att = model_tmp.get_layer('bidirectional').output_shape[-1]//2
                    tdense_att = model_tmp.get_layer('time_distributed').output_shape[-1]

                    # Architecture to return attention weights
                    K.clear_session()
                    att_tmp = model.LSTMAttModel.create(input_tokens=self.max_length,
                                                        vocab_size=len(self.tokens),
                                                        embed_units=embed_att,
                                                        lstm_units=lstm_att,
                                                        tdense_units=tdense_att,
                                                        dense_depth=0,
                                                        return_prob=True)
                    att_tmp.load_weights(model_file, by_name=True, skip_mismatch=True)
                    intermediate_layer_model = Model(inputs=att_tmp.get_layer('smiles').input,
                                                     outputs=att_tmp.get_layer('attention').output)
                    fold_att_list.append(intermediate_layer_model)
            
            # Save models for the current fold
            self.model_dic['Fold_{}'.format(ifold)] = fold_model_list
            # Save truncated models for the current fold if requested
            if self.return_attention:
                self.att_dic['Fold_{}'.format(ifold)] = fold_att_list
            
            # Collect the scalers
            if self.scale_output:
                output_scaler_file = '{}/{}_Scaler_Outputs_Fold_{}.pkl'.format(scaler_dir, self.data_name, ifold)
                self.output_scaler_dic["Fold_{}".format(ifold)] = load(open(output_scaler_file, 'rb'))
            if self.extra:
                extra_scaler_file = '{}/{}_Scaler_Extra_Fold_{}.pkl'.format(scaler_dir, self.data_name, ifold)
                self.extra_scaler_dic["Fold_{}".format(ifold)] = load(open(extra_scaler_file, 'rb'))
                
        print("\n*** MODELS LOADED ***")
##