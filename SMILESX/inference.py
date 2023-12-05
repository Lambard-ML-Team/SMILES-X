__version__ = '2.1'
__author__ = 'Guillaume Lambard, Ekaterina Gracheva'

import os
import glob
import logging

import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import Optional
from typing import List

from rdkit import Chem

from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from SMILESX import utils, token, augm

def infer(model, data_smiles, data_extra=None, augment=False, check_smiles: bool = True, smiles_concat: bool = False, batch_size: int = 32, log_verbose: bool = True):
    """Inference based on ensemble of trained SMILESX models

    Prediction of the property based on the ensemble of SMILESX models.
    Mean and standard deviation are computed over multiple models' predictions.

    Parameters
    ----------
    model: list
        The list of models to be used for inference.
    data_smiles: list(str)
        The list of SMILES to be characterized.
    data_extra:
        Additional data passed together with SMILES.
    smiles_concat: bool
        Whether to apply SMILES concatenation when multiple SMILES per entry are given.
    batch_size: int
        The batch size to be used for inference. (Default: 32)
    log_verbose: bool
        Whether to print the output to the console. (Default: True)
    check_smiles: bool
        Whether to check the SMILES via RDKit. (Default: True)

    Returns
    -------
    pd.DataFrame
        Dataframe of SMILES with their inferred property (SMILES, mean, standard deviation)
    """
    
    save_dir =  "{}/{}/{}/Inference/{}".format(model.outdir,
                                               model.data_name,
                                               'Augm' if model.augment else 'Can',
                                               'Augm' if augment else 'Can')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger, logfile = utils.log_setup(save_dir, 'Inference', log_verbose)
    
    logging.info("*************************************")
    logging.info("***   SMILESX INFERENCE STARTED   ***")
    logging.info("*************************************")
    logging.error("")
    
    logging.info("Inference logs path:")
    logging.info(logfile)
    logging.info("")

    if model.extra and data_extra is None:
        logging.error("ERROR:")
        logging.error("Additional input data has been used during the training of the loaded")
        logging.error("model, but none are provided for inference. Please, use `data_extra`")
        logging.error("to provide additional data.")
        logging.error("")
        logging.error("*** INFERENCE ABORTED ***")
        raise utils.StopExecution
    
    model_type = model.model_type
    n_class = model.n_class
    logging.info("model_type = \'{}\'".format(model_type))
    if model_type == 'multiclass_classification':
        logging.info("n_class = {}".format(n_class))
    logging.info("Full vocabulary: {}".format(model.tokens))
    logging.info("Vocabulary size: {}".format(len(model.tokens)))
    logging.info("Maximum length of tokenized SMILES: {} tokens.\n".format(model.max_length))

    data_smiles = np.array(data_smiles)
    if model.extra:
        data_extra = np.array(data_extra)
    # Checking and/or augmenting the SMILES if requested

    smiles_enum, extra_enum, _, _, smiles_enum_card, _ = augm.augmentation(data_smiles=data_smiles,
                                                                     indices=[i for i in range(len(data_smiles))],
                                                                     data_extra=data_extra,
                                                                     data_prop=None,
                                                                     check_smiles=check_smiles,
                                                                     augment=augment)

    # Concatenate multiple SMILES into one via 'j' joint
    if smiles_concat:
        smiles_enum = utils.smiles_concat(smiles_enum)
        
    logging.info("Number of enumerated SMILES: {}".format(len(smiles_enum)))
    logging.info("")
    logging.info("Tokenization of SMILES...")
    smiles_enum_tokens = token.get_tokens(smiles_enum)
    smiles_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=smiles_enum_tokens,
                                                       max_length=model.max_length,
                                                       vocab=model.tokens)
    # Model ensembling
    #preds_enum = np.empty((len(smiles_enum), model.k_fold_number*model.n_runs), dtype='float')
    for ifold in range(model.k_fold_number):
        # Scale additional data if provided
        if model.extra:
            # Load the scalers from pickle
            data_extra = model.extra_scaler_dic["Fold_{}".format(ifold)].transform(extra_enum)
        for run in range(model.n_runs):
            imodel = model.model_dic["Fold_{}".format(ifold)][run]
            # Predict and compare for the training, validation and test sets
            # Compute a mean per set of augmented SMILES
            if model.extra:
                ipred = imodel.predict({"smiles": smiles_enum_tokens_tointvec, "extra": extra_enum}, 
                                       batch_size=batch_size)
            else:
                ipred = imodel.predict({"smiles": smiles_enum_tokens_tointvec}, 
                                       batch_size=batch_size)
            if model.scale_output:
                # Unscale predictions
                ipred_unscaled = model.output_scaler_dic["Fold_{}".format(ifold)].inverse_transform(ipred.reshape(-1,1))
            else:
                ipred_unscaled = ipred
            # Store predictions in an array
            if ifold == 0 and run == 0:
                if model_type == 'multiclass_classification':
                    preds_enum = np.empty((len(smiles_enum)*n_class, model.k_fold_number*model.n_runs), dtype='float')
                else:
                    preds_enum = np.empty((len(smiles_enum), model.k_fold_number*model.n_runs), dtype='float')
            
            preds_enum[:, ifold * model.n_runs + run] = ipred_unscaled.flatten()

    if model_type == 'multiclass_classification':
        preds_enum = preds_enum.reshape(len(smiles_enum), n_class, model.k_fold_number*model.n_runs)  
    preds_mean, preds_std = utils.mean_result(smiles_enum_card, preds_enum, model_type)

    if model_type == 'multiclass_classification':
        preds_mean = np.argmax(preds_mean, axis=1)
        preds_std = preds_std[np.arange(len(preds_std)), preds_mean.tolist()]

    preds = pd.DataFrame()
    preds['SMILES'] = pd.DataFrame(data_smiles)
    preds['mean'] = preds_mean
    if model_type == 'binary_classification':
        preds['mean'] = (preds['mean'] > 0.5).astype("int8")
    preds['sigma'] = preds_std
    logging.info("")
    logging.info("Prediction results:\n" \
                 + tabulate(preds, ['SMILES', 'Prediction (mean)', 'Prediction (std)']))
    logging.info("")

    logging.info("***************************************")
    logging.info("***   SMILESX INFERENCE COMPLETED   ***")
    logging.info("***************************************")

    return preds
##