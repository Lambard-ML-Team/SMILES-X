__version__ = '2.1'
__author__ = 'Guillaume Lambard, Ekaterina Gracheva'

import os
import time
import glob
import logging
import datetime
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
from numpy import random

import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

from SMILESX import model, utils

def geom_search(data_token, data_extra, subsample_size, hyper_bounds, hyper_opt, dense_depth, vocab_size, max_length, geom_file, strategy, model_type):
    """Trainless geometry optimization.
    
    The geometry optimization is performed by using a method of zero-cost model evaluation,
    similar to the following paper: 
    E Gracheva, Array, Volume 12, 2021, 100082.

    Candidate architectures are initialized with several shared constant weights,
    and the one showing the lowest value of the coefficient of variation over the initializations,
    normalized by the coefficient of variation over a fixed batch, is selected for training.

    Only geometry-related hyperparameters are optimized (number of units in the LSTM layer, 
    time-distributed dense layer and embedding dimensionality). Other parameters should be optimized
    via Bayesian optimization.

    Parameters
    ----------
    data_token: list
        List of tokenized SMILES (training set data are passed from the main.py).
    data_extra: numpy.array, optional
        2D array with additional input data.
    subsample_size: int
        Size of the data sample used for trainless geometry evaluation.
    hyper_bounds: dict
        Dictionary of format {param_name:list(bounds)} containing hyperparameter bounds, 
        defines the search space.
    hyper_opt: dict
        Dictionary of format {param_name:value}. Initially the values are set to default ones,
        then they are updated with optimal values during optimization.
    dense_depth: int
        The number of additional dense layers added after attention to deepen the network.
        Can be used with and without `data_extra` data.
    vocab_size: int
        The size of the vocabulary.
    max_length: int
        Maximum SMILES length within the data.
    geom_file: str
        The name of the path to save the geometry scores.
    strategy:
        GPU strategy to be used. 
    model_type: str
        The type of the model to be used. Can be either 'regression' or 'classification'.

    Returns
    -------
    hyper_opt: dict
        Hyperparameters dictionary with geometry-related hyperparameters values updated.
    hyper_bounds: dict
        Hyperparameters dictionary with geometry-related bounds set to `None` to prevent
        repeated optimization during consequent Bayesian optimization.
    """
    
    start = time.time()

    if data_extra is not None:
        extra_dim = data_extra.shape[1]
    else:
        extra_dim = None

    # Check whether all the principal geometry-related bounds have been given
    if None in [hyper_bounds[key] for key in ['Embedding', 'LSTM', 'TD dense']]:
        logging.warning("ATTENTION! Geometry optimisation is requested, but not all the bounds are given.")
        logging.info("")
        logging.warning("Please specify the following bounds:")
        logging.warning("\t  - embed_bounds")
        logging.warning("\t  - lstm_bounds")
        logging.warning("\t  - tdense_bounds")
        logging.info("")
        raise utils.StopExecution

    geom_bounds = {}
    for key in ['Embedding', 'LSTM', 'TD dense']:
        geom_bounds[key] = hyper_bounds[key]

    # Get the logger
    logger = logging.getLogger()
    logging.info("Geometry will be optimized via trainless geometry optimization.")
    logging.info("")
    logging.info("*** Geometry optimization ***")
    logging.info("")

    np.random.seed(21)
    # Random sampling
    picked_ind = np.random.choice(range(len(data_token)), subsample_size)
    data = data_token[picked_ind, :]

    if data_extra is not None:
        extra = data_extra[picked_ind, :]
    else:
        extra = None
        
    if os.path.exists(geom_file):
        logging.info("*Note: Geometry optimization for the current dataset has been performed before.")
        logging.info("       Refer to the geometry score file:")
        logging.info("       {}".format(geom_file))
        logging.info("")
        logging.info("       Retrieving the best geometry hyperparameters from the score file")
        logging.info("")
        scores = pd.read_csv(geom_file)
        for key in ['Embedding', 'LSTM', 'TD dense']:
            hyper_opt[key] = scores.loc[0, key]  # Updates optimal hyperparameters dictionary
            hyper_bounds[key] = None # Prevents repeated optimization via Bayesian optimization
        return hyper_opt, hyper_bounds
    else:
        scores = []
        best_weights = geom_prescore(data,
                                     extra,
                                     geom_bounds,
                                     extra_dim,
                                     dense_depth,
                                     vocab_size,
                                     max_length,
                                     strategy,
                                     model_type
                                    )
#         best_weights = [1e-05, 0.1]
#         # Estimate optimal weights by looking at 30 random architectures
#         for n in range(20):
#             geom = [random.choice(g) for g in geom_bounds.values()]
#             best_weights = geom_prescore(data,
#                                          extra,
#                                          geom,
#                                          extra_dim,
#                                          dense_depth,
#                                          vocab_size,
#                                          max_length,
#                                          strategy, 
#                                          model_type)
#             weights.append(best_weights)
# 
#         overall_weights = pd.DataFrame([i for line in weights for i in line])
#         weight_range = overall_weights.stack().value_counts().index.tolist()[1:6]

        # Extensively score all the available architectures
        for geom in itertools.product(*geom_bounds.values()):
            score, nparams = geom_score(data,
                                        extra,
                                        geom,
                                        extra_dim,
                                        dense_depth,
                                        best_weights,
                                        vocab_size,
                                        max_length,
                                        strategy, 
                                        model_type)
            # print([score, nparams] + list(geom))
            scores.append([score, nparams] + list(geom))

    scores = pd.DataFrame(scores)
    scores.columns = ['Score', 'Number of parameters'] + [key for key in geom_bounds.keys() if geom_bounds[key] is not None]
    # Sort according to the score value in descending order
    scores = scores.sort_values(by = 'Score', ascending=False).reset_index(drop=True)

    for key in ['Embedding', 'LSTM', 'TD dense']:
        hyper_opt[key] = scores.loc[0, key] # Update optimal hyperparameters dictionary
        hyper_bounds[key] = None # Prevent repeated optimization via Bayesian optimization

    scores.to_csv(geom_file, index=False)
    logging.info("Sorted scores\n{}".format(tabulate(scores.head(), scores.columns)))

    end = time.time()
    elapsed = end - start
    logging.info("Geometry search duration: {}".format(str(datetime.timedelta(seconds=elapsed))))

    return hyper_opt, hyper_bounds
##

# def geom_prescore(data, extra, geom, extra_dim, dense_depth, vocab_size, max_length, strategy, model_type):
#     """Find the optimal weights for the trainless geometry search
    
#     Several architectures are randomly selected among all the possible combinations
#     and initialized with extensive range of weights. The least correlated weights are
#     selected for each architecture.
    
#     Parameters
#     ----------
#     data:
#         List of tokenized SMILES (training set data are passed from the main.py).
#     extra: numpy.array, optional
#         2D array with additional numeric data.
#     geom: list
#         List of values for embedding, LSTM and time-distributed dense sizes,
#         defining a single geometry to be tested.
#     extra_dim: int
#         Dimensionality of the additional input data.
#     dense_depth: int
#         The number of additional dense layers added after attention to deepen the network.
#         Can be used with and without `data_extra` data.
#     vocab_size: int
#         The size of the vocabulary.
#     max_length: int
#         Maximum SMILES length within the data.
#     strategy:
#         GPU strategy to be used.
#     model_type: str
#         Type of the model to be used. Can be either 'regression' or 'classification'.
        
#     Returns
#     -------
#     best_weights: list
#         List of the least correlated weight for the requested geometry
#     """
    
#     embed_units, lstm_units, tdense_units = geom

#     # Test range
#     weight_range = [i*10**j for j in range(-6,6) for i in [1,3,6]]
#     best_weights = []
#     ranks = []
#     for weight in weight_range:
#         K.clear_session()
#         model_geom = model.LSTMAttModel.create(input_tokens=max_length + 1,
#                                                extra_dim=extra_dim,
#                                                vocab_size=vocab_size,
#                                                embed_units=embed_units,
#                                                lstm_units=lstm_units,
#                                                tdense_units=tdense_units,
#                                                dense_depth=dense_depth,
#                                                geom_search=True,
#                                                weight=weight, 
#                                                model_type=model_type)
#         if extra is not None:
#             pred = model_geom.predict({"smiles": data, "extra": extra}, verbose=0)
#         else:
#             pred = model_geom.predict({"smiles": data}, verbose=0)
#         rank = np.argsort(pred.flatten())
#         ranks.append(rank)

#     ranks = np.array(ranks)
#     corr_ranks = stats.spearmanr(ranks, axis=1)[0]
#     corr_ranks_triu = np.triu(corr_ranks, k=1)
#     to_keep = np.sort(np.argsort(np.max(corr_ranks_triu, axis=1))[:6])
#     weight_range_tmp = [weight_range[j] for j in to_keep]
#     best_weights.append(weight_range_tmp)

#     return best_weights
# ##

def geom_prescore(data, extra, geom, extra_dim, dense_depth, vocab_size, max_length, strategy, model_type):
    """Find the optimal weights for the trainless geometry search
    
    Several architectures are randomly selected among all the possible combinations
    and initialized with extensive range of weights. The least correlated weights are
    selected for each architecture.
    
    Parameters
    ----------
    data:
        List of tokenized SMILES (training set data are passed from the main.py).
    extra: numpy.array, optional
        2D array with additional numeric data.
    geom: list
        List of values for embedding, LSTM and time-distributed dense sizes,
        defining a single geometry to be tested.
    extra_dim: int
        Dimensionality of the additional input data.
    dense_depth: int
        The number of additional dense layers added after attention to deepen the network.
        Can be used with and without `data_extra` data.
    vocab_size: int
        The size of the vocabulary.
    max_length: int
        Maximum SMILES length within the data.
    strategy:
        GPU strategy to be used.
    model_type: str
        The type of the model to be used. Can be either 'regression' or 'classification'.
        
    Returns
    -------
    best_weights: list
        List of the least correlated weight for the requested geometry
    """

    LOW = -4
    HIGH = 4
    window_size = HIGH-LOW

    while window_size != 0:
        efficiencies = []
        for w in range(LOW, HIGH+1-window_size):
            weights = [10**w, 10**(w+window_size)]
            scores = []
            # Keep the same random set of architectures across tested weights
            np.random.seed(21)
            for it in range(100):
                # Pick up a random set of architectures, keep it same for all weights
                embed_units, lstm_units, tdense_units = [random.choice(g) for g in geom.values()]
                preds = []
                for weight in weights:
                    K.clear_session()
                    model_geom = model.LSTMAttModel.create(input_tokens=max_length + 1,
                                                           extra_dim=extra_dim,
                                                           vocab_size=vocab_size,
                                                           embed_units=embed_units,
                                                           lstm_units=lstm_units,
                                                           tdense_units=tdense_units,
                                                           dense_depth=dense_depth,
                                                           geom_search=True,
                                                           weight=weight,
                                                           model_type=model_type)

                    if extra is not None:
                        pred = model_geom.predict({"smiles": data, "extra": extra}, verbose=0)
                    else:
                        pred = model_geom.predict({"smiles": data}, verbose=0)
                    pred = pred.flatten()
                    pred_norm = (pred - np.nanmin(pred)) / (np.nanmax(pred) - np.nanmin(pred))
                    preds.append(pred_norm)
                    
                preds = np.array(preds)
                preds[np.where(preds==0)] = np.nan
                mae = np.nanmean(np.abs(preds[0,:]-preds[1,:]))
                mean = np.nanmean(preds)
                score = mae/mean
                
                scores.append(score)
            # Verify how many architectures got NaN scores
            print(weights, np.sum(np.isnan(scores))/len(scores))
            if np.sum(np.isnan(scores))/len(scores) < 0.1:
                return weights
        window_size -= 1

#             efficiencies.append((weights, np.sum(np.isnan(scores))/len(scores)))
#         efficiencies = sorted(efficiencies, key=lambda i:i[1])
#         if efficiencies[0][1]<0.5:
#             print("Final weights: ", efficiencies[0][0])
#             return efficiencies[0][0]
#         window_size -= 1
##

def geom_score(data, extra, geom, extra_dim, dense_depth, weights, vocab_size, max_length, strategy, model_type):
    """Find the optimal weights for the trainless geometry search
    
    Several architectures are randomly selected among all the possible combinations
    and initialized with extensive range of weights. The least correlated weights are
    selected for each architecture.
    
    Parameters
    ----------
    data:
        List of tokenized SMILES (training set data are passed from the main.py).
    extra: numpy.array, optional
        2D array with additional numeric data.
    geom: list
        List of values for embedding, LSTM and time-distributed dense sizes,
        defining a single geometry to be tested.
    extra_dim: int
        Dimensionality of the additional input data.
    dense_depth: int
        The number of additional dense layers added after attention to deepen the network.
        Can be used with and without `data_extra` data.
    weight: list
        List of weights to be used for constant shared weight initialization.
    vocab_size: int
        The size of the vocabulary.
    max_length: int
        Maximum SMILES length within the data.
    strategy:
        GPU strategy to be used.
    model_type: str
        Type of the model to be used. Can be either 'regression' or 'classification'.
        
    Returns
    -------
    score: float
        Estimated score for the requested geometry.
    nparams: int
        Number of trainable parameters contained within the requested geometry.
    """
    
    embed_units, lstm_units, tdense_units = geom

    # Working range
    preds = []
    for weight in weights:
        K.clear_session()
        with strategy.scope():
            model_geom = model.LSTMAttModel.create(input_tokens=max_length + 1,
                                                   extra_dim=extra_dim,
                                                   vocab_size=vocab_size,
                                                   embed_units=embed_units,
                                                   lstm_units=lstm_units,
                                                   tdense_units=tdense_units,
                                                   dense_depth=dense_depth,
                                                   geom_search=True,
                                                   weight=weight, 
                                                   model_type = model_type)
            if extra is not None:
                pred = model_geom.predict({"smiles": data, "extra": extra}, verbose=0)
            else:
                pred = model_geom.predict({"smiles": data}, verbose=0)
            pred_min = np.nanmin(pred)
            pred_max = np.nanmax(pred)
            pred_norm = (pred - pred_min)/(pred_max - pred_min)
            preds.append(pred_norm)

    preds = np.array(preds)
        # Compute the score
#     preds = np.array(preds)
    preds[np.where(preds==0.)] = np.nan
    mae = np.abs(preds[0,:]-preds[1,:])
    score = np.nanmean(mae)/np.nanmean(preds)
    
#     mean_batch = np.mean(preds, axis=1)
#     std_batch = np.std(preds, axis=1)

#     # CV(CV)
#     score = np.std(std_batch/mean_batch) / np.mean(std_batch/mean_batch)
    nparams = model_geom.count_params()

    return score, nparams
##