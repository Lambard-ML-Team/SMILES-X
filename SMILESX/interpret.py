import numpy as np
import pandas as pd
import os
import math
import glob

import logging
from tabulate import tabulate

from rdkit import Chem
from rdkit.Chem import Draw

from typing import Optional
from typing import List

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from SMILESX import utils, token, augm, visutils, inference

def interpret(model, smiles, true=None, true_err=None, pred=None, log_verbose: bool = True, check_smiles: bool = True, smiles_concat: bool = False, font_size: int = 15, font_rotation: str = 'horizontal'):
    """Inference based on ensemble of trained SMILESX models

    Prediction of the property based on the ensemble of SMILESX models.
    Mean and standard deviation are computed over multiple models' predictions.

    Parameters
    ----------
    model: list
        The list of models to be used for inference
    smiles: list(str)
        The list of SMILES to be characterized
    log_verbose: bool
        Whether to print the output to the consol (Default: True)
    check_smiles: bool
        Whether to check the SMILES via RDKit (Default: True)
    font_size: int
        Size of the font SMILES tokens. (Default: 15)
    font_rotation: {'horizontal', 'vertical'}
        font's size for writing SMILES tokens (Default: 15)

    Returns
    -------
    pd.DataFrame
        Dataframe of SMILES with their inferred property (SMILES, mean, standard deviation)
    """

    save_dir =  "{}/{}/{}/Interpret".format(model.outdir,
                                            model.data_name,
                                            'Augm' if model.augment else 'Can')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger, logfile = utils.log_setup(save_dir, 'Interpret', log_verbose)

    logging.info("******************************************")
    logging.info("***   SMILESX INTERPRETATION STARTED   ***")
    logging.info("******************************************")
    logging.info("")

    logging.info("Interpretation logs path:")
    logging.info(logfile)
    logging.info("")
    
    smiles = smiles.values
    if pred is None:
        logging.info("Predictions are not provided.")
        logging.info("Only ground truth values will be displayed on the attention maps.")
        logging.info("Predictions can be obtained via `inference` funcionality.")
        logging.info("")
        print_pred = False
    else:
        pred = pred.values
        print_pred = True
    if true is None:
        logging.info("Ground truth property values are not provided.")
        logging.info("They will not be displayed on the attention maps.")
        logging.info("")
        print_true = False
    else:
        true = true.values
        print_true = True
    if true_err is None:
        print_err = False
    else:
        # Setting up errors printing format
        true_err = true_err.values
        if true_err.shape[1] is None:
            true_err = true_err.reshape(-1,1)
        if true_err.shape[1] == 1:
            err_format = 'std'
        elif true_err.shape[1] == 2:
            err_format = 'minmax'
            true_err = visutils.error_format(true.reshape(-1,1), true_err, 'minmax').T
            print(true_err)
            print(true)
        print_err = True

    logging.info("Full vocabulary: {}".format(model.tokens))
    logging.info("Vocabulary size: {}".format(len(model.tokens)))
    logging.info("Maximum length of tokenized SMILES: {} tokens.\n".format(model.max_length))
    
    # For the moment, no augmentation is implemented
    smiles_enum, _, _, _ = augm.augmentation(data_smiles=smiles,
                                             data_extra=None,
                                             data_prop=None,
                                             check_smiles=check_smiles,
                                             augment=False)

    # Concatenate multiple SMILES into one via 'j' joint
    if smiles_concat:
        smiles_enum = utils.smiles_concat(smiles_enum)

    # Tokenize SMILES
    smiles_enum_tokens = token.get_tokens(smiles_enum)
    # Encode the tokens into integers
    smiles_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = smiles_enum_tokens, 
                                                       max_length = model.max_length, 
                                                       vocab = model.tokens)
    # Collect attention vectors from trained models
    att_map = np.empty((model.k_fold_number * model.n_runs, len(smiles_enum_tokens), model.max_length), dtype='float')
    for ifold in range(model.k_fold_number):
        for run in range(model.n_runs):
            imodel = model.att_dic["Fold_{}".format(ifold)][run]
            iatt = imodel.predict({"smiles": smiles_enum_tokens_tointvec})
            smiles_att = np.squeeze(iatt, axis=2).reshape(1, len(smiles_enum_tokens), model.max_length)
            att_map[ifold*model.n_runs + run, :, :] = smiles_att
    # Mean and standard deviation on attention maps from models ensembling
    att_map_mean = np.mean(att_map, axis = 0)
    att_map_std = np.std(att_map, axis = 0)
    
    print(smiles_enum_tokens)
    for i, ismiles in enumerate(smiles_enum_tokens):
        
        logging.info("*******")
        logging.info("SMILES: {}".format(smiles[i]))

        ismiles = ismiles[1:-1] # Remove padding
        ismiles_len = len(ismiles)
        iatt_map_mean = att_map_mean[i,-ismiles_len-1:-1]

        # 1D attention map
        plt.matshow([iatt_map_mean], cmap='Reds') # SMILES padding excluded
        plt.tick_params(axis='x', bottom = False)
        plt.xticks(np.arange(ismiles_len), ismiles)#, fontsize=font_size, rotation=font_rotation)
        plt.yticks([])
        plt.savefig(save_dir + '/1D_Interpretation_' + str(i) + '.png', bbox_inches='tight')
        plt.show()
        
        if print_pred:
            prec = visutils.output_prec(pred[i, 0], 3)
        if print_true:
            true_val = "{0}".format(true[i])
            if print_err:
                if err_format == 'minmax':
                    true_val += "$_{{-{0}}}^{{+{1}}}$".format(str(true_err[i, 0]),
                                                              str(true_err[i, 1]))
                elif err_format == 'std':
                    true_val += "$\pm$ {1:0.02f}".format(true_err[i])
        else:
            true_val = None
        if print_pred:
            pred_val="{1:{0}f} $\pm$ {2:{0}f}".format(prec,
                                                      pred[i, 0],
                                                      pred[i, 1])
        else:
            pred_val = None
        
        # 2D attention map
        mol_tmp = Chem.MolFromSmiles(smiles[i])
        mol_df_tmp = pd.DataFrame([ismiles, att_map_mean[i].\
                                   flatten().tolist()[-ismiles_len-1:-1]]).transpose()
        bond = ['-','=','#','$','/','\\','.','(',')']
        mol_df_tmp = mol_df_tmp[~mol_df_tmp.iloc[:,0].isin(bond)]
        mol_df_tmp = mol_df_tmp[[not itoken.isdigit() for itoken in mol_df_tmp.iloc[:,0].values.tolist()]]

        minmaxscaler = MinMaxScaler(feature_range=(0,1))
        norm_weights = minmaxscaler.fit_transform(mol_df_tmp.iloc[:,1].values.reshape(-1,1)).flatten().tolist()

        fig_tmp = get_similarity_map_from_weights(mol=mol_tmp,
                                                  pred_val=pred_val,
                                                  true_val=true_val,
                                                  sigma=0.05,
                                                  weights=norm_weights, 
                                                  colorMap='Reds', 
                                                  alpha = 0.25)
        
        plt.savefig(save_dir+'/2D_Interpretation_'+str(i)+'.png', bbox_inches='tight')
        plt.show()
        
        # Temporal relative distance plot
        # Cannot be built in case where the model is trained with additional numerical data
        # Observation based on non-augmented SMILES because of SMILES sequential elongation
        if model.extra is False:
            plt.figure(figsize=(15,7))
            diff_topred_list = list()
            subsmiles_list = []
            fragment = ""
            for j in ismiles:
                fragment += j
                subsmiles_list.append(fragment)
            # Predict property for subsmiles
            ipreds = inference.infer(model=model,
                                     data_smiles=subsmiles_list,
                                     data_extra=None,
                                     augment=False,
                                     check_smiles=False,
                                     log_verbose=False)
            ipreds_mean = ipreds['mean'].values
            relative_diff = (ipreds_mean-ipreds_mean[-1])/np.abs(ipreds_mean[-1])
            max_relative_diff = np.max(relative_diff)

            markers, stemlines, baseline = plt.stem([ix for ix in range(ismiles_len)],
                                                    relative_diff,
                                                    'k.-',
                                                    use_line_collection=True)

            plt.setp(baseline, color='k', linewidth=2, linestyle='--')
            plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black')
            plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
            plt.xticks(range(ismiles_len),
                       ismiles,
                       fontsize = font_size,
                       rotation = font_rotation)
            plt.yticks(fontsize = 20)
            plt.ylabel('Cumulative SMILES path', fontsize = 20, labelpad = 15)
            plt.ylabel('Temporal relative distance', fontsize = 20, labelpad = 15)
            plt.savefig(save_dir+'Temporal_Relative_Distance_smiles_'+str(i)+'.png', bbox_inches='tight')
            plt.show()
    
    logging.info("********************************************")
    logging.info("***   SMILESX INTERPRETATION COMPLETED   ***")
    logging.info("********************************************")
##

## Attention weights depiction
# from https://github.com/rdkit/rdkit/blob/24f1737839c9302489cadc473d8d9196ad9187b4/rdkit/Chem/Draw/SimilarityMaps.py
def get_similarity_map_from_weights(mol, weights, pred_val, true_val = False, colorMap=None, size=(250, 250), sigma=0.05, coordScale=1.5, step=0.01, colors='k', contourLines=10, alpha=0.5, **kwargs):
    """Generates the similarity map for a molecule given the atomic weights.

    Attention weights depiction from
    https://github.com/rdkit/rdkit/blob/24f1737839c9302489cadc473d8d9196ad9187b4/rdkit/Chem/Draw/SimilarityMaps.py.

    Parameters
    ----------
    mol: RDKit.mol
        The molecule of interest.
    weights:
        Attention weights representing the importance of each token.
    pred_val: str
        Predicted property value to be displayed.
    true_val: str, optional
        Ground truth property value to be displayed.
    colorMap:
        The matplotlib color map scheme, default is custom PiWG color map.
    size: (int, int)
        The size of the figure.
    sigma: float
        The sigma for the Gaussians.
    coordScale: float
        Scaling factor for the coordinates.
    step: float
        The step for calcAtomGaussian.
    colors: str
        Color of the contour lines (Default: 'k').
    contourLines: int, list(int)
        If integer, the number of contour lines to be drawn.
        If list, the positions of contour lines to be drawn.
    alpha: float
        The alpha channel (transparancy degree) for the contour lines.
    kwargs:
        Additional arguments for drawing.

    Returns
    -------
    figure
        Figure with similarity map for a molecule given the attention weights.
    """

    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
    ax = fig.gca()
    
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    # scaling
    maxScale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
    minScale = min(math.fabs(np.min(z)), math.fabs(np.max(z)))
    
    fig.axes[0].imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower',
                     extent=(0, 1, 0, 1), vmin=minScale, vmax=maxScale)
    ax.imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower',
                     extent=(0, 1, 0, 1), vmin=minScale, vmax=maxScale)
    # Contour lines
    # Only draw them when at least one weight is not zero
    if len([w for w in weights if w != 0.0]):
        contourset = ax.contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
        for j, c in enumerate(contourset.collections):
            if contourset.levels[j] == 0.0:
                c.set_linewidth(0.0)
            elif contourset.levels[j] < 0:
                c.set_dashes([(0, (3.0, 3.0))])
    if true_val:
        ax.text(0.97, 0.02, r"Experimental: "+true_val+"\nPredicted: "+pred_val,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=16)
    else:
        ax.text(0.97, 0.02, r"Predicted: "+pred_val,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=16)
    ax.set_axis_off()
    return fig
##