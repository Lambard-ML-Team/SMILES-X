import logging
import pandas as pd
import numpy as np

from typing import Optional
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score

from SMILESX import utils

# Smooth logging
logger = logging.getLogger()

# Learning curve plotting
def learning_curve(train_loss, val_loss, save_dir: str, data_name: str, ifold: int, run: int, model_type: str) -> None:

    fig = plt.figure(figsize=(6.75, 5), dpi=200)

#     plt.title('')

    ax = fig.add_subplot(111)

    if val_loss is not None:
        ax.set_ylim(0, max(max(train_loss), max(val_loss))+0.005)
    else: 
        ax.set_ylim(0, max(train_loss)+0.005)

    if model_type == 'regression':
        plt.ylabel('Loss (RMSE, scaled)', fontsize=18)
    else:
        plt.ylabel('Loss (cross-entropy)', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    
    ax.plot(train_loss, color='#3783ad')
    if val_loss is not None:
        ax.plot(val_loss, color='#a3cee6')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="x", direction="inout")
    ax.tick_params(axis="y", direction="inout")

    # Ticks decoration
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="x",
                   which="minor",
                   direction="out",
                   top=True,
                   labeltop=True,
                   bottom=True,
                   labelbottom=True)

    ax.tick_params(axis="y",
                   which="minor",
                   direction="out",
                   right=True,
                   labelright=True,
                   left=True,
                   labelleft=True)
    if ifold is not None: 
        ax.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
        plt.savefig('{}/{}_LearningCurve_Fold_{}_Run_{}.png'\
                    .format(save_dir, data_name, ifold, run), bbox_inches='tight')
    else:
        ax.legend(['Train'], loc='upper right', fontsize=14)
        plt.savefig('{}/{}_LearningCurve_Run_{}.png'\
                .format(save_dir, data_name, run), bbox_inches='tight')
    plt.close()
##

# Metric curve plotting for LM
def lm_metric_curve(train_metric, imetrics, imetrics_p, save_dir: str, data_name: str, run: int) -> None:

    fig = plt.figure(figsize=(6.75, 5), dpi=200)

    ax = fig.add_subplot(111)

    ax.set_ylim(min(train_metric)-0.05, max(train_metric)+0.05)

    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    
    ax.plot(train_metric, color='#3783ad')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="x", direction="inout")
    ax.tick_params(axis="y", direction="inout")

    # Ticks decoration
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="x",
                   which="minor",
                   direction="out",
                   top=True,
                   labeltop=True,
                   bottom=True,
                   labelbottom=True)

    ax.tick_params(axis="y",
                   which="minor",
                   direction="out",
                   right=True,
                   labelright=True,
                   left=True,
                   labelleft=True)
    
    ax.legend([imetrics.replace('_accuracy',''), imetrics_p.replace('_accuracy','')], loc='lower right', fontsize=14)
    plt.savefig('{}/{}_MetricCurve_Run_{}.png'\
            .format(save_dir, data_name, run), bbox_inches='tight')
    plt.close()
##

## Compute diverse scores to quantify model's performance on classification tasks
def classification_metrics(y_true, y_pred, model_type, prec, average=None, labels=None):
    """Computes precision, recall, F1, support, ROC AUC and PR AUC scores for
    classification tasks.

    Parameters
    ----------
    y_true: list
        List of true values.
    y_pred: list
        List of predicted values.
    model_type: str
        Type of the model to be used. Can be either 'regression', 'binary_classification', or 'multi_classification'. (Default: 'regression')
    prec: int
        Printing precision. (Default: 4)
    average: str, optional
        Type of averaging performed on the data. If None, no averaging is
        performed. (Default: None)
    labels: list, optional
        List of labels for classification tasks. (Default: None)

    Returns
    -------
    precision: float
        Precision score.
    recall: float
        Recall score.
    f1_score: float
        F1 score.
    support: int
        Support.
    precision_prec: float
        Precision score with printing precision.
    recall_prec: float 
        Recall score with printing precision.
    f1_score_prec: float
        F1 score with printing precision.
    roc_auc: float
        ROC AUC score.
    prc_auc: float
        PR AUC score.
    roc_auc_prec: float
        ROC AUC score with printing precision.
    prc_auc_prec: float
        PR AUC score with printing precision.
    """

    # Extract the predicted class
    if model_type == 'binary_classification':
        y_pred_class = (y_pred > 0.5).astype("int8")
    elif model_type == 'multiclass_classification':
        if len(y_pred.shape) == 2:
            y_pred_class = np.argmax(y_pred, axis=1).astype("int32")
        else:
            y_pred_class = y_pred.astype("int32")

    # classification report
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred_class, average=average, labels=labels)
    # replace nan values with 0 in case (precision + recall) = 0
    f1_score = np.nan_to_num(f1_score)
    precision_prec = output_prec(precision, prec)
    recall_prec = output_prec(recall, prec)
    f1_score_prec = output_prec(f1_score, prec)

    if model_type == 'multiclass_classification': # and average == 'micro':
        roc_auc = None
    else:
        roc_auc = roc_auc_score(y_true, y_pred, average=average, multi_class='ovr', labels=labels) # ovr -> One-vs-rest: AUC of each class against the rest. Sensitive to class imbalance.
    if model_type == 'binary_classification':
        prc_auc = average_precision_score(y_true, y_pred, average=average)
    else:
        prc_auc = None
    roc_auc_prec = output_prec(roc_auc, prec) if roc_auc is not None else None
    prc_auc_prec = output_prec(prc_auc, prec) if prc_auc is not None else None

    return precision, recall, f1_score, support, precision_prec, recall_prec, f1_score_prec, roc_auc, prc_auc, roc_auc_prec, prc_auc_prec
##

def print_stats(trues, preds, errs_pred=None, prec: int = 4, model_type = 'regression', labels=None):
    """Computes, prints and returns RMSE, MAE and R2 for the predictions

    Parameters
    ----------
    trues: list
        List of train, validation and test true values.
    preds: list
        List of train, validation and test predicted values.
    errs_pred: list, optional
        List of train, validation and test errors associated with the
        predictions. (Default: None)
    prec: int
        Printing precision. (Default: 4)
    model_type: str
        Type of the model to be used. Can be either 'regression', 'binary_classification', or 'multi_classification'. (Default: 'regression')
    labels: list, optional
        List of labels for classification tasks. (Default: None)

    Returns
    -------
    Optionally returns the following values:

    For regression tasks:
        rmse_str: str
            Root mean square error (RMSE) together with error obtained via
            error propagation based on the input prediction errors.
        mae_str: float
            Mean absolute error (MAE) together with error obtained via
            error propagation based on the input prediction errors.
        r2_str: float
            R2 correlation score together with error obtained via
            error propagation based on the input prediction errors.
    For classification tasks:
        acc_str: str
            Accuracy score.
        prec_str: str
            Precision score.
        recell_str: str
            Recall score.
        f1_str: str
            F1 score.
        prp_auc_str: str
            Area under the precision-recall curve. More fitted to imbalanced data than ROC AUC.
        conf_mat_arr: 2D numpy array
            Confusion matrix.
    """

    # TODO: switch to list and back to numpy so that python hinting works
    # Reason: MyPy compatibility
    #         For CUDA 10, Tensorflow is 2.3 at max, with number 1.19 at max
    #         Function numpy.npt, which allows for numpy array hinting, is available from numpy 1.20
    set_names = ['test', 'validation', 'train']

    if errs_pred is None:
        errs_pred = [None]*len(preds)

    outputs = []
    for true, pred, err_pred in zip(trues, preds, errs_pred):
        #true, pred = np.array(true).ravel(), np.array(pred).ravel()

        if model_type == 'regression':
            true, pred = np.array(true).ravel(), np.array(pred).ravel()
            rmse = np.sqrt(mean_squared_error(true, pred))
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)

            prec_rmse = output_prec(rmse, prec)
            prec_mae = output_prec(mae, prec)

            if err_pred is None:
                # When used for single run predictions (no standard deviation is available)
                logging.info('Model performance metrics for the ' + set_names.pop() + ' set:')
                logging.info("Averaged RMSE: {0:{1}f}".format(rmse, prec_rmse))
                logging.info("Averaged MAE: {0:{1}f}\n".format(mae, prec_mae))
                logging.info("Averaged R^2: {0:0.4f}".format(r2))

                outputs.append([rmse, mae, r2])
            else:
                err_pred = np.array(err_pred).ravel()
                # When used for fold/total predictions
                d_r2 = sigma_r2(true, pred, err_pred)
                d_rmse = sigma_rmse(true, pred, err_pred)
                d_mae = sigma_mae(err_pred)

                if len(trues)==1:
                    logging.info("Final cross-validation statistics:")
                else:
                    logging.info("Model performance metrics for the " + set_names.pop() + " set:")

                logging.info("Averaged RMSE: {0:{2}f}+-{1:{2}f}".format(rmse, d_rmse, prec_rmse))
                logging.info("Averaged MAE: {0:{2}f}+-{1:{2}f}".format(mae, d_mae, prec_mae))
                logging.info("Averaged R^2: {0:0.4f}+-{1:0.4f}".format(r2, d_r2))
                logging.info("")

                outputs.append([rmse, d_rmse, mae, d_mae, r2, d_r2])
        elif model_type.split('_')[1] == 'classification':
            ## TODO: add classification metrics per class. With label == None, precision, etc are each returned as a list of scores for each class
            
            if err_pred is None:
                logging.info('Model performance metrics for the ' + set_names.pop() + ' set:')
            else:
                if len(trues)==1:
                    logging.info("Final cross-validation statistics:")
                else:
                    logging.info("Model performance metrics for the " + set_names.pop() + " set:")

            for i_average in ['micro', 'macro', 'weighted']: #, None]:
                precision, recall, f1_score, support, \
                precision_prec, recall_prec, f1_score_prec, \
                roc_auc, prc_auc, \
                roc_auc_prec, prc_auc_prec = classification_metrics(true, pred, 
                                                                    model_type, 
                                                                    prec = prec, 
                                                                    average = i_average, 
                                                                    labels = None if i_average is None else labels)
            
                if err_pred is None:
                    logging.info("{0:} avg".format(i_average))
                    logging.info("precision: {0:{1}f}".format(precision, precision_prec))
                    logging.info("recall: {0:{1}f}".format(recall, recall_prec))
                    logging.info("f1-score: {0:{1}f}".format(f1_score, f1_score_prec))
                    logging.info("support: {0:}".format(support))
                    if roc_auc is not None:
                        logging.info("roc_auc: {0:{1}f}".format(roc_auc, roc_auc_prec))
                    else:
                        logging.info("roc_auc: {0:}".format(roc_auc))
                    if prc_auc is not None:
                        logging.info("prc_auc: {0:{1}f}".format(prc_auc, prc_auc_prec))
                    else:
                        logging.info("prc_auc: {0:}".format(prc_auc))
                    logging.info("")

                    if i_average is 'micro':
                        outputs.append([precision, recall, f1_score, roc_auc, prc_auc])
                    else:
                        outputs[-1].append(precision, recall, f1_score, roc_auc, prc_auc)
                else:
                    err_pred = np.array(err_pred).ravel()

                    d_precision, d_recall, d_f1_score, d_support, \
                    _, _, _, \
                    d_roc_auc, d_prc_auc, \
                    _, _ = sigma_classification_metrics(true, pred, err_pred, 
                                                        model_type, prec, i_average, 
                                                        labels = None if i_average is None else labels)
                    
                    logging.info("{0:} avg".format(i_average))
                    logging.info("precision: {0:{2}f}+-{1:{2}f}".format(precision, d_precision, precision_prec))
                    logging.info("recall: {0:{2}f}+-{1:{2}f}".format(recall, d_recall, recall_prec))
                    logging.info("f1-score: {0:{2}f}+-{1:{2}f}".format(f1_score, d_f1_score, f1_score_prec))
                    logging.info("support: {0:}".format(support))
                    if roc_auc is not None:
                        logging.info("roc_auc: {0:{2}f}+-{1:{2}f}".format(roc_auc, d_roc_auc, roc_auc_prec))
                    else:
                        logging.info("roc_auc: {0:}".format(roc_auc))
                    if prc_auc is not None:
                        logging.info("prc_auc: {0:{2}f}+-{1:{2}f}".format(prc_auc, d_prc_auc, prc_auc_prec))
                    else:
                        logging.info("prc_auc: {0:}".format(prc_auc))
                    logging.info("")
                    
                    scores_list = [precision, d_precision, recall, d_recall, f1_score, d_f1_score, roc_auc, d_roc_auc, prc_auc, d_prc_auc]
                    if i_average is 'micro':
                        outputs.append(scores_list)
                    else:
                        for iscore in scores_list:
                            outputs[-1].append(iscore)

    return outputs
##

# Setup the output format for the dataset automatically, based on the precision requested by user
def output_prec(val, prec):
    # Setup the precision of the displayed error to print it cleanly
    if val == 0:
        precision = '0.' + str(prec - 1) # prevent diverging logval if val == 0.
    else:
        logval = np.log10(np.abs(val))
        if logval > 0:
            if logval < prec - 1:
                precision = '1.' + str(int(prec - 1 - np.floor(logval)))
            else:
                precision = '1.0'
        else:
            precision = '0.' + str(int(np.abs(np.floor(logval)) + prec - 1))
    return precision
##

# Plot individual plots per run for the internal tests
def plot_fit(trues, preds, errs_true, errs_pred, err_bars: str, save_dir: str, dname: str, dlabel: str, units: str, fold: Optional[int] = None, run: Optional[int] = None, final: bool = False, model_type='regression') -> None:
    """
    Parameters
    ----------
    true: list
        List of true values.
    preds: list
        List of predicted values.
    errs_true: list array
        List or array of errors associated with the true values ([min, max] array
        or standard deviation list).
    errs_pred: list
        List of errors associated with the predicted values (standard deviations
        computed over augmentations and/or runs).
    err_bars: {'minmax','std'}, optional
        Format for the error bars to be printed (symmetric for standard deviation,
        assymetric for [min, max] range).
    save_dir: str
        Directory to store the plots into.
    dname: str
        Dataset name.
    dlabel: str
        Dataset label used for plot titles.
    units: str
        Data units used for plot titles.
    fold: int, optional
        Cross-validation fold index.
    run: int, optional
        Run index.
    final: bool
        Whether the plot is built for the final out-of-sample predictions.
    model_type: str
        Type of the model to be used. Can be either 'regression', 'binary_classification', or 'multi_classification'. (Default: 'regression')
    """

    set_names = ['Test', 'Validation', 'Train']

    if model_type == 'regression':

        fig = plt.figure(figsize=(6.75, 5), dpi=200)

        ax = fig.add_subplot(111)

        # Setting plot limits
        y_true_min = min([t.min() for t in trues])
        y_true_max = max([t.max() for t in trues])
        y_pred_min = min([p.min() for p in preds])
        y_pred_max = max([p.max() for p in preds])

        # Expanding slightly the canvas around the data points (by 10%)
        axmin = y_true_min-0.1*(y_true_max-y_true_min)
        axmax = y_true_max+0.1*(y_true_max-y_true_min)
        aymin = y_pred_min-0.1*(y_pred_max-y_pred_min)
        aymax = y_pred_max+0.1*(y_pred_max-y_pred_min)

        ax.set_xlim(min(axmin, aymin), max(axmax, aymax))
        ax.set_ylim(min(axmin, aymin), max(axmax, aymax))

        colors = ['#cc1b00', '#db702e', '#519fc4']

        if errs_pred is None:
            errs_pred = [None]*len(preds)

        for true, pred, err_true, err_pred in zip(trues, preds, errs_true, errs_pred):
            # Put the shapes of the errors to the format accepted by matplotlib
            # (N, ) for symmetric errors, (2, N) for asymmetric errors
            if err_bars is not None:
                err_true = error_format(true, err_true, err_bars)

            # Legend printing for train/val/test
            if final:
                # No legend is needed for the final out-of-sample prediction
                set_name = None
            else:
                set_name = set_names.pop()

            ax.errorbar(true.ravel(),
                        pred.ravel(),
                        xerr = err_true,
                        yerr = err_pred,
                        fmt='o',
                        label=set_name,
                        ecolor='#bababa',
                        elinewidth = 0.5,
                        ms=5,
                        mfc=colors.pop(),
                        markeredgewidth = 0,
                        alpha=0.7)

        # Define file name
        if final:
            file_name = '{}/Figures/Pred_vs_True/{}_PredvsTrue_Plot_Final.png'.format(save_dir, dname)
        elif run is None:
            file_name = '{}/Figures/Pred_vs_True/Folds/{}_PredvsTrue_Plot_Fold_{}.png'.format(save_dir, dname, fold)
        else:
            file_name = '{}/Figures/Pred_vs_True/Runs/{}_PredvsTrue_Plot_Fold_{}_Run_{}.png'.format(save_dir, dname, fold, run)

        # Plot X=Y line
        ax.plot([max(plt.xlim()[0], plt.ylim()[0]),
                min(plt.xlim()[1], plt.ylim()[1])],
                [max(plt.xlim()[0], plt.ylim()[0]),
                min(plt.xlim()[1], plt.ylim()[1])],
                ':', color = '#595f69')

        if len(units) != 0:
            units = ' (' + units + ')'
        if len(dlabel) != 0:
            plt.xlabel(r"{}, experimental {}".format(dlabel, units), fontsize = 18)
            plt.ylabel(r"{}, prediction {}".format(dlabel, units), fontsize = 18)
    #         plt.xlabel('{}, ground truth {}'.format(dlabel, units), fontsize = 18)
    #         plt.ylabel('{}, prediction {}'.format(dlabel, units), fontsize = 18)
        else:
            plt.xlabel('Ground truth {}'.format(units), fontsize = 18)
            plt.ylabel('Prediction {}'.format(units), fontsize = 18)
        if not final:
            ax.legend(fontsize=14)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="x", direction="inout")
        ax.tick_params(axis="y", direction="inout")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="x", which="minor", direction="out",
            top=True, labeltop=True, bottom=True, labelbottom=True)
        ax.tick_params(axis="y", which="minor", direction="out",
            right=True, labelright=True, left=True, labelleft=True)

        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    elif model_type.split('_')[1] == 'classification':
        
        #TODO(Guillaume): Account for errors when plotting
        for true, pred, err_pred in zip(trues, preds, errs_pred):
            if model_type == 'regression':
                true, pred = np.array(true).ravel(), np.array(pred).ravel()

            # Extract the predicted class
            if model_type == 'binary_classification':
                pred_class = (pred > 0.5).astype("int8")
            elif model_type == 'multiclass_classification':
                if len(pred.shape) == 2:
                    pred_class = np.argmax(pred, axis=1).astype("int32")
                else:
                    pred_class = pred.astype("int32")

            # Legend printing for train/val/test
            if final:
                # No legend is needed for the final out-of-sample prediction
                set_name = 'final'
            else:
                set_name = set_names.pop()

            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

            conf_mat = confusion_matrix(true, pred_class)
            cm = ax.matshow(conf_mat)
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center', size='xx-large')
            plt.title(set_name+' set confusion matrix')
            fig.colorbar(cm, ax=ax)
            plt.ylabel('Observed label', fontsize = 18)
            plt.xlabel('Predicted label', fontsize = 18)

            # Define file name
            if final:
                file_name = '{}/Figures/Pred_vs_True/{}_PredvsTrue_{}_ConfMatrix_Final.png'.format(save_dir, set_name, dname)
            elif run is None:
                file_name = '{}/Figures/Pred_vs_True/Folds/{}_PredvsTrue_{}_ConfMatrix_Fold_{}.png'.format(save_dir, set_name, dname, fold)
            else:
                file_name = '{}/Figures/Pred_vs_True/Runs/{}_PredvsTrue_{}_ConfMatrix_Fold_{}_Run_{}.png'.format(save_dir, set_name, dname, fold, run)

            plt.savefig(file_name, bbox_inches='tight')
            plt.close()

            if model_type == 'binary_classification':
                # Precision-recall curve plot 
                # plot the precision-recall curves
                prp_precision, prp_recall, prp_th = precision_recall_curve(true, pred)
                # calculate f score
                fscore = (2 * prp_precision * prp_recall) / (prp_precision + prp_recall)
                # replace nan values with 0 in case (prp_precision + prp_recall) = 0
                fscore[np.isnan(fscore)] = 0
                # locate the index of the largest f score
                best_fscore_idx = np.argmax(fscore)
                logging.info("{} set:".format(set_name))
                logging.info("Precision-Recall curve best threshold = {0:0.4f}, F1 = {1:0.4f}\n".format(prp_th[best_fscore_idx], fscore[best_fscore_idx]))
                
                random_perf = len(true[true==1]) / len(true)
                plt.plot([0, 1], [random_perf, random_perf], linestyle='--', label='Random')
                plt.plot(prp_recall, prp_precision, marker='.', label='SMILES-X')
                plt.scatter(prp_recall[best_fscore_idx], prp_precision[best_fscore_idx], marker='o', color='black', label='Best F1 score')
                plt.xlabel('Recall', fontsize = 18)
                plt.ylabel('Precision', fontsize = 18)
                plt.legend()

                # Define file name
                if final:
                    file_name = '{}/Figures/Pred_vs_True/{}_PredvsTrue_{}_PrecisionRecall_curve_Final.png'.format(save_dir, set_name, dname)
                elif run is None:
                    file_name = '{}/Figures/Pred_vs_True/Folds/{}_PredvsTrue_{}_PrecisionRecall_curve_Fold_{}.png'.format(save_dir, set_name, dname, fold)
                else:
                    file_name = '{}/Figures/Pred_vs_True/Runs/{}_PredvsTrue_{}_PrecisionRecall_curve_Fold_{}_Run_{}.png'.format(save_dir, set_name, dname, fold, run)

                plt.savefig(file_name, bbox_inches='tight')
                plt.close()
##

def error_format(val, err, bars):
    # If any error is given
    if err is not None:
        # If one error value is given, it is treated as standard deviation
        if err.shape[1]==1:
            return err.ravel()
        # If two error values are given, they are treated as [min, max]
        elif err.shape[1]==2:
            # Switch from min/max range to the lengths of error bars
            # to the left/right from the mean or median value
            val = val.reshape(-1,1)
            return np.abs(err-val).T
        # If three error values are given, they are treated as [std, min, max]
        elif err.shape[1]==3:
            if bars == 'minmax':
                # Switch from min/max range to the lengths of error bars
                # to the left/right from the mean or median value
                return np.abs(val-err[:,1:]).T
            elif bars == 'std':
                return err[:,0].ravel()
            else:
                logging.warning("ERROR:")
                logging.warning("Error bars format is not understood.")
                logging.warning("")
                logging.warning("SMILES-X execution is aborted.")
                raise utils.StopExecution
    else:
        return err
##

# Compute the error on the estimated R2-score based on the prediction error
# Checked!
def sigma_r2(true, pred, err_pred):
    sstot = np.sum(np.square(true - np.mean(true)))
    sigma_r2 = 2/sstot*np.sqrt(np.square(true-pred).T.dot(np.square(err_pred)))
    return float(sigma_r2)
##

# Compute the error on the estimated RMSE based on the prediction error
# Checked!
def sigma_rmse(true, pred, err_pred):
    N = float(len(err_pred))
    ssres = np.sum(np.square(true - pred))
    sigma_rmse = np.sqrt(np.square(true-pred).T.dot(np.square(err_pred))/N/ssres)
    return float(sigma_rmse)
##

# Compute the error on the estimated MAE based on the prediction error
# Checked!
def sigma_mae(err_pred):
    N = float(len(err_pred))
    sigma_mae = np.sqrt(np.sum(np.square(err_pred))) / N
    return float(sigma_mae)
##

# Compute the error on the estimated classification scores based on the prediction error
# via a Monte-Carlo simulation
def sigma_classification_metrics(true, pred, err_pred, model_type, prec, average, labels, n_mc=1000):
    N = float(len(err_pred))
    sigma = np.zeros((n_mc, 11))
    for i in range(n_mc):
        pred_mc = pred + np.random.normal(0, err_pred).reshape(pred.shape)
        sigma[i,:] = classification_metrics(true, pred_mc, model_type, prec, average, labels)
    sigma = np.std(sigma, axis=0)
    return sigma.ravel()
## 
