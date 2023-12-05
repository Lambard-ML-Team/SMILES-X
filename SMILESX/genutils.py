__version__ = '2.1'
__author__ = 'Guillaume Lambard, Ekaterina Gracheva'

"""Functions used during the generation phase.
"""

from scipy import stats
import numpy as np

np.random.seed(seed=21)

def p_to_one(dist):
    # Convert a probability distribution to a one-hot vector
    # dist is a 2D array where each row is a probability distribution
    # Returns a 2D array where each row is a one-hot vector
    if len(dist.shape) == 1:
        dist = dist.reshape(1,-1)
    return dist/np.sum(dist, axis=1).reshape(-1,1)

def normalize(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    # Equivalent to p_to_one without temperature
    preds = preds.astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds, axis=1).reshape(-1,1)
    return preds

def softmax(x, axis=-1):
    # Compute softmax values for each sets of scores in x
    e_x = np.exp(x - np.max(x, axis=axis).reshape(-1,1))
    return e_x / np.sum(e_x, axis=axis).reshape(-1,1)

def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    preds = normalize(preds, temperature)
    probas = np.zeros(preds.shape)
    for ipred in range(preds.shape[0]):
        probas[ipred] = np.random.multinomial(1, preds[ipred], 1)[0]
    return probas.astype('bool')

def remove_from_list(list_tmp, to_remove = ''): # list_tmp = list(list())
    # Helper function to remove a token from a list of tokens
    return [list(filter(lambda t: t != to_remove, ilist)) for ilist in list_tmp]

def remove_schar(list_tmp):
    # Helper function to remove 'pad', '!', 'E' characters
    list_tmp = remove_from_list(list_tmp, 'pad')
    list_tmp = remove_from_list(list_tmp, '!')
    list_tmp = remove_from_list(list_tmp, 'E')
    return list_tmp

def join_tokens(list_tmp): 
    # Helper function to join tokens
    list_tmp = [''.join(ismiles) for ismiles in list_tmp]
    return list_tmp

def t_cdf(x):
    # Helper function to compute the t-cdf
    return stats.t.cdf(x, 10000)

##