"""Add main docstring discription

"""

import re
import ast

import numpy as np

from tensorflow.keras import backend as K

def int_vec_encode(tokenized_smiles_list, max_length, vocab):
    """Encodes SMILES as a vector of integers

    Parameters
    ----------
    tokenized_smiles_list: list(list(str))
        List of tokenized SMILES, where every list(str) corresponds to a single SMILES
    max_length: int
        Maximum SMILES length
    vocab:
        Vocabulary, or a list of all possible tokens contained within the data

    Returns
    -------
    int_smiles_array: np.array
        Numpy array of encoded SMILES of shape (len(tokenized_smiles_list), max_length)
    """

    token_to_int = get_tokentoint(vocab)
    int_smiles_array = np.zeros((len(tokenized_smiles_list), max_length), dtype=np.int32)
    for csmiles,ismiles in enumerate(tokenized_smiles_list):
        ismiles_tmp = list()
        if len(ismiles)<= max_length:
            ismiles_tmp = ['pad']*(max_length-len(ismiles))+ismiles # Force output vectors to have same length
        else:
            ismiles_tmp = ismiles[-max_length:] # longer vectors are truncated (to be changed...)
        integer_encoded = [token_to_int[itoken] if(itoken in vocab) \
                           else token_to_int['unk']\
                           for itoken in ismiles_tmp]
        int_smiles_array[csmiles] = integer_encoded

    return int_smiles_array
##

def get_tokens(smiles_list, split_l = 1):
    """Gets tokens from a list of tokens from SMILES

    Parameters
    ----------
    smiles_list: list
        List of of SMILES
    split_l: int
        Number of tokens contained within a split (default: 1)
        Result examples for different `split_l` values:
        split_l = 1 -> np.array(['CC=O']) => [[' ', 'C', 'C', '=', 'O', ' ']],
        split_l = 2 -> np.array(['CC=O']) => [[' C', 'CC', 'C=', '=O', 'O ']],
        split_l = 3 -> np.array(['CC=O']) => [[' CC', 'CC=', 'C=O', '=O ']],

    Returns
    -------
    tokenized_smiles_list: list(str)
        List of tokenized SMILES
    """

    tokenized_smiles_list = list()
    for ismiles in smiles_list:
        if not isinstance(ismiles, str):
            ismiles = ismiles[0]
        # TODO: Kathya
        # Make tokenisation from multiple SMILES strings for paper on copolymers
        # (Modify the SMILES-X to take multiple inputs)
        tokenized_smiles_tmp = smiles_tokenizer(ismiles)
        tokenized_smiles_list.append([''.join(tokenized_smiles_tmp[i:i+split_l])
                                  for i in range(0,len(tokenized_smiles_tmp)-split_l+1,1)
                                 ])
    return tokenized_smiles_list
##

def smiles_tokenizer(smiles):
    """Tokenize SMILES

    Splits molecules into tokens, which represent:
    aliphatic organic compounds, aromatic organic compounds,
    isotopes, chirality, hydrogen count, charge, class (with respective squared brackets)
    bonds, rings, wildcards and branches

    Parameters
    ----------
    smiles: str
        Input SMILES string to tokenize

    Returns
    -------
    tokenized_smiles_list: list(str)
        List of tokens extended with a termination character ' '
    """

    patterns = "(\*|" +\
               "N|O|S|P|F|Cl?|Br?|I|" +\
               "b|c|n|o|s|p|j|" +\
               "\[.*?\]|" +\
               "-|=|#|\$|:|/|\\|\.|" +\
               "[0-9]|\%[0-9]{2}|" +\
               "\(|\))"
    regex = re.compile(patterns)
    tokens = [token for token in regex.findall(smiles)]

    return [' '] + tokens + [' ']
##

def extract_vocab(lltokens):
    """Vocabulary extraction

    Parameters
    ----------
    lltokens: list
        list of lists of tokens (list of tokenized SMILES)

    Returns
    -------
        Dictionary containing all the individual tokens
    """

    return set([itoken for ismiles in lltokens for itoken in ismiles])
##

def get_tokentoint(tokens):
    """Translates string tokens into integers

    Parameters
    ----------
    tokens: list
        List of tokens

    Returns
    -------
        Dictionary with tokens as keys and corresponding integers as values
    """

    return dict((c, i) for i, c in enumerate(tokens))
##

def get_inttotoken(tokens):
    """Translates string tokens into integers

    Parameters
    ----------
    tokens: list
        List of tokens

    Returns
    -------
        Dictionary with ingeres as keys and corresponding tokens as values
    """

    return dict((i, c) for i, c in enumerate(tokens))
##

def save_vocab(vocab, tokens_file):
    """Stores vocabulary for further use of trained models

    Parameters
    ----------
    vocab: list
        List of tokens formin vocabulary
    tokens_file: str
        Name of the file to store the vocabulary (*.txt)
    """

    with open(tokens_file,'w') as f:
        f.write(str(list(vocab)))
##

def get_vocab(tokens_file):
    """Retrieves previously saved vocabulary

    Parameters
    ----------
    tokens_file
        Text file name with directory in which the vocabulary is saved (*.txt)

    Returns
    -------
    tokens:
        Set of individual tokens forming a vocabulary
    """
    
    with open(tokens_file,'r') as f:
        tokens = ast.literal_eval(f.read())
    return tokens
##