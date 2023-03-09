"""Add main docstring discription

"""

import logging
import itertools

import numpy as np
import pandas as pd

from rdkit import Chem

from SMILESX import utils

def augmentation(data_smiles, data_extra=None, data_prop=None, check_smiles=True, augment=False):
    """Augmentation

    Parameters
    ----------
    data_smiles:
        SMILES array for augmentation
    data_extra:
        Corresponding extra data array for augmentation
    data_prop:
        Corresponding property array for augmentation (default: None)
    check_smiles: bool
        Whether to verify SMILES correctness via RDKit (default: True)
    augment: bool
        Whether to augment the data by atom rotation (default: False)

    Returns
    -------
    smiles_enum
        Array of augmented SMILES
    extra_enum
        Array of related additional inputs
    prop_enum
        Array of related property inputs
    miles_enum_card    
        Number of augmentation per SMILES
    """

    # Get the logger
    logger = logging.getLogger()
    
    if augment and not check_smiles:
        logging.error("ERROR:")
        logging.error("Augmentation is requested, but SMILES checking via RDKit is set to False.")
        logging.error("Augmentation cannot be performed on potentially invalid SMILES.")
        logging.error("")
        logging.error("*** AUGMENTATION ABORTED ***")
        raise utils.StopExecution

    smiles_enum = []
    prop_enum = []
    extra_enum = []
    smiles_enum_card = []
    rejected_smiles = []
    
    for csmiles, ismiles in enumerate(data_smiles.tolist()):
        if augment:
            enumerated_smiles = generate_smiles(ismiles, rotate=True)
        else:
            if check_smiles:
                enumerated_smiles = generate_smiles(ismiles, rotate=False)
            else:
                enumerated_smiles = [ismiles]
        if None not in enumerated_smiles:
            # Store indices where same index corresponds to the same orginal SMILES
            smiles_enum_card.extend([csmiles] * len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            if data_prop is not None:
                prop_enum.extend([data_prop[csmiles]] * len(enumerated_smiles))
            if data_extra is not None:
                extra_enum.extend([data_extra[csmiles]] * len(enumerated_smiles))
        else:
            rejected_smiles.extend(ismiles)
    if len(smiles_enum) == 0:
        logging.error("None of the provided SMILES is recognized as correct by RDKit.")
        logging.error("In case the SMILES data cannot be put to a correct format, set `check_smiles=False`.")
        logging.error("*Note: setting `check_smiles` to False disables augmentation.")
        logging.error("")
        logging.error("*** Process of inference is automatically aborted! ***")
        raise utils.StopExecution
    if len(rejected_smiles) > 0:
        logging.error("Some of the provided SMILES are recognized as incorrect by RDKit.")
        logging.error("The following list of SMILES have been rejected:")
        logging.error(rejected_smiles)

    if data_extra is None:
        extra_enum = None
    else:
        extra_enum = np.array(extra_enum)
    if data_prop is None:
        prop_enum = None
    else:
        prop_enum = np.array(prop_enum)
    return smiles_enum, extra_enum, prop_enum, smiles_enum_card
##

def rotate_atoms(li, x):
    """Rotate atoms' index in a list.

    Parameters
    ----------
    li: list
        List to be rotated.
    x: int
        Index to be placed first in the list.

    Returns
    -------
        A list of rotated atoms.
    """

    return (li[x%len(li):]+li[:x%len(li)])
##

def generate_smiles(smiles, kekule = False, rotate = False):
    """Generate SMILES list

    Parameters
    ----------
    smiles: list(str)
        SMILES list to be prepared.
    kekule: bool
        Kekulize option setup. (Default: False)
    canon: bool
        Canonicalize. (Default: True)
    rotate: bool
        Rotation of atoms's index for augmentation. (Default: False)

    Returns
    -------
    smiles_augment: list
        A list of augmented SMILES (non-canonical equivalents from canonical SMILES representation).
    """
    
    output_augm = []
    augms = []
    # Single SMILES per input
    if isinstance(smiles, str):
        smiles = [smiles]
    # Multiple SMILES per input (e.g., copolymers, additives, etc.)
    for ismiles in smiles:
        if ismiles != '':
            mols = [] # only for this smiles augmenting and returning a list of augms
            ismiles_augm = []
            # Get augmentations per SMILES
            try:
                mol = Chem.MolFromSmiles(ismiles)
                mols.append(mol)
                n_atoms = mol.GetNumAtoms()
                n_atoms_list = [nat for nat in range(n_atoms)]
                if rotate:
                    canon = False
                    if n_atoms != 0:
                        for iatoms in range(n_atoms):
                            n_atoms_list_tmp = rotate_atoms(n_atoms_list, iatoms)
                            rot_mol = Chem.RenumberAtoms(mol, n_atoms_list_tmp)
                            mols.append(rot_mol)
                else:
                    canon = True
            except:
                mol = None

            for mol in mols:
                try:
                    ismiles = Chem.MolToSmiles(mol,
                                               isomericSmiles=True,
                                               kekuleSmiles=kekule,
                                               rootedAtAtom=-1,
                                               canonical=canon,
                                               allBondsExplicit=False,
                                               allHsExplicit=False)
                except:
                    ismiles=None
                ismiles_augm.append(ismiles)
            # Remove duplicates
            ismiles_augm = list(dict.fromkeys(ismiles_augm).keys())
            # Store augmentations for each of multiple SMILES provided for a single data point
            augms.append(ismiles_augm)
        else:
            augms.append([''])
    # All the possible combinations of multiple augmented SMILES
    output_augm = [list(au) for au in itertools.product(*augms)]
    return output_augm
##