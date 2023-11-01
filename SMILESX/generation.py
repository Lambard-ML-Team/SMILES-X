__version__ = '2.1'
__author__ = 'Guillaume Lambard, Ekaterina Gracheva'

"""Add main docstring description
TODO(Guillaume): update the description
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

import numpy as np
import os
import glob

from SMILESX import model, inference, token, utils, trainutils, loadmodel, genutils

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

from rdkit import Chem

from scipy.special import softmax
from scipy import stats

import matplotlib.pyplot as plt
from IPython.display import clear_output

class Generation(object):
    '''
    Class for SMILES generation directed by SMILES-X targeted properties inference

    Attributes
    ----------
    data_name : str
        dataset's name
    g_run_number : int
        number of the run to be considered for the generation model (Default: 0)
    gen_augmentation : bool
        SMILES's augmentation at generator training (Default: False)
    infer_augmentation : bool   
        SMILES's augmentation at predictor training (Default: False)
    infer_use_n_folds : int
        number of folds to be considered for the predictor model. 
        If put to None, all available folds are used for inference (Default: None)
    infer_use_n_runs : int
        number of runs to be considered for the predictor model. 
        If put to None, all available runs are used for inference (Default: None)
    indir : str 
        directory of already trained prediction models (*.hdf5) and vocabulary (*.txt) (Default: './outputs')
    outdir : str
        directory for outputs (plots + .txt files) -> 'Inference/'+'{}/{}/'.format(data_name,g_dir_temp) is then created (Default: './outputs')
    n_gpus : int
        number of GPUs to use (Default: 1)
    gpus_list : list
        list of GPUs to use (Default: None)
    gpus_debug : bool
        debug mode for GPUs (Default: False)
    prop_names_list : list  
        list of properties' names (Default: None)

    Methods
    -------
    __init__(self,
             data_name,
             g_run_number = 0,   
             gen_augmentation = False,
             infer_augmentation = False,
             infer_use_n_folds = None,   
             infer_use_n_runs = None,
             indir = "./output/",
             outdir = "./output/",
             vocab_name = 'vocab',
             n_gpus = 1,
             gpus_list = None,
             gpus_debug = False,
             prop_names_list = None)
    Initialize the class Generation with the given parameters (see Attributes). 
    '''

    def __init__(self, 
                 data_name, 
                 g_run_number = 0, 
                 gen_augmentation = False, 
                 infer_augmentation = False, 
                 infer_use_n_folds = None,
                 infer_use_n_runs = None,
                 indir = "./outputs", 
                 outdir = "./outputs", 
                 n_gpus = 1, 
                 gpus_list = None, 
                 gpus_debug = False,
                 prop_names_list = None):
        
        # GPUs options
        self.strategy, self.gpus = utils.set_gpuoptions(n_gpus = n_gpus, 
                                                        gpus_list = gpus_list, 
                                                        gpus_debug = gpus_debug)
        if self.strategy is None:
            raise utils.StopExecution
        ##
        
        self.data_name = data_name
        self.g_run_number = g_run_number
        self.gen_augmentation = gen_augmentation
        self.infer_augmentation = infer_augmentation
        self.infer_use_n_folds = infer_use_n_folds
        self.infer_use_n_runs = infer_use_n_runs
        self.indir = indir
        self.outdir = outdir
        self.prop_names_list = prop_names_list
        
        infer_model_dir_list = []
        if self.prop_names_list is not None:
            for iprop_name in self.prop_names_list:
                infer_model_dir_tmp = '{}/{}/{}/Train/Models'.format(indir, iprop_name, 'Augm' if infer_augmentation else 'Can')
                infer_model_dir_list.append(infer_model_dir_tmp)
        
        gen_input_dir = '{}/{}/{}/{}/Train'.format(indir, data_name, 'LM', 'Augm' if gen_augmentation else 'Can')
        gen_model_dir = gen_input_dir + '/Models'

        vocab_file = '{}/Other/{}_Vocabulary.txt'.format(gen_input_dir, data_name)
        if not os.path.exists(vocab_file):
            print("***Process of generation automatically aborted!***")
            print("The {} file does not exist.\n".format(vocab_file))
            print("Please, train the generator first.\n")
            return

        infer_input_dir = '{}/{}/{}/Train'.format(indir, data_name, 'Augm' if infer_augmentation else 'Can')
        
        if not os.path.exists(gen_model_dir):
            print("***Process of generation automatically aborted!***")
            print("The {} directory does not exist.\n".format(gen_model_dir))
            print("Please, train the generator first.\n")
            return

        gen_models = glob.glob(gen_model_dir + '/*_Best_Epoch.hdf5')
        if len(gen_models) == 0:
            print("***Process of generation automatically aborted!***")
            print("The {} directory does not contain any best generator model with high CUN score.\n".format(gen_model_dir))
            print("The training of the generator may have been aborted.\n")
            print("Please, train the generator again.\n")
            return
        len_gen_models = len(gen_models)
        print("The {} directory contains {} best generator model(s) with high CUN score.\n".format(gen_model_dir, len_gen_models))

        if self.prop_names_list is not None:
            for imodel_dir in infer_model_dir_list:
                if (not os.path.exists(imodel_dir)):
                    print("***Process of generation automatically aborted!***")
                    print("The {} directory does not exist.\n".format(imodel_dir))
                    print("Please, train the predictor first.\n")
                    return

                infer_models_tmp = glob.glob(imodel_dir + '/*.hdf5')
                if len(infer_models_tmp) == 0:
                    print("***Process of generation automatically aborted!***")
                    print("The {} directory does not contain any predictor model.\n".format(imodel_dir))
                    print("Please, train the predictor again.\n")
                    return
                len_infer_models = len(infer_models_tmp)
                print("The {} directory contains {} predictor model(s).".format(imodel_dir, len_infer_models))
                nfold_infer_models = np.unique([int(imodel.split('_Run_')[0].split('_')[-1]) for imodel in infer_models_tmp]).shape[0]
                print("The {} directory contains {}-fold cross-validation.".format(imodel_dir, nfold_infer_models))
                nrun_infer_models = np.unique([int(imodel.split('_Run_')[1].split('.')[0]) for imodel in infer_models_tmp]).shape[0]
                print("The {} directory contains {} runs per fold.\n".format(imodel_dir, nrun_infer_models))
        
        print("{} fold(s) and {} run(s) will be used for each property inference on request.\n".format(self.infer_use_n_folds, self.infer_use_n_runs))

        # Predictor(s) initialization --- Put it later when GPU options get an external fonction
        if self.prop_names_list is not None:
            self.prop_names_list_len = len(self.prop_names_list)
            self.infer_model_list = list()
            for iprop in range(self.prop_names_list_len):
                infer_model_tmp = loadmodel.LoadModel(data_name = self.prop_names_list[iprop],
                                                        augment = self.infer_augmentation, 
                                                        use_n_folds = self.infer_use_n_folds,
                                                        use_n_runs = self.infer_use_n_runs,
                                                        outdir = indir,
                                                        gpu_name = self.gpus,
                                                        strategy = self.strategy, 
                                                        log_verbose = False,
                                                        return_attention = False)
                self.infer_model_list.append(infer_model_tmp)
        else:
            self.prop_names_list_len = 0
            self.infer_model_list = None
            print("No property to infer.\n")
        ##
            
        # Setting up the trained model for generation, the generator vocabulary, and the predictor for inference
        
        # Tokens as a list
        self.gen_tokens = token.get_vocab(vocab_file)
        # Add 'pad', 'unk' tokens to the existing list
        self.gen_tokens.insert(0, 'unk')
        self.gen_tokens.insert(0, 'pad')
        self.vocab_size = len(self.gen_tokens)
        self.token_to_int = token.get_tokentoint(self.gen_tokens)
        self.int_to_token = token.get_inttotoken(self.gen_tokens)
        
        K.clear_session()
        # mirror_strategy workaround for model loading
        with tf.device(self.gpus[0].name): 
            self.gen_model = load_model('{}/{}_Model_Run_{}_Best_Epoch.hdf5'.format(gen_model_dir, data_name, g_run_number), 
                                        custom_objects={'SoftAttention': model.SoftAttention()})
        self.gen_max_length = self.gen_model.layers[0].output_shape[-1][1]

        print("*************************************")
        print("***Molecular generation initiated.***")
        print("*************************************\n")

    
    def display(self, prior, likelihood, posterior, int_tokens_list):
        # Display the prior, likelihood, and posterior distributions in a Jupyter notebook
        # prior, likelihood, posterior are 2D arrays where each row is a distribution
        # int_tokens_list is a list of lists of integers
        # Returns None
        
        clear_output(wait=True)
        
        plt.figure(figsize=(15,7))
        plt.subplot(311)
        markers, stemlines, baseline = plt.stem([ix for ix in range(prior.shape[1])], 
                                                prior.flatten().tolist(), 
                                                'k.-', 
                                                use_line_collection=True)
        plt.setp(baseline, color='k', linewidth=2, linestyle='--')
        plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black', color = 'y')
        plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
        plt.xticks(range(prior.shape[1]), 
                   ['' for iv in range(prior.shape[1])],
                   fontsize = 9, 
                   rotation = 90)
        plt.yticks(fontsize = 20)
        plt.ylabel('Prior', fontsize = 15, labelpad = 15)

        plt.subplot(312)
        markers, stemlines, baseline = plt.stem([ix for ix in range(likelihood.shape[1])], 
                                                likelihood.flatten().tolist(), 
                                                'k.-', 
                                                use_line_collection=True)
        plt.setp(baseline, color='k', linewidth=2, linestyle='--')
        plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black', color = 'g')
        plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
        plt.xticks(range(likelihood.shape[1]), 
                   ['' for iv in range(likelihood.shape[1])],
                   fontsize = 9, 
                   rotation = 90)
        plt.yticks(fontsize = 20)
        plt.ylabel('Gaussian Likelihood', fontsize = 15, labelpad = 15)

        plt.subplot(313)
        markers, stemlines, baseline = plt.stem([ix for ix in range(posterior.shape[1])], 
                                                posterior.flatten().tolist(), 
                                                'k.-', 
                                                use_line_collection=True)
        plt.setp(baseline, color='k', linewidth=2, linestyle='--')
        plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black', color = 'b')
        plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
        plt.xticks(range(posterior.shape[1]), 
                   [self.int_to_token[iitoken] for iitoken in int_tokens_list],
                   fontsize = 9, 
                   rotation = 90)
        plt.yticks(fontsize = 20)
        plt.ylabel('Posterior', fontsize = 15, labelpad = 15)
        plt.show();
    
    def generate(self, 
                 starter = None, 
                 n_generate = 1000, 
                 top_k = None, 
                 diversity = 1., 
                 batch_size = 128, 
                 target_sf = 'uniform', 
                 bounds_list = None, 
                 tolerance = 10, 
                 prop_gamma_list = None, 
                 prior_gamma = 1.):
        '''
        Generate SMILES from the model.
        starter: str
            String of tokens to start with (Default: None, i.e. random)
        n_generate: int 
            Number of SMILES to generate, must be a multiple of batch_size (Default: 1000)
        top_k: int
            Top k tokens to take into account for scoring (Default: None, i.e. no limit), must be > 0 and <= vocab_size (Default: None)
        diversity: floatDataSequence
            Diversity (or temperature) controller during sampling, must be > 0 (Default: 1.)
        batch_size: int
            Number of smiles generated in parallel, must be a multiple of n_generate (Default: 128)
        target_sf: str
            Target sampling function, either 'uniform' or 'gaussian' (Default: 'uniform')
        bounds_list : list
            list of properties' bounds (Default: None)
        tolerance: float
            Tolerance for the target sampling function (Default: 10)
        prop_gamma_list : list
            list of properties' gamma (Default: None)
        prior_gamma : float
            gamma for the prior (Default: 1.)

        Returns
        -------
        new_smiles_list: list
            List of generated SMILES
        '''

        if self.prop_names_list is not None:
            if prop_gamma_list is None:
                prop_gamma_list = [1.] * self.prop_names_list_len
            if (bounds_list is not None):
                bounds_list_len = len(bounds_list)
                prop_gamma_list_len = len(prop_gamma_list)
                if self.prop_names_list_len != ((bounds_list_len+prop_gamma_list_len) // 2):                    
                    print("The provided bounds_list and/or prop_gamma_list are empty or their length differs from the prop_name_list.\n")
                    return
            else:
                print("The provided bounds_list is empty.\n")
                return
        else:
            print("No property to infer.\n")

        # list of generated SMILES
        pad_token_int = self.token_to_int['pad']
        unk_token_int = self.token_to_int['unk']
        pre_token_int = self.token_to_int[' ']
        token_int_veto = [pad_token_int, unk_token_int, pre_token_int]
        suf_token_int = self.token_to_int[' ']
        gen_tokens_toint = [self.token_to_int[itoken] for itoken in self.gen_tokens]
       
        # control top_k according to vocab_size
        if top_k is not None:
            top_k = top_k if top_k <= self.vocab_size else self.vocab_size
        else:
            top_k = self.vocab_size
    
        # Array of temporary new SMILES
        starter_pre = [pre_token_int]
        if starter is not None:
            starter_pre += [self.token_to_int[ist] for ist in token.get_tokens(np.array([starter]))[0][1:-1]]
        starter_len = len(starter_pre)
        starter_row = np.array([pad_token_int]*(self.gen_max_length-starter_len)+starter_pre)
        new_smiles_tmp = np.copy(starter_row)
        new_smiles_tmp = np.tile(new_smiles_tmp, (n_generate, 1))
        # Array of new SMILES to return
        new_smiles = np.empty(shape=(0,self.gen_max_length), dtype=np.int32)
        new_smiles_shape = new_smiles.shape[0]
        
        # Array of targets (gaussians of sampled mean in [bounds_list range])
        # One Gaussian of fixed mean per temporary new SMILES per target property
        targets_array = np.empty(shape=(n_generate,0), dtype=np.float64)
        if self.prop_names_list is not None:
            for iinfer in range(self.prop_names_list_len):
                if target_sf == 'uniform':
                    targets_col_tmp = np.random.uniform(bounds_list[iinfer][0], bounds_list[iinfer][1], n_generate).reshape(-1,1)
                elif target_sf == 'gaussian':
                    targets_col_tmp = np.random.normal((bounds_list[iinfer][0]+bounds_list[iinfer][1])/2., 
                                                       (bounds_list[iinfer][1]-bounds_list[iinfer][0])/2., 
                                                       n_generate). reshape(-1,1)
                targets_array = np.concatenate([targets_array, targets_col_tmp], axis=-1)
            targets_array = np.tile(targets_array, (1,top_k)).reshape(n_generate*top_k,self.prop_names_list_len)
        
        # Generation evolution tracking
        cor_list = []
        uniq_list = []
        novel_list = []
        
        i = 0
        while new_smiles_shape < n_generate:
            # shape: (batch_size, vocab_size)
            prior = self.gen_model.predict(trainutils.DataSequence(new_smiles_tmp, None, None, batch_size)).astype('float64')
            # extract a substitution to the prior
            sub_prior = np.sort(prior)[:,-top_k:]
            # extract top_k integered tokens with highest probability of occurence
            sub_prior_tokens = np.argsort(prior)[:,-top_k:]
            sub_prior_tokens_list = sub_prior_tokens.flatten().tolist()
            
            if self.prop_names_list is not None:
                prob_smiles_list_toinfer = np.copy(new_smiles_tmp)
                prob_smiles_list_toinfer = np.tile(prob_smiles_list_toinfer, (1,top_k)).reshape(n_generate*top_k,self.gen_max_length)
                prob_smiles_list_toinfer[:,:(self.gen_max_length-1)] = prob_smiles_list_toinfer[:,1:]
                prob_smiles_list_toinfer[:,-1] = sub_prior_tokens_list
    
                # convert back to SMILES and remove special characters
                new_smiles_list_tmp = list()
                for ismiles in prob_smiles_list_toinfer:
                    smiles_tmp = list()
                    for itoken in ismiles: 
                        smiles_tmp.append(self.int_to_token[itoken])
                        smi_tmp = genutils.join_tokens(genutils.remove_schar([smiles_tmp]))
                    new_smiles_list_tmp.append(smi_tmp[0])

                lik_array = np.ones(shape=(n_generate,top_k), dtype='float64')
                for iinfer in range(self.prop_names_list_len):
                    lik_array_tmp = inference.infer(model=self.infer_model_list[iinfer],
                                                    data_smiles = new_smiles_list_tmp,
                                                    check_smiles = False, 
                                                    augment = False, 
                                                    batch_size = batch_size,
                                                    log_verbose = False)
                    lik_array_tmp_mean = lik_array_tmp['mean'].values.astype('float64')
                    
                    ##
                    # Maximum likelihood
                    ##
                    # Lowering the expected standard deviation --> diminish the correctness of generative model
                    # Predicted standard deviation can't be used as it is
                    # A cooling schedule and/or fake, large enough, standard deviation is necessary to drive a "correct" generation
                    # Heuristic - Ex.: 40C is close to the actual std from inference model
                    lik_array_tmp_std = np.ones(shape=lik_array_tmp.shape[0]) * tolerance
                    lik_min = (bounds_list[iinfer][0] - lik_array_tmp_mean) / lik_array_tmp_std
                    lik_max = (bounds_list[iinfer][1] - lik_array_tmp_mean) / lik_array_tmp_std
                    p_gen_min = np.array(list(map(genutils.t_cdf, lik_min)))
                    p_gen_max = np.array(list(map(genutils.t_cdf, lik_max)))
                    p_gen_in = p_gen_max - p_gen_min
                    p_gen_in[p_gen_in < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
                    p_gen_in = p_gen_in.reshape(n_generate,top_k)
                    lik_array *= (p_gen_in**prop_gamma_list[iinfer])
                    ##
                    #
                    ##

                posterior = (sub_prior**prior_gamma) * lik_array 
                posterior = genutils.p_to_one(posterior)
            else:
                posterior = sub_prior
            
            # shape: (n_generate,)
            new_tokens = self.sample(posterior, temperature = diversity)
            new_tokens = sub_prior_tokens[new_tokens]
            new_smiles_tmp[:,:(self.gen_max_length-1)] = new_smiles_tmp[:,1:]
            new_smiles_tmp[:,-1] = new_tokens
            finished_smiles_idx_tmp = np.where(new_tokens == suf_token_int)[0].tolist()
            unfinished_smiles_idx_tmp = np.where(new_tokens != suf_token_int)[0].tolist()
            if len(finished_smiles_idx_tmp) != 0:
                new_smiles = np.append(new_smiles, 
                                       new_smiles_tmp[finished_smiles_idx_tmp], 
                                       axis=0)
                new_smiles_tmp = new_smiles_tmp[unfinished_smiles_idx_tmp]
                new_smiles_tmp = np.append(new_smiles_tmp, 
                                           np.tile(np.copy(starter_row), 
                                                           (len(finished_smiles_idx_tmp), 1)), 
                                           axis = 0)
                new_smiles_shape = new_smiles.shape[0]
                if self.prop_names_list is not None:
                    targets_array = targets_array.reshape(n_generate,top_k,self.prop_names_list_len)
                    targets_array = targets_array[unfinished_smiles_idx_tmp,:,:].reshape(len(unfinished_smiles_idx_tmp)*top_k,self.prop_names_list_len)
                    targets_array_tmp = np.empty(shape=(len(finished_smiles_idx_tmp),0), dtype=np.float64)
                    for iinfer in range(self.prop_names_list_len):
                        if target_sf == 'uniform':
                            targets_col_tmp = np.random.uniform(bounds_list[iinfer][0], bounds_list[iinfer][1], len(finished_smiles_idx_tmp)).reshape(-1,1)
                        elif target_sf == 'gaussian':
                            targets_col_tmp = np.random.normal((bounds_list[iinfer][0]+bounds_list[iinfer][1])/2., 
                                                               (bounds_list[iinfer][1]-bounds_list[iinfer][0])/2., 
                                                               len(finished_smiles_idx_tmp)). reshape(-1,1)
                        targets_array_tmp = np.concatenate([targets_array_tmp, targets_col_tmp], axis=-1)
                    targets_array_tmp = np.tile(targets_array_tmp, (1,top_k)).reshape(len(finished_smiles_idx_tmp)*top_k,self.prop_names_list_len)
                    targets_array = np.concatenate([targets_array, targets_array_tmp], axis=0)
                print("Generation #{}".format(new_smiles_shape), end='\r') # Add updated average of targeted properties
            i+=1
                
        new_smiles_list = list()
        for ismiles in new_smiles:
            smiles_tmp = list()
            for itoken in ismiles: 
                smiles_tmp.append(self.int_to_token[itoken])
                smi_tmp = genutils.join_tokens(genutils.remove_schar([smiles_tmp]))
            try:
                mol_tmp = Chem.MolFromSmiles(smi_tmp[0])
                smi_tmp = Chem.MolToSmiles(mol_tmp)
                new_smiles_list.append(smi_tmp)
                print("Generation #{}".format(len(new_smiles_list)), end='\r')
            except:
                continue # do nothing
                    
        return new_smiles_list