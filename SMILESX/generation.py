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

from SMILES import model, inference, token, trainutils

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
    data_units : str
        property's SI units
    k_fold_number : int
        number of k-folds used for inference (Default: None, i.e. automatically detect k_fold_number from main.Main phase)
    k_fold_index : int
        k-fold index to be used for inference (Default: None, i.e. use all the models, then average)
    gen_augmentation : bool
        SMILES's augmentation at generator training (Default: False)
    infer_augmentation : bool   
        SMILES's augmentation at predictor training (Default: False)
    indir : str 
        directory of already trained prediction models (*.hdf5) and vocabulary (*.txt) (Default: './output/')
    outdir : str
        directory for outputs (plots + .txt files) -> 'Inference/'+'{}/{}/'.format(data_name,g_dir_temp) is then created (Default: './output/')
    vocab_name : str
        vocabulary's name (Default: 'vocab')
    n_gpus : int
        number of GPUs to use (Default: 1)
    gpus_list : list
        list of GPUs to use (Default: None)
    gpus_debug : bool
        debug mode for GPUs (Default: False)
    prop_names_list : list  
        list of properties' names (Default: None)
    bounds_list : list
        list of properties' bounds (Default: None)
    prop_gamma_list : list
        list of properties' gamma (Default: None)
    prior_gamma : float
        gamma for the prior (Default: 1.)

    Methods
    -------
    __init__(self,
             data_name,
             data_units = '',
             k_fold_number = None,
             k_fold_index = None,
             gen_augmentation = False,
             infer_augmentation = False,
             indir = "./output/",
             outdir = "./output/",
             vocab_name = 'vocab',
             n_gpus = 1,
             gpus_list = None,
             gpus_debug = False,
             prop_names_list = None,
             bounds_list = None,
             prop_gamma_list = None,
             prior_gamma = 1.)
    Initialize the class Generation with the given parameters (see Attributes). 
    '''

    def __init__(self, 
                 data_name, 
                 data_units = '',
                 k_fold_number = None,
                 k_fold_index = None, 
                 gen_augmentation = False, 
                 infer_augmentation = False, 
                 indir = "./output/", 
                 outdir = "./output/", 
                 vocab_name = 'vocab', 
                 n_gpus = 1, 
                 gpus_list = None, 
                 gpus_debug = False,
                 prop_names_list = None, 
                 bounds_list = None, 
                 prop_gamma_list = None, 
                 prior_gamma = 1.):
        
        # GPUs options
        #if n_gpus > 1:
        self.strategy, gpus = utils.set_gpuoptions(n_gpus = n_gpus, 
                                                   gpus_list = gpus_list, 
                                                   gpus_debug = gpus_debug)
        if self.strategy is None:
            return
        ##
        
        self.data_name = data_name
        self.data_units = data_units
        self.k_fold_number = k_fold_number
        self.k_fold_index = k_fold_index
        self.gen_augmentation = gen_augmentation
        self.infer_augmentation = infer_augmentation
        self.prop_names_list = prop_names_list
        self.bounds_list = bounds_list
        self.prop_gamma_list = prop_gamma_list
        self.prior_gamma = prior_gamma
        
        if self.gen_augmentation:
            g_dir_temp = 'Augm'
        else:
            g_dir_temp = 'Can'
        
        self.input_dir = indir+'LanguageModel/'+'{}/{}/'.format(data_name,g_dir_temp)
        self.vocab_dir = indir+'Vocabulary/'+'{}/{}/'.format(vocab_name,g_dir_temp)
        if not os.path.exists(self.input_dir):
            print("***Process of generation automatically aborted!***")
            print("The {} directory does not exist.\n".format(self.input_dir))
            return
        if not os.path.exists(self.vocab_dir):
            print("***Process of generation automatically aborted!***")
            print("The {} directory does not exist.\n".format(self.vocab_dir))
            return
        self.save_dir = outdir+'Generation/'+'{}/{}/'.format(data_name,g_dir_temp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        for itype in ["txt","hdf5"]:
            if itype == "txt":
                exists_file = glob.glob(self.vocab_dir + "*." + itype)
            else:
                exists_file = glob.glob(self.input_dir + "*." + itype)
            exists_file_len = len(exists_file)
            if exists_file_len == 0:
                print("***Process of generation automatically aborted!***")
                if (itype == "hdf5"):
                    print("The {} directory does not contain any trained language model (*.hdf5 file).\n".format(self.input_dir))
                else:
                    print("The {} directory does not contain any vocabulary (Vocabulary.txt file).\n".format(self.vocab_dir))
                return
        
        # Predictor(s) initialization --- Put it later when GPU options get an external fonction
        if (self.prop_names_list is not None):
            self.prop_names_list_len = len(self.prop_names_list)
            self.infer_class_list = list()
            if self.prop_gamma_list is None:
                self.prop_gamma_list = [1.] * self.prop_names_list_len
            if (bounds_list is not None):
                bounds_list_len = len(self.bounds_list)
                prop_gamma_list_len = len(self.prop_gamma_list)
                if self.prop_names_list_len == ((bounds_list_len+prop_gamma_list_len) // 2):
                    for iprop in range(self.prop_names_list_len):
                        infer_class_tmp = inference.Inference(data_name = self.prop_names_list[iprop], 
                                                              data_units = self.data_units,
                                                              k_fold_number = self.k_fold_number,
                                                              k_fold_index = self.k_fold_index, 
                                                              augmentation = self.infer_augmentation, 
                                                              indir = indir, 
                                                              outdir = outdir,
                                                              vocab_name = vocab_name, 
                                                              n_gpus = n_gpus, 
                                                              gpus = gpus, 
                                                              gpus_list = gpus_list, 
                                                              gpus_debug = gpus_debug)
                        self.infer_class_list.append(infer_class_tmp)
                else:
                    print("The provided bounds_list and/or prop_gamma_list are empty or their length differs from the prop_name_list.\n")
                    return
            
        # Setting up the trained model for generation, the generator vocabulary, and the predictor for inference
        
        # Tokens as a list
        self.gen_tokens = token.get_vocab(self.vocab_dir+'Vocabulary.txt')
        # Add 'pad', 'unk' tokens to the existing list
        self.vocab_size = len(self.gen_tokens)
        self.gen_tokens, self.vocab_size = token.add_extra_tokens(self.gen_tokens, self.vocab_size)
        self.token_to_int = token.get_tokentoint(self.gen_tokens)
        self.int_to_token = token.get_inttotoken(self.gen_tokens)
        
        K.clear_session()
        # mirror_strategy workaround for model loading
        with tf.device(gpus[0].name): 
            self.gen_model = load_model(self.input_dir+'LSTMAtt_'+self.data_name+'_model_best.hdf5', 
                                        custom_objects={'AttentionM': model.AttentionM()})
        self.gen_max_length = self.gen_model.layers[0].output_shape[-1][1]

        print("*************************************")
        print("***Molecular generation initiated.***")
        print("*************************************\n")
    
    def p_to_one(self, dist):
        if len(dist.shape) == 1:
            dist = dist.reshape(1,-1)
        return dist/np.sum(dist, axis=1).reshape(-1,1)
    
    # Equivalent to p_to_one without temperature
    def normalize(self, preds, temperature=1.0):
        preds = preds.astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds, axis=1).reshape(-1,1)
        return preds
    
    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis).reshape(-1,1))
        return e_x / np.sum(e_x, axis=axis).reshape(-1,1)
    
    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = self.normalize(preds, temperature)
        probas = np.zeros(preds.shape)
        for ipred in range(preds.shape[0]):
            probas[ipred] = np.random.multinomial(1, preds[ipred], 1)[0]
        return probas.astype('bool') #np.argmax(preds, axis=1)
    
    def remove_from_list(self, list_tmp, to_remove = ''): # list_tmp = list(list())
        return [list(filter(lambda t: t != to_remove, ilist)) for ilist in list_tmp]

    def remove_schar(self, list_tmp): # remove 'pad', '!', 'E' characters
        list_tmp = self.remove_from_list(list_tmp, 'pad')
        list_tmp = self.remove_from_list(list_tmp, '!')
        list_tmp = self.remove_from_list(list_tmp, 'E')
        return list_tmp

    def join_tokens(self, list_tmp): 
        list_tmp = [''.join(ismiles) for ismiles in list_tmp]
        return list_tmp
    
    def t_cdf(self, x):
        return stats.t.cdf(x, 10000)
    
    def display(self, prior, likelihood, posterior, int_tokens_list):
        
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
    
    def generate(self, starter = None, n_generate = 1000, top_k = None, diversity = 1., batch_size = 128, target_sf = 'uniform'):
        '''
        Generate SMILES from the model.
        starter: str
            String of tokens to start with (Default: None, i.e. random)
        n_generate: int 
            Number of SMILES to generate, must be a multiple of batch_size (Default: 1000)
        top_k: int
            Top k tokens to take into account for scoring (Default: None, i.e. no limit), must be > 0 and <= vocab_size (Default: None)
        diversity: float
            Diversity (or temperature) controller during sampling, must be > 0 (Default: 1.)
        batch_size: int
            Number of smiles generated in parallel, must be a multiple of n_generate (Default: 128)
        target_sf: str
            Target sampling function, either 'uniform' or 'prior' (Default: 'uniform')

        Returns
        -------
        new_smiles_list: list
            List of generated SMILES
        '''

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
                    targets_col_tmp = np.random.uniform(self.bounds_list[iinfer][0], self.bounds_list[iinfer][1], n_generate).reshape(-1,1)
                elif target_sf == 'gaussian':
                    targets_col_tmp = np.random.normal((self.bounds_list[iinfer][0]+self.bounds_list[iinfer][1])/2., 
                                                       (self.bounds_list[iinfer][1]-self.bounds_list[iinfer][0])/2., 
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
    
                lik_array = np.ones(shape=(n_generate,top_k), dtype='float64')
                for iinfer in range(self.prop_names_list_len):
                    lik_array_tmp = self.infer_class_list[iinfer].infer(smiles_input = prob_smiles_list_toinfer, 
                                                                        generation = True, 
                                                                        batch_size = batch_size)
                    lik_array_tmp_mean = lik_array_tmp.ens_pred_mean.values.astype('float64')#.reshape(-1,1)
                    
                    ##
                    # Maximum likelihood
                    ##
                    # Lowering the expected standard deviation --> diminish the correctness of generative model
                    # Predicted standard deviation can't be used as it is
                    # A cooling schedule and/or fake, large enough, standard deviation is necessary to drive a "correct" generation
                    # Not theory strong
                    lik_array_tmp_std = np.ones(shape=lik_array_tmp.shape[0]) * 20. # 40C close to actual std from inference model
                    lik_min = (self.bounds_list[iinfer][0] - lik_array_tmp_mean) / lik_array_tmp_std
                    lik_max = (self.bounds_list[iinfer][1] - lik_array_tmp_mean) / lik_array_tmp_std
                    p_gen_min = np.array(list(map(self.t_cdf, lik_min)))
                    p_gen_max = np.array(list(map(self.t_cdf, lik_max)))
                    p_gen_in = p_gen_max - p_gen_min
                    p_gen_in[p_gen_in < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
                    p_gen_in = p_gen_in.reshape(n_generate,top_k)
                    lik_array *= (p_gen_in**self.prop_gamma_list[iinfer])
                    ##
                    #
                    ##
                                    
                posterior = (sub_prior**self.prior_gamma) * lik_array 
                posterior = self.p_to_one(posterior)
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
                            targets_col_tmp = np.random.uniform(self.bounds_list[iinfer][0], self.bounds_list[iinfer][1], len(finished_smiles_idx_tmp)).reshape(-1,1)
                        elif target_sf == 'gaussian':
                            targets_col_tmp = np.random.normal((self.bounds_list[iinfer][0]+self.bounds_list[iinfer][1])/2., 
                                                               (self.bounds_list[iinfer][1]-self.bounds_list[iinfer][0])/2., 
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
                smi_tmp = self.join_tokens(self.remove_schar([smiles_tmp]))
            try:
                mol_tmp = Chem.MolFromSmiles(smi_tmp[0])
                smi_tmp = Chem.MolToSmiles(mol_tmp)
                new_smiles_list.append(smi_tmp)
                print("Generation #{}".format(len(new_smiles_list)), end='\r')
            except:
                continue
                # do nothing
                    
        return new_smiles_list