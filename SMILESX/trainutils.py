"""Add main docstring discription

"""

import time
import numpy as np
import logging

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback

class DataSequence(Sequence):
    """Split data into batches for trainin

    Parameters
    ----------
    smiles:
        2D Numpy array of tokenized SMILES of shape (number_of_SMILES, max_length)
    extra: np.array
        2D Numpy array containing additional inputs of shape (number_of_SMILES, n_extra_features)
    props_set: np.array
        1D Numpy array of target propertyof of shape (number_of_SMILES,)
    TODO: check the exact shape)
    batch_size: int
        Batch size

    Returns
    -------
    A tuple containing a batch of input SMILES, extra data and property values.
    The inputs (SMILES, extra) are of a dictionary shape in accordance with Keras format.

    Examples:
        {"smiles": batch_smiles, "extra": batch_extra}, batch_prop - when extra data is provided
        {"smiles": batch_smiles}, batch_prop - when no extra data is provided
    """

    def __init__(self, smiles, extra, props, batch_size):
        self.smiles = smiles
        self.extra = extra
        self.props = props
#         print("EXACT SHAPE OF PROPERTIES IN SEQUENCE")
#         print(self.extra.shape)
        self.batch_size = int(batch_size)
        self.iepoch = 0

    def on_epoch_end(self):
        self.iepoch += 1
        
    def __len__(self):
        return int(np.ceil(len(self.smiles) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_smiles = self.smiles[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_prop = self.props[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.extra is not None:
            batch_extra = self.extra[idx * self.batch_size:(idx + 1) * self.batch_size]
            return {"smiles": batch_smiles, "extra": batch_extra}, batch_prop
        else:
            return {"smiles": batch_smiles}, batch_prop
##

class CyclicLR(Callback):
    """
    Implement a cyclical learning rate policy.
    
    The method is presented by Leslie N. Smith in the paper
    'Cyclical learning rates for training neural networks' (IEEE WACV, 2017)
    Learning rate is changed between two boundaries with some constant frequency.
    The boudaries ar define as `base_lr` and `max_lr`, and their difference corresponds
    to the amplitude. At every iteration, the learning rate is defined as the sum of `base_lr`
    and scaled amplitude. Scaling can be done on per-iteration or per-cycle bases.
    
    :class:`CyclicLR` class has three built-in scaling policies:
        - `triangular`
          Basic triangular cycle with no amplitude scaling.
        - `triangular2`
          Basic triangular cycle that scales initial amplitude by half each cycle.
        - `exp_range`
          Cycle that scales initial amplitude by gamma**(cycle iterations) at each iteration.
    

    Parameters
    ----------
        base_lr: float
            Initial learning rate which is the lower boundary in the cycle.
            (Default 0.001)
        max_lr: float
            Upper boundary for the learning rate used for amplitude calculation,
            may not be actually reached. (Default 0.006)
        step_size: int
            Number of training iterations per half cycle. Authors suggest setting 
            `step_size` to 2-8 x training iterations per epoch. (Default 2000)
        mode: {'triangular', 'triangular2', 'exp_range'}
            Policies which define the amplitude scaling as a function of iterations.
            If `scale_fn` is not `None`, `mode` argument is ignored. (Default 'triangular')
        gamma: float
            A constant used in 'exp_range' scaling function: gamma**(cycle iterations).
            (Default 1.)
        scale_fn: lambda
            Custom scaling policy defined by a single argument lambda function,
            where 0 <= scale_fn(x) <= 1 for all x >= 0. If `scale_fn` is not `None`,
            `mode` argument is ignored. (Default None)
        scale_mode: {'cycle', 'iterations'}
            Defines whether `scale_fn` is evaluated based on cycle number or cycle 
            iterations (training iterations since start of cycle). (Default 'cycle')
            
    Examples
    --------
    Basic usage:

    .. code-block:: python

        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                       step_size=2000., mode='triangular')
        model.fit(X_train, Y_train, callbacks=[clr])

    Class also supports custom scaling functions:

    .. code-block:: python

        clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                       step_size=2000., scale_fn=clr_fn,
                       scale_mode='cycle')
        model.fit(X_train, Y_train, callbacks=[clr])
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations
        
        Optional boundary/step size adjustment
        """
        
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

class StepDecay():
    """Step decay the learning rate during training
    
    Parameters
    ----------
    initAlpha: float
        Initial learning rate. (Default 1e-3)
    finalAlpha: float
        Final learning rate. (Default 1e-5)
    gamma: float
        NewAlpha = initAlpha * (gamma ** exp), where `exp` is determined 
        based on the desired number of epochs. (Default 0.95)
    epochs: int
        Desired number of epochs for training. (Default 100)
    """

    def __init__(self, initAlpha = 1e-3, finalAlpha = 1e-5, gamma = 0.95, epochs = 100):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.finalAlpha = finalAlpha
        self.gamma = gamma
        self.epochs = epochs
        self.beta = (np.log(self.finalAlpha) - np.log(self.initAlpha)) / np.log(self.gamma)
        self.dropEvery = self.epochs / self.beta

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = epoch%self.dropEvery # epoch starts from 0, callbacks called from the beginning
        alpha = self.initAlpha * (self.gamma ** exp)

        # return the learning rate
        return float(alpha)
##

class CosineAnneal(Callback):
    def __init__(self, initial_learning_rate = 1e-2, final_learning_rate = 1e-5, epochs = 100):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.epochs = epochs
      
    def __call__(self, epoch):
        # compute the learning rate for the current epoch        
        step = min(epoch, self.epochs)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.epochs))
        decayed = (1 - alpha) * cosine_decay + self.final_learning_rate
        alpha = self.initial_learning_rate * decayed
        # return the learning rate
        return float(alpha)
##

class LoggingCallback(Callback):
    """Implement custom logging class to continue logging during training
    """
    
    # Callback that logs message at end of epoch.
    def __init__(self, print_fcn=print, verbose=1):
        Callback.__init__(self)
        self.print_fcn = print_fcn
        self.start = 0
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs={}):
        self.start = time.time()
        
    def on_epoch_end(self, epoch, logs={}):
        if self.verbose!=0:
            msg = "{Epoch: %i} duration: %.3f secs %s" % (epoch, 
                                                         time.time()-self.start, 
                                                         ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
            self.print_fcn(msg)
##

class IgnoreBeginningSaveBest(Callback):
    """Save the best weights only after some number of epochs has been trained

    Parameters
    ----------
    filepath: str
        Path to the directory where the model is saved.
    n_epochs: int
        Number of epochs requested for training.
    best_loss: float
        Best loss achieved so far. (Default: np.Inf)
    best_epoch: int
        Number of the epoch with lowest validation loss achieved so far. (Default: 0)
    initial_epoch: int
        The number of the initial epoch (needed for continuous training). (Default: 0)
    ignore_first_epochs: int
        How many epochs to ignore in the beginning of the training before to start
        registering the best validation loss. (Default 0)
    last: bool
        In case of multi-step training, defines whether the last step is run.
        Used for final printouts. (Default: False)
    """

    def __init__(self, filepath, n_epochs, best_loss = np.Inf, best_epoch = 0, initial_epoch=0, ignore_first_epochs=0, last = False):
        super(IgnoreBeginningSaveBest, self).__init__()

        self.filepath = filepath
        self.ignore_first_epochs = ignore_first_epochs
        self.initial_epoch = initial_epoch
        self.best_loss = best_loss
        self.best_epoch = best_epoch
        self.last = last
        self.end_epoch = self.initial_epoch + n_epochs

        # Store the weights at which the minimum loss occurs
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        # Start saving only starting from a certain epoch
        if epoch > self.ignore_first_epochs:
            if np.less(current_loss, self.best_loss):
                self.best_loss = current_loss
                # Record the best weights if the current loss result is lower
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch

    def on_train_end(self, logs=None):
        # Save the model with the best weights if more apochs has spun than requested to ignore
        if self.ignore_first_epochs <= self.end_epoch:
            if self.best_weights is not None:
                # Save current weights
                self.curr_weights = self.model.get_weights()
                # Save the model with the best weights
                self.model.set_weights(self.best_weights)
                self.model.save(self.filepath)
                if not self.last:
                    logging.info("Updating current best validation loss in accordance with epoch #{}"\
                                 .format(self.best_epoch))
                    # Return back to the current state to continue training
                    self.model.set_weights(self.curr_weights)   
            if self.last:
                logging.info("")
                logging.info("The best validation loss of {:.2f} is achieved at epoch #{}"\
                             .format(self.best_loss, self.best_epoch))
                logging.info("")
                logging.info("Saving the best model to {}"\
                             .format(self.filepath))
##

## Custom metric 
class CxUxN(object):
    """Implement CxUxN score calculation during training

    Parameters
    ----------
    init_data: list
        List of initial SMILES
    data_name: str
        Name of the dataset
    vocab_path: str
        Path to the vocabulary file
    gen_max_length: int
        Maximum length of the generated SMILES
    gpus: list
        List of GPUs to use
    model_init: keras model
        Model to use for generation
    n_generate: int
        Number of SMILES to generate
    warm_up: int
        Number of epochs to wait before to start generation
    batch_size: int
        Batch size
    print_fcn: function
        Function to use for printing
    save_dir: str
        Path to the directory where the model is saved.

    Returns
    -------
    A tuple containing a batch of input SMILES, extra data and property values.
    The inputs (SMILES, extra) are of a dictionary shape in accordance with Keras format.

    Examples:
        {"smiles": batch_smiles, "extra": batch_extra}, batch_prop - when extra data is provided
        {"smiles": batch_smiles}, batch_prop - when no extra data is provided
    """
    
    def __init__(self, init_data, data_name, 
                 vocab_path, gen_max_length,
                 gpus, 
                 model_init, 
                 n_generate = 1000, 
                 warm_up = 0, 
                 batch_size = 128, 
                 print_fcn = print, 
                 save_dir = None):
        #super(CxUxN, self).__init__() #Callback.__init__(self) 
        self.init_data_set = set(init_data)
        self.data_name = data_name
        # Tokens as a list
        self.gen_tokens = token.get_vocab(vocab_path)
        self.gen_max_length = gen_max_length
        self.gpus = gpus
        self.model_init = model_init
        self.n_generate = n_generate
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.print_fcn = print_fcn
        self.save_dir = save_dir
        
        # Add 'pad', 'unk' tokens to the existing list
        self.vocab_size = len(self.gen_tokens)
        self.gen_tokens, self.vocab_size = token.add_extra_tokens(self.gen_tokens, self.vocab_size)
        token_to_int = token.get_tokentoint(self.gen_tokens)
        self.int_to_token = token.get_inttotoken(self.gen_tokens)
        
        self.pad_token_int = token_to_int['pad']
        self.pre_token_int = token_to_int[' ']
        self.suf_token_int = token_to_int[' ']
        
        # model loading on CPU
#         model_weights = self.model.get_weights()
        lstmunits = self.model_init.layers[2].output_shape[-1]//2
        denseunits = self.model_init.layers[3].output_shape[-1]
        embedding = self.model_init.layers[1].output_shape[-1]
        with tf.device(self.gpus[-1].name):
            K.clear_session()
            # Model's architecture for generation
            self.gen_model = model.LSTMAttModel.create(inputtokens = self.gen_max_length, 
                                                       vocabsize = self.vocab_size, 
                                                       lstmunits= lstmunits, 
                                                       denseunits = denseunits, 
                                                       embedding = embedding)
        
        # Scores list
        self.cor_list = []
        self.uniq_list = []
        self.novel_list = []
        self.cun_list = []
        
        # CxUxN Score max and related epoch
        self.cun_max = 0
        self.best_epoch = 0

    ## Tools for generation
    def normalize(self, preds, temperature=1.0):
        preds = preds.astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds, axis=1).reshape(-1,1)
        return preds

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = self.normalize(preds, temperature)
        probas = np.zeros(preds.shape)
        for ipred in range(preds.shape[0]):
            probas[ipred] = np.random.multinomial(1, preds[ipred], 1)[0]
        return probas.astype('bool') #np.argmax(probas, axis=1)

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
        
    def evaluation(self, epoch):
        if epoch >= self.warm_up:
            start_time = time.time()

            self.gen_model.load_weights(self.save_dir+'LSTMAtt_'+self.data_name+'_model_epoch{:02d}.hdf5'.format(epoch+1))

            # array of temporary new SMILES
            starter_row = np.array([self.pad_token_int]*(self.gen_max_length-1)+[self.pre_token_int])
            new_smiles_tmp = np.copy(starter_row)
            new_smiles_tmp = np.tile(new_smiles_tmp, (self.n_generate, 1))
            # array of new SMILES to return
            new_smiles = np.empty(shape=(0,self.gen_max_length), dtype=np.int16)
            new_smiles_shape = new_smiles.shape[0]
            ##

            # Generate new SMILES
            while new_smiles_shape < self.n_generate:
                # shape: (batch_size, vocab_size) 
                prior = self.gen_model.predict(DataSequence(new_smiles_tmp, self.batch_size), 
                                               max_queue_size = self.batch_size).astype('float64')
                # extract a substitution to the prior
                top_k = self.vocab_size
                sub_prior = np.sort(prior)[:,-top_k:]
                # extract top_k integered tokens with highest probability of occurence
                sub_prior_tokens = np.argsort(prior)[:,-top_k:]

                # shape: (n_generate,)
                new_tokens = self.sample(sub_prior)
                new_tokens = sub_prior_tokens[new_tokens]
                new_smiles_tmp[:,:(self.gen_max_length-1)] = new_smiles_tmp[:,1:]
                new_smiles_tmp[:,-1] = new_tokens
                finished_smiles_idx_tmp = np.where(new_tokens == self.suf_token_int)[0].tolist()
                unfinished_smiles_idx_tmp = np.where(new_tokens != self.suf_token_int)[0].tolist()
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

            new_smiles_list = list()
            for ismiles in new_smiles:
                smiles_tmp = list()
                for itoken in ismiles: 
                    smiles_tmp.append(self.int_to_token[itoken])
                try:
                    smi_tmp = self.join_tokens(self.remove_schar([smiles_tmp]))
                    mol_tmp = Chem.MolFromSmiles(smi_tmp[0])
                    smi_tmp = Chem.MolToSmiles(mol_tmp)
                    new_smiles_list.append(smi_tmp)
                except:
                    continue

            new_smiles_list_len = len(new_smiles_list)
            if new_smiles_list_len > 0:
                correctness = (new_smiles_list_len/self.n_generate)
                unique_smiles = np.unique(new_smiles_list)
                uniqueness = unique_smiles.shape[0]/float(new_smiles_list_len)
                novelty = len(set(unique_smiles).difference(self.init_data_set))/unique_smiles.shape[0]

                cun_val = (correctness * uniqueness * novelty)
            else:
                correctness = 0
                uniqueness = 0
                novelty = 0
                cun_val = 0

            # Keep the best CxUxN score with the related epoch in memory
            if cun_val > self.cun_max:
                self.cun_max = cun_val
                self.best_epoch = epoch

            # Update CxUxN scores list
            self.cor_list.append(correctness)
            self.uniq_list.append(uniqueness)
            self.novel_list.append(novelty)
            self.cun_list.append(cun_val)
            time_range = time.time() - start_time
            self.print_fcn("{{Epoch: {}}} {} generations, {} valid generations, CxUxN score: {:.2f} %, Duration: {:.3f} secs".format(epoch, 
                                                                                                                                     new_smiles_shape, 
                                                                                                                                     new_smiles_list_len, 
                                                                                                                                     cun_val*100., 
                                                                                                                                     time_range))
            
    def on_evaluation_end(self):
        # Save best weights
        shutil.copy(self.save_dir+'LSTMAtt_'+self.data_name+'_model_epoch{:02d}.hdf5'.format(self.best_epoch+1), 
                    self.save_dir+'LSTMAtt_'+self.data_name+'_model_best.hdf5')
        self.print_fcn("\nBest CxUxN score @ Epoch #{}\n".format(self.best_epoch))
        # plot scores history
        if self.save_dir is not None:
            plt.plot(self.cor_list)
            plt.plot(self.uniq_list)
            plt.plot(self.novel_list)
            plt.plot(self.cun_list)
            plt.legend(['Correctness','Uniqueness','Novelty','CxUxN'], loc='lower right')
            plt.ylabel('Score')
            plt.title('')
            plt.xlabel('Epoch')
            plt.savefig(self.save_dir+'History_CxUxN_score_LSTMAtt_'+self.data_name+'_model.png', bbox_inches='tight')
            plt.close()
##