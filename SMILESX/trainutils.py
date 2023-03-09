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