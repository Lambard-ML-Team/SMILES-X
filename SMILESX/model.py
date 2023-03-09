"""Add main docstring discription

"""

import collections

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Embedding
from tensorflow.keras.layers import Bidirectional, TimeDistributed, LSTM
from tensorflow.keras.layers import concatenate

import tensorflow as tf

class SoftAttention(Layer):
    def __init__(self, geom_search=False, return_prob=False, weight=None, **kwargs):
        """Initializes attention layer 

        Custom attention layer modified from https://github.com/sujitpal/eeap-examples
        
        Parameters
        ----------
        geom_search: bool
            Whether the model is built for training/prediction or for trainless geometry search.
            If `True`, weight are initialized with a shared constant value, which is set by
            `weight` parameter. If `False`, GlorotNormal weight initialization is applied.
            (Default: False)
        return_prop: bool
            Whether the model is used for training/prediction or for interpretation.
            If `True`, a 2D tensor is returned during the call. If `False`, a 1D
            attention vector is returned. (Default: False)
        weight: int, optional
            The value of the shared constant weights used for layer initialization. (Default: None)
            
        
        Examples
        --------
        Can be integrated within a model as follows:
        
        .. code-block:: python
        
            enc = LSTM(embed_size, return_sequences=True)(...)
            att = SoftAttention(geom_search=False, return_prob=False, weight=None)
            att = att(enc, mask=None)
        """
        
        self.geom_search = geom_search
        self.return_prob = return_prob
        self.weight = weight
        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """Builds and initializes attention layer
        
        If `geom_search` parameter is set to `True`, the weights are initialized 
        with a shared constant value defined with `weight` parameter. If `geom_search`
        is set to `False`, GlorotNormal weight initialization is applied.
        
        Weights' tensor shape is (EMBED_SIZE, 1),
        bias tensor shape is (MAX_TOKENS, 1).
        
        Parameters
        ----------
        input_shape: tuple
            Input tensor shape. Passed internally by Keras between layers.
        """

        if self.geom_search:
            att_initializer = tf.keras.initializers.constant(value=self.weight)
        else:
            att_initializer = tf.keras.initializers.GlorotNormal()

        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer=att_initializer)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(SoftAttention, self).build(input_shape)

    def call(self, x, mask=None):
        """Computes an attention vector on an input matrix
        
        Collapses the tokens dimension by summing up the products of
        the each of the token's weights with corresponding to this token matrix values.
        The weights are optimized during the training.

        Dimensions:
        et: (batch_size, max_tokens)
        at: (batch_size, max_tokens)
        ot: (batch_size, max_tokens, tdense_units)

        Parameters
        ----------
        x: tensor
            The output from the time-distributed dense layer (batch_size, max_tokens, tdense_units).
        mask: tensor (optional)
            The mask to apply.

        Returns
        ------
        During training and prediction
            2D tensor of shape (batch_size, embed_size)
        During interpretation (visualization)
            1D tensor of shape (max_tokens,)
        """
        
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        atx = K.expand_dims(at, axis=-1)
        if self.return_prob:
            return atx # For visualization of the attention weights
        else:
            ot = x * atx
            return K.sum(ot, axis=1) # For prediction

    def compute_mask(self, input, input_mask=None):
        """Prevent the mask being passed to the next layers"""
        return None

    def compute_output_shape(self, input_shape):
        """Compute output tensor shape"""
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        """Get configration of the layer"""
        base_config = super(SoftAttention, self).get_config()
        base_config['weight'] = self.weight
        return base_config
##

class LSTMAttModel:
    """Builds neural architecture of the SMILESX

    Parameters
    ----------
    input_tokens: int
        Maximum length for the encoded and tokenized SMILES
    vocab_size: int
        Size of the vocabulary
    embed_units: int
        Dimension of the embedding
    lstm_units: int
        The number of LSTM units
    tdense_units: int
        The number of dense units
    geom_search: bool
        Whether to initialize the weight with shared constant for geometry search (Defaults: False)
    return_prob: bool
        Whether to return the attention vector or not (Default: False)
    weight: int, float
        The value of the shared constant weights used for layer initialization (Default: None)
    model_type: str
        The type of the model. Can be either 'regression' (last_activation = 'linear') or 
        'classification' (last_activation = 'sigmoid'). (Default: 'regression')

    Returns
    -------
    keras.Model
        A model in the Keras API format
    """

    @staticmethod
    def create(input_tokens,
               vocab_size,
               embed_units=32,
               lstm_units=16,
               tdense_units=16,
               dense_depth=None,
               extra_dim=None,
               geom_search=False,
               return_prob=False,
               weight = None, 
               model_type = 'regression'):

        smiles_input = Input(shape=(int(input_tokens),), name="smiles") #shape=(input_tokens,)

        # Initialize with constant weights during geometry search
        if geom_search:
            embeddings_initializer = tf.keras.initializers.RandomNormal(mean=weight, stddev=weight/10, seed=0)
            recurrent_initializer = tf.keras.initializers.constant(value=weight)
            kernel_initializer = tf.keras.initializers.constant(value=weight)
        # Initialize for training
        else:
            embeddings_initializer = tf.keras.initializers.he_uniform()
            recurrent_initializer = tf.keras.initializers.Orthogonal(gain=1.0)
            kernel_initializer = tf.keras.initializers.GlorotUniform()

        embedding = Embedding(input_dim=int(vocab_size),
                              output_dim=int(embed_units),
                              input_length=int(input_tokens),
                              embeddings_initializer=embeddings_initializer)
        smiles_net = embedding(smiles_input)

        # Bidirectional LSTM layer
        lstm = Bidirectional(LSTM(int(lstm_units),
                             return_sequences=True,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer))
        smiles_net = lstm(smiles_net)

        # Time distributed layer
        timedist = TimeDistributed(Dense(int(tdense_units),
                                         kernel_initializer=kernel_initializer))
        smiles_net = timedist(smiles_net)

        # Custom attention layer
        attention = SoftAttention(geom_search=geom_search,
                               return_prob=return_prob,
                               weight=weight,
                               name="attention")
        smiles_net = attention(smiles_net)

        # In case additional inputs of 'additional_input' are added
        if extra_dim is not None:
            extra_input = Input(shape=(int(extra_dim),), name="extra")
            smiles_net = concatenate([smiles_net, extra_input])

        # In case where additional nonlinearity is added after extra input
        if (dense_depth is not None or dense_depth > 0):
            dense_units = tdense_units
            # Accepts a list in case there user requests to extend the model with multiple layers
            for dense in range(dense_depth):
                dense_units=dense_units // 2
                # Do not add layers consisting of a single unit
                if dense_units == 1:
                    break
                smiles_net = Dense(int(dense_units), activation="relu", kernel_initializer=kernel_initializer)(smiles_net)

        # Output layer
        last_activation = {'regression': 'linear', 'classification': 'sigmoid'}[model_type]
        smiles_net = Dense(1, activation=last_activation, kernel_initializer=kernel_initializer)(smiles_net)
        if extra_dim is not None:
            model = Model(inputs=[smiles_input, extra_input],
                          outputs=smiles_net)
        else:
            model = Model(inputs=[smiles_input],
                          outputs=smiles_net)

        return model
##