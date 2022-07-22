"""DNA models.

Provides models trained with DNA sequence windows.
"""

from __future__ import division
from __future__ import print_function

import inspect

from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kr
from tensorflow.keras import models as km
from tensorflow.keras.layers import concatenate

from mettl.utils import get_from_module, is_input_layer


class Model(object):
    """Abstract model call.

    Abstract class of DNA, CpG, and Joint models.

    Parameters
    ----------
    dropout: float
        Dropout rate.
    l1_decay: float
        L1 weight decay.
    l2_decay: float
        L2 weight decay.
    init: str
        Name of Keras initialization.
    """

    def __init__(self, dropout=0.0, l1_decay=0.0, l2_decay=0.0,
                 batch_norm=False, init='glorot_uniform'):
        self.dropout = dropout
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay
        self.batch_norm = batch_norm
        self.init = init
        self.name = self.__class__.__name__
        self.scope = None

    def inputs(self, *args, **kwargs):
        """Return list of Keras model inputs."""
        pass

    def _build(self, input, output):
        """Build final model at the end of `__call__`."""
        model = km.Model(input, output, name=self.name)
        if self.scope:
            for layer in model.layers:
                if not is_input_layer(layer):
                    layer._name = '%s/%s' % (self.scope, layer.name)
        return model

    def __call__(self, inputs=None):
        """Build model.

        Parameters
        ----------
        inputs: list
            Keras model inputs
        """
        pass


class MetNet(Model):
    """Abstract class of a CpG model."""

    def __init__(self, *args, **kwargs):
        super(MetNet, self).__init__(*args, **kwargs)
        self.scope = 'cpg'

    def inputs(self, cpg_wlen, replicate_names):
        inputs = []
        shape = (len(replicate_names), cpg_wlen)
        inputs.append(kl.Input(shape=shape, name='cpg/state'))
        inputs.append(kl.Input(shape=shape, name='cpg/dist'))
        return inputs

    def _merge_inputs(self, inputs):
        return concatenate(inputs, axis=2)


class MetRnnL1(MetNet):
    """Bidirectional GRU with one layer.

    .. code::

        Parameters: 810,000
        Specification: fc[256]_bgru[256]_do
    """

    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(MetRnnL1, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def _replicate_model(self, input):
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(input)
        x = kl.Activation(self.act_replicate)(x)

        return km.Model(input, x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        #         shape = getattr(x, '_keras_shape')
        shape = x.get_shape()
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        gru = kl.GRU(256, kernel_regularizer=kernel_regularizer)
        x = kl.Bidirectional(gru)(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class SeqNet(Model):
    """Abstract class of a DNA model."""

    def __init__(self, *args, **kwargs):
        super(SeqNet, self).__init__(*args, **kwargs)
        self.scope = 'dna'

    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]


class SeqCnnL1h128(SeqNet):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(SeqCnnL1h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(self.l1_decay, self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class SeqCnnL1h256(SeqCnnL1h128):
    """CNN with one convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_fc[256]_do
    """

    def __init__(self, *args, **kwargs):
        super(SeqCnnL1h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class SeqCnnL2h128(SeqNet):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(SeqCnnL2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class SeqCnnL2h256(SeqCnnL2h128):
    """CNN with two convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[256]_do
    """

    def __init__(self, *args, **kwargs):
        super(SeqCnnL2h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class JointNet(Model):
    """Abstract class of a Joint model."""

    def __init__(self, *args, **kwargs):
        super(JointNet, self).__init__(*args, **kwargs)
        self.mode = 'concat'
        self.scope = 'joint'

    def _get_inputs_outputs(self, models):
        inputs = []
        outputs = []
        for model in models:
            inputs.extend(model.inputs)
            outputs.extend(model.outputs)
        return (inputs, outputs)

    def _build(self, models, layers=[]):
        for layer in layers:
            layer._name = '%s/%s' % (self.scope, layer._name)

        inputs, outputs = self._get_inputs_outputs(models)
        x = concatenate(outputs)
        for layer in layers:
            x = layer(x)

        model = km.Model(inputs, x, name=self.name)
        return model


class JointL0(JointNet):
    """Concatenates inputs without trainable layers.

    .. code::

        Parameters: 0
    """

    def __call__(self, models):
        return self._build(models)


class JointL1h512(JointNet):
    """One fully-connected layer with 512 units.

    .. code::

        Parameters: 524,000
        Specification: fc[512]
    """

    def __init__(self, nb_layer=1, nb_hidden=512, *args, **kwargs):
        super(JointL1h512, self).__init__(*args, **kwargs)
        self.nb_layer = nb_layer
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        layers = []
        for layer in range(self.nb_layer):
            kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
            layers.append(kl.Dense(self.nb_hidden,
                                   kernel_initializer=self.init,
                                   kernel_regularizer=kernel_regularizer))
            layers.append(kl.Activation('relu'))
            if self.batch_norm:
                layers.append(kl.BatchNormalization())
            layers.append(kl.Dropout(self.dropout))

        return self._build(models, layers)


class JointL2h512(JointL1h512):
    """Two fully-connected layers with 512 units.

    .. code::

        Parameters: 786,000
        Specification: fc[512]_fc[512]
    """

    def __init__(self, *args, **kwargs):
        super(JointL2h512, self).__init__(*args, **kwargs)
        self.nb_layer = 2


class JointL3h512(JointL1h512):
    """Three fully-connected layers with 512 units.

    .. code::

        Parameters: 1,000,000
        Specification: fc[512]_fc[512]_fc[512]
    """

    def __init__(self, *args, **kwargs):
        super(JointL3h512, self).__init__(*args, **kwargs)
        self.nb_layer = 3


def list_models():
    """Return the name of models in the module."""

    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and name.lower().find('model') == -1:
            models[name] = value
    return models


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
