from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.layers.recurrent import RNN
from utils import find_bilinear_dimensions
if K.backend() != 'tensorflow':
    raise ValueError('Currently this model is built on Tensorflow')
else:
    import tensorflow as tf

def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)

class BiLSTMCell(Layer):
    """Cell class for the LSTM layer (implemented by bilinear projections).
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(BiLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.d1, self.d2 = find_bilinear_dimensions(input_dim)
        self.u1, self.u2 = find_bilinear_dimensions(self.units)

        self.left_kernel = self.add_weight(shape=(self.d1, self.u1 * 4),
                                      name='left_kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.right_kernel = self.add_weight(shape=(self.d2, self.u2 * 4),
                                      name='right_kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.left_recurrent_kernel = self.add_weight(
            shape=(self.u1, self.u1 * 4),
            name='left_recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.right_recurrent_kernel = self.add_weight(
            shape=(self.u2, self.u2 * 4),
            name='right_recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.left_kernel_i = self.left_kernel[:, :self.u1]
        self.left_kernel_f = self.left_kernel[:, self.u1: self.u1 * 2]
        self.left_kernel_c = self.left_kernel[:, self.u1 * 2: self.u1 * 3]
        self.left_kernel_o = self.left_kernel[:, self.u1 * 3:]
        self.right_kernel_i = self.right_kernel[:, :self.u2]
        self.right_kernel_f = self.right_kernel[:, self.u2: self.u2 * 2]
        self.right_kernel_c = self.right_kernel[:, self.u2 * 2: self.u2 * 3]
        self.right_kernel_o = self.right_kernel[:, self.u2 * 3:]

        self.left_recurrent_kernel_i = self.left_recurrent_kernel[:, :self.u1]
        self.left_recurrent_kernel_f = self.left_recurrent_kernel[:, self.u1: self.u1 * 2]
        self.left_recurrent_kernel_c = self.left_recurrent_kernel[:, self.u1 * 2: self.u1 * 3]
        self.left_recurrent_kernel_o = self.left_recurrent_kernel[:, self.u1 * 3:]
        self.right_recurrent_kernel_i = self.right_recurrent_kernel[:, :self.u2]
        self.right_recurrent_kernel_f = self.right_recurrent_kernel[:, self.u2: self.u2 * 2]
        self.right_recurrent_kernel_c = self.right_recurrent_kernel[:, self.u2 * 2: self.u2 * 3]
        self.right_recurrent_kernel_o = self.right_recurrent_kernel[:, self.u2 * 3:]

        if self.use_bias:
            self.bias_i = K.reshape(self.bias[:self.units], (self.u1, self.u2)) 
            self.bias_f = K.reshape(self.bias[self.units: self.units * 2], (self.u1, self.u2))
            self.bias_c = K.reshape(self.bias[self.units * 2: self.units * 3], (self.u1, self.u2))
            self.bias_o = K.reshape(self.bias[self.units * 3:], (self.u1, self.u2))
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state


        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs                
            inputs_c = inputs
            inputs_o = inputs

        x_i = K.reshape(inputs_i, (-1, self.d1, self.d2))
        x_i = tf.tensordot(x_i, self.left_kernel_i, axes=[[1],[0]])
        x_i = tf.tensordot(x_i, self.right_kernel_i, axes=[[1],[0]])

        x_f = K.reshape(inputs_f, (-1, self.d1, self.d2))
        x_f = tf.tensordot(x_f, self.left_kernel_f, axes=[[1],[0]])
        x_f = tf.tensordot(x_f, self.right_kernel_f, axes=[[1],[0]])
        
        x_c = K.reshape(inputs_c, (-1, self.d1, self.d2))
        x_c = tf.tensordot(x_c, self.left_kernel_c, axes=[[1],[0]])
        x_c = tf.tensordot(x_c, self.right_kernel_c, axes=[[1],[0]])

        x_o = K.reshape(inputs_o, (-1, self.d1, self.d2))
        x_o = tf.tensordot(x_o, self.left_kernel_o, axes=[[1],[0]])
        x_o = tf.tensordot(x_o, self.right_kernel_o, axes=[[1],[0]])

        if self.use_bias:
            x_i = tf.add(x_i, self.bias_i)
            x_f = tf.add(x_f, self.bias_f)
            x_c = tf.add(x_c, self.bias_c)
            x_o = tf.add(x_o, self.bias_o)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        h_tm1_i = K.reshape(h_tm1_i, (-1, self.u1, self.u2))
        h_tm1_f = K.reshape(h_tm1_f, (-1, self.u1, self.u2))
        h_tm1_c = K.reshape(h_tm1_c, (-1, self.u1, self.u2))
        h_tm1_o = K.reshape(h_tm1_o, (-1, self.u1, self.u2))
        
        i = tf.tensordot(h_tm1_i, self.left_recurrent_kernel_i, axes=[[1],[0]])
        i = tf.tensordot(i, self.right_recurrent_kernel_i, axes=[[1],[0]])        
        i = self.recurrent_activation(x_i + i)
        f = tf.tensordot(h_tm1_f, self.left_recurrent_kernel_f, axes=[[1],[0]])
        f = tf.tensordot(f, self.right_recurrent_kernel_f, axes=[[1],[0]])
        f = self.recurrent_activation(x_f + f)
        c = tf.tensordot(h_tm1_c, self.left_recurrent_kernel_c, axes=[[1],[0]])
        c = tf.tensordot(c, self.right_recurrent_kernel_c, axes=[[1],[0]])
        c = f * K.reshape(c_tm1, (-1, self.u1, self.u2)) + i * self.activation(x_c + c)
        o = tf.tensordot(h_tm1_o, self.left_recurrent_kernel_o, axes=[[1],[0]])
        o = tf.tensordot(o, self.right_recurrent_kernel_o, axes=[[1],[0]])        
        o = self.recurrent_activation(x_o + o)
        h = o * self.activation(c)
        h = K.reshape(h, (-1, self.u1*self.u2))
        c = K.reshape(c, (-1, self.u1*self.u2))
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(BiLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BiLSTM(RNN):
    """Bilinear projections based Long Short-Term Memory layer.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):

        cell = BiLSTMCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout)
        super(BiLSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(BiLSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)



    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout


    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(BiLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)