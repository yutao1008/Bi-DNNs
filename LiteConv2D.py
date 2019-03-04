import math
from keras.layers.convolutional import _Conv
from keras.engine import InputSpec
from keras import backend as K
from utils import find_bilinear_dimensions
from keras.utils import conv_utils

if K.backend() != 'tensorflow':
    raise ValueError('Currently this model is built on Tensorflow')
else:
    import tensorflow as tf

class BiConv2D(_Conv):
    """2D convolution layer with the bilinear projection (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: A tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
        strides: A tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
        padding: one of `"valid"` or `"same"` (case-sensitive).
            Note that `"same"` is slightly inconsistent across backends with
            `strides` != 1, as described
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        scale: The scaling parameter for bilinear projection
    # Input shape
        4D tensor with shape:
        `(samples, rows, cols, channels)`
    # Output shape
        4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 scale=1,
                 **kwargs):
        super(BiConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=(1,1),
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.scale = scale

    def build(self, input_shape):
        channel_axis = -1
        depth = input_shape[channel_axis]
        input_dim = depth*self.kernel_size[0]*self.kernel_size[1]
        self.d1, self.d2 = find_bilinear_dimensions(input_dim)
        self.u1, self.u2 = find_bilinear_dimensions(self.filters)
        self.left_kernel = self.add_weight(shape=(self.d1, self.u1*self.scale),
                                      initializer=self.kernel_initializer,
                                      name='left_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.right_kernel = self.add_weight(shape=(self.d2, self.u2*self.scale),
                                      initializer=self.kernel_initializer,
                                      name='right_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters*self.scale*self.scale,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: depth})
        self.built = True

    def call(self, inputs):
        if self.padding.lower() == 'same':
            padding = 'SAME'
        elif self.padding.lower() == 'valid':
            padding = 'VALID'
        else:
            raise ValueError('Padding must be either \'same\' or \'valid\'.')
        x = tf.extract_image_patches(inputs, [1, self.kernel_size[0], self.kernel_size[1], 1], 
                [1,self.strides[0],self.strides[1],1], [1,self.dilation_rate[0],self.dilation_rate[1],1], padding=padding)
        height, width = K.int_shape(x)[1], K.int_shape(x)[2]
        x = K.reshape(x, (-1, height, width, self.d1, self.d2))
        x = tf.tensordot(x, self.left_kernel, axes=[[3],[0]])      
        x = tf.tensordot(x, self.right_kernel, axes=[[3],[0]])
        outputs = K.reshape(x, (-1, height, width, self.filters*self.scale*self.scale))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters*self.scale*self.scale,)


    def get_config(self):
        config = super(BiConv2D, self).get_config()
        return config


