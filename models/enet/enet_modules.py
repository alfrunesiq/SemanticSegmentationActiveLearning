import tensorflow as tf

from tensorflow.keras.layers    import Layer
from tensorflow.python.training import moving_averages

from ..util import extra_ops as xops

class Initial(Layer):

    def __init__(self, output_channels,
                 kernel_size=(3,3),
                 strides=(2,2),
                 pool_op=tf.nn.max_pool,
                 pool_size=(2,2),
                 padding="SAME",
                 dilation_rate=(1,1),
                 kernel_initializer=tf.initializers.glorot_uniform(),
                 alpha_initializer=tf.initializers.constant(0.25),
                 trainable=True,
                 kernel_regularizer=None,
                 alpha_regularizer=None,
                 batch_norm_momentum=0.90,
                 name="Initial",
                 **kwargs):
        """
        ENet initial block:
                     +-------+
                     | Input |
                     +-------+
                        / \
                      /     \
                    /         \
        +--------------+  +----------------+
        | 3x3 conv x13 |  | 2x2/s2 MaxPool |
        +--------------+  +----------------+
                    \         /
                      \     /
                        \ /
                  +-------------+
                  |    Concat   |
                  | ->BatchNorm |
                  | ->PReLU     |
                  +-------------+
        :param filters:             Number of filters in conv branch.
        :param kernel_size:         Kernel size for the convolution operation.
        :param strides:             Downsampling stride for conv and pool op.
        :param pool_op:             Either tf.nn.max_pool or tf.nn.avg_pool.
        :param padding:             Either "VALID" or "SAME".
        :param dilation_rate:       Optional dilation rate for conv op.
        :param kernel_initializer:  Optional initializer.
        :param kernel_regularizer:  Optional regularizer.
        :param batch_norm_momentum: Batch norm moving average momentum.
        """

        super(Initial, self).__init__(name=name)

        self.output_channels = output_channels
        self.kernel_size     = kernel_size
        self.strides         = strides
        self.pool_op         = pool_op
        self.pool_size       = pool_size
        self.padding         = padding
        self.dilation_rate   = dilation_rate

        self.kernel_initializer  = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer  = tf.keras.regularizers.get(kernel_regularizer)
        self.alpha_initializer   = tf.keras.initializers.get(alpha_initializer)
        self.alpha_regularizer   = tf.keras.regularizers.get(alpha_regularizer)
        self.batch_norm_momentum = batch_norm_momentum

    def build(self, input_shape):
        """
        :param input_shape: input shape NHWC
        """
        if self.built:
            return

        with tf.name_scope("Convolution") as scope:
            self._conv_scope = scope
        with tf.name_scope("Residual") as scope:
            self._res_scope = scope
        # Compute shapes
        input_channels  = int(input_shape[-1])
        filters = self.output_channels - input_channels
        kernel_shape    = self.kernel_size + (input_channels, filters)
        # Create weights
        with tf.variable_scope(self._conv_scope, reuse=tf.AUTO_REUSE):
            self.kernel = self.add_weight(
                name="Kernel",
                dtype=tf.float32,
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
        # Batch normalization parameters
        with tf.variable_scope(self._conv_scope+"BatchNorm/",
                               reuse=tf.AUTO_REUSE):
            # NOTE: moving average statistics during training
            self.mean = self.add_weight(
                name="Mean",
                shape=[self.output_channels],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False
               )
            self.variance = self.add_weight(
                name="Variance",
                shape=[self.output_channels],
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
                trainable=False
               )
            # Trainable batch_norm parameters
            self.gamma = self.add_weight(
                name="Gamma",
                shape=[self.output_channels],
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
                trainable=True
               )
            self.beta = self.add_weight(
                name="Beta",
                shape=[self.output_channels],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=True
               )

        with tf.variable_scope(self._res_scope, reuse=tf.AUTO_REUSE):
            self.alpha = self.add_weight(
                name="Alpha",
                shape=[self.output_channels],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                trainable=True
            )
        self.built = True

    def call(self, inputs, training, **kwargs):
        """
        :param inputs: input tensor
        :returns: output tensor
        :rtype:   tf.Tensor
        """
        dilations = self.dilation_rate if len(self.dilation_rate) == 4 \
            else (1, *self.dilation_rate, 1)
        strides   = self.strides if len(self.strides) == 4 \
            else (1, *self.strides, 1)
        ksize = self.pool_size if len(self.pool_size) == 4 \
            else (1, *self.pool_size, 1)
        output = None

        with tf.name_scope(self._conv_scope):
            conv_out = tf.nn.conv2d(inputs, self.kernel,
                                    strides=strides,
                                    dilations=dilations,
                                    padding=self.padding,
                                    name="Conv2D")

        with tf.name_scope(self._res_scope):
            pool_out = self.pool_op(inputs, ksize, strides, self.padding,
                                    name="MaxPool")
            output = tf.concat([conv_out, pool_out],
                               axis=-1, name="Concat")
            with tf.name_scope(self._res_scope+"BatchNorm/"):
                output, update_mean, update_var = xops.batch_norm(
                    output, mean=self.mean, var=self.variance,
                    gamma=self.gamma, beta=self.beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            output = xops.prelu(output, self.alpha)
        return output

class Bottleneck(Layer):

    def __init__(self,
                 output_channels,
                 kernel_size=(3,3),
                 asymmetric=False,
                 padding="SAME",
                 projection_rate=4,
                 dilation_rate=(1,1),
                 kernel_initializer=tf.initializers.glorot_uniform(),
                 kernel_regularizer=None,
                 alpha_initializer = tf.initializers.constant(0.25),
                 trainable=True,
                 drop_rate=0.1,
                 batch_norm_momentum=0.90,
                 name="Bottleneck",
                 **kwargs):
        """
        Implements the plain bottleneck module in ENet, including
        possibility of dilated convolution and asymmetric (spatially
        separable) convolution.
                     +-------+
                     | Input |
                     +-------+
                         ||
            +------------++----------+
            |                        |
            |           +-------------------------+
            |           |         1x1 conv        |
            |           |  x(input_ch/proj_rate)  |
            |           |  -> BatchNorm -> PReLU  |
            |           +-------------------------+
            |                        | Projection
            |                        V
            |           +-------------------------+
            |           |         3x3 conv        |
            |           |  x(input_ch/proj_rate)  |
            |           |  -> BatchNorm -> PReLU  |
            |           +-------------------------+
            |                        | Convolution
            |                        V
            |           +-------------------------+
            |           |        1x1 conv         |
            |           |      x(input_ch)        |
            |           |     -> BatchNorm        |
            |           +-------------------------+
            |                        | Expansion
            +------------++----------+
                         \/ Residual connection
                   +-----------+
                   |    Add    |
                   | -> PReLU  |
                   +-----------+

        :param kernel_size:         Kernel size for the convolution operation.
        :param asymmetric:          Whether to use spatially separable conv.
        :param projection_rate:     How much to compress the channel dimension.
                                    in projection stage.
        :param padding:             Either "VALID" or "SAME".
        :param dilation_rate:       Optional dilation rate for conv op.
        :param kernel_initializer:  Optional kernel initializer.
        :param alpha_initializer:   Optional PReLU initializer
        :param kernel_regularizer:  Optional regularizer.
        :param drop_rate:           Optional dropout rate.
        :param batch_norm_momentum: BatchNorm moving average momentum.
        :param name:                Scope name for the block
        """

        super(Bottleneck, self).__init__(name=name, dtype=tf.float32)

        self.output_channels = output_channels
        self.kernel_size     = kernel_size
        self.asymmetric      = asymmetric
        self.projection_rate = projection_rate
        self.padding         = padding
        self.dilation_rate   = dilation_rate

        self.alpha_initializer   = tf.keras.initializers.get(alpha_initializer)
        self.kernel_initializer  = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer  = tf.keras.regularizers.get(kernel_regularizer)
        self.drop_rate           = drop_rate
        self.batch_norm_momentum = batch_norm_momentum

    def build(self, input_shape):
        """
        :param input_shape: input shape NHWC
        """
        if self.built:
            return

        # Save absolute name_scopes
        with tf.name_scope("Projection") as scope:
            self._proj_scope = scope
        with tf.name_scope("Convolution") as scope:
            self._conv_scope = scope
        with tf.name_scope("Expansion") as scope:
            self._exp_scope = scope
        with tf.name_scope("Residual") as scope:
            self._res_scope = scope

        # Compute shapes
        input_channels = int(input_shape[-1])
        conv_filters = input_channels // self.projection_rate

        proj_shape = (1, 1, input_channels, conv_filters)
        conv_shape = None
        if self.asymmetric:
            conv_shape = [
                [self.kernel_size[0], 1, conv_filters, conv_filters],
                [1, self.kernel_size[1], conv_filters, conv_filters]
            ]
        else:
            conv_shape = self.kernel_size + (conv_filters, conv_filters)
        exp_shape = (1, 1, conv_filters, self.output_channels)
        # batch norm parameters)
        with tf.variable_scope(self._proj_scope, reuse=tf.AUTO_REUSE):
            self.proj_kernel = self.add_weight(
                name="Kernel",
                shape=proj_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=self.trainable
            )
            # PReLU alpha
            self.proj_alpha = self.add_weight(
                name="Alpha",
                shape=[proj_shape[-1]],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                regularizer=self.kernel_regularizer,
                trainable=self.trainable
            )
            with tf.variable_scope(self._proj_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.proj_mean = self.add_weight(
                    name="Mean",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.proj_variance = self.add_weight(
                    name="Variance",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.proj_gamma = self.add_weight(
                    name="Gamma",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=self.trainable
                )
                self.proj_beta = self.add_weight(
                    name="Beta",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=self.trainable
                )
        with tf.variable_scope(self._conv_scope, reuse=tf.AUTO_REUSE):
            if self.asymmetric:
                self.conv_kernel = [
                    self.add_weight(
                        name="KernelCol",
                        shape=conv_shape[0],
                        dtype=tf.float32,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        trainable=self.trainable
                    ),
                    self.add_weight(
                        name="KernelRow",
                        shape=conv_shape[1],
                        dtype=tf.float32,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        trainable=self.trainable
                    )
                ]
            else: # regular conv
                self.conv_kernel = self.add_weight(
                    name="Kernel",
                    shape=conv_shape,
                    dtype=tf.float32,
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    trainable=self.trainable
                )
            # PReLU alpha
            self.conv_alpha = self.add_weight(
                name="Alpha",
                shape=[conv_filters],
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=self.trainable
            )
            with tf.variable_scope(self._conv_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.conv_mean = self.add_weight(
                    name="Mean",
                    shape=[conv_filters],
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.conv_variance = self.add_weight(
                    name="Variance",
                    shape=[conv_filters],
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.conv_gamma = self.add_weight(
                    name="Gamma",
                    shape=[conv_filters],
                    initializer=tf.initializers.ones(),
                    trainable=self.trainable
                )
                self.conv_beta = self.add_weight(
                    name="Beta",
                    shape=[conv_filters],
                    initializer=tf.initializers.zeros(),
                    trainable=self.trainable
                )

        with tf.variable_scope(self._exp_scope, reuse=tf.AUTO_REUSE):
            self.exp_kernel = self.add_weight(
                name="Kernel",
                shape=exp_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=self.trainable
            )
            with tf.variable_scope(self._exp_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.exp_mean = self.add_weight(
                    name="Mean",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.exp_variance = self.add_weight(
                    name="Variance",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.exp_gamma = self.add_weight(
                    name="Gamma",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=self.trainable
                )
                self.exp_beta = self.add_weight(
                    name="Beta",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=self.trainable
                )

        with tf.variable_scope(self._res_scope, reuse=tf.AUTO_REUSE):
            self.residual_alpha = self.add_weight(
                name="Alpha",
                shape=[exp_shape[-1]],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                regularizer=self.kernel_regularizer,
                trainable=self.trainable
            )
        self.built = True

    def call(self, inputs, training, **kwargs):
        """
        :param inputs: input tensor
        :returns: output tensor
        :rtype:   tf.Tensor
        """
        # Convert parameters to legal shapes
        dilations = self.dilation_rate if len(self.dilation_rate) == 4 \
            else (1, *self.dilation_rate, 1)

        # Main branch:
        with tf.name_scope(self._proj_scope):
            conv_out = tf.nn.conv2d(inputs, self.proj_kernel,
                                    strides=[1,1,1,1],
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._proj_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.proj_mean,
                    self.proj_variance, self.proj_gamma,
                    self.proj_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            conv_out = xops.prelu(conv_out, self.proj_alpha)

        with tf.name_scope(self._conv_scope):
            if self.asymmetric:
                conv_out = tf.nn.conv2d(conv_out, self.conv_kernel[0],
                                        strides=[1,1,1,1],
                                        dilations=dilations,
                                        padding=self.padding,
                                        name="Conv1DRow")
                conv_out = tf.nn.conv2d(conv_out, self.conv_kernel[1],
                                        strides=[1,1,1,1],
                                        dilations=dilations,
                                        padding=self.padding,
                                        name="Conv1DCol")
            else:
                conv_out = tf.nn.conv2d(conv_out, self.conv_kernel,
                                        strides=[1,1,1,1],
                                        dilations=dilations,
                                        padding=self.padding,
                                        name="Conv2D")
            with tf.name_scope(self._conv_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.conv_mean,
                    self.conv_variance, self.conv_gamma,
                    self.conv_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            conv_out = xops.prelu(conv_out, self.conv_alpha)

        with tf.name_scope(self._exp_scope):
            conv_out = tf.nn.conv2d(conv_out, self.exp_kernel,
                                    strides=[1,1,1,1],
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._exp_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.exp_mean,
                    self.exp_variance, self.exp_gamma,
                    self.exp_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
                if self.drop_rate > 0.0:
                    conv_out = xops.spatial_dropout(conv_out, self.drop_rate)

        with tf.name_scope(self._res_scope):
            output = tf.math.add(conv_out, inputs, name="Residual")
            output = xops.prelu(output, self.residual_alpha)
        return output

class BottleneckDownsample(Layer):

    def __init__(self,
                 output_channels,
                 kernel_size=(3,3),
                 padding="SAME",
                 projection_rate=4,
                 dilation_rate=(1,1),
                 kernel_initializer=tf.initializers.glorot_uniform(),
                 kernel_regularizer=None,
                 alpha_initializer = tf.initializers.constant(0.25),
                 drop_rate=0.1,
                 batch_norm_momentum=0.90,
                 name="BottleneckDownsample",
                 **kwargs):
        """
        Implements the plain bottleneck module with downsampling in ENet,
        including possibility of dilated convolution and asymmetric
        (spatially separable) convolution.
                     +-------+
                     | Input |
                     +-------+
                         ||
            +------------++----------+
            |                        V
            |           +-------------------------+
            |           |       2x2/s2 conv       |
            |           |  x(input_ch/proj_rate)  |
            |           |  -> BatchNorm -> PReLU  |
            V           +-------------------------+
    +----------------+               | Projection
    | 2x2/s2 MaxPool |               V
    +----------------+  +-------------------------+
            |           |         3x3 conv        |
            |           |  x(input_ch/proj_rate)  |
            V           |  -> BatchNorm -> PReLU  |
    +--------------+    +-------------------------+
    | Zero padding |                 | Convolution
    +--------------+                 V
            |           +-------------------------+
            |           |        1x1 conv         |
            |           |      x(2*input_ch)      |
            |           |     -> BatchNorm        |
            |           +-------------------------+
            |                        | Expansion
            +------------++----------+
                         \/ Residual connection
                   +-----------+
                   |    Add    |
                   | -> PReLU  |
                   +-----------+

        :param kernel_size:         Kernel size for the convolution operation.
        :param projection_rate:     How much to compress the channel dimension.
                                    in projection stage.
        :param padding:             Either "VALID" or "SAME".
        :param dilation_rate:       Optional dilation rate for conv op.
        :param kernel_initializer:  Optional kernel initializer.
        :param alpha_initializer:   Optional PReLU initializer
        :param kernel_regularizer:  Optional regularizer.
        :param drop_rate:           Optional dropout rate.
        :param batch_norm_momentum: BatchNorm moving average momentum.
        :param name:                Scope name for the block
        """

        super(BottleneckDownsample, self).__init__(name=name,
                                                        dtype=tf.float32)

        self.output_channels = output_channels
        self.kernel_size     = kernel_size
        self.projection_rate = projection_rate
        self.padding         = padding
        self.dilation_rate   = dilation_rate

        self.alpha_initializer   = tf.keras.initializers.get(alpha_initializer)
        self.kernel_initializer  = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer  = tf.keras.regularizers.get(kernel_regularizer)
        self.drop_rate           = drop_rate
        self.batch_norm_momentum = batch_norm_momentum

    def build(self, input_shape):
        """
        :param input_shape: input shape NHWC
        """
        if self.built:
            return
        # Save absolute name_scopes
        with tf.name_scope("Projection") as scope:
            self._proj_scope = scope
        with tf.name_scope("Convolution") as scope:
            self._conv_scope = scope
        with tf.name_scope("Expansion") as scope:
            self._exp_scope = scope
        with tf.name_scope("Residual") as scope:
            self._res_scope = scope

        # Compute shapes
        input_channels  = int(input_shape[-1])
        # NOTE: Compensating for downsampling by doubling filter bank
        conv_filters = 2*(input_channels // self.projection_rate)
        self.zero_padding = [[0,0],[0,0],
                             [0,0],[0,self.output_channels-input_channels]]

        proj_shape = (2, 2, input_channels, conv_filters)
        conv_shape = self.kernel_size + (conv_filters, conv_filters)
        exp_shape  = (1, 1, conv_filters, self.output_channels)
        # Create weights for each sub-layer (conv kernel, PReLU weights and
        # batch norm parameters)
        with tf.variable_scope(self._proj_scope, reuse=tf.AUTO_REUSE):
            self.proj_kernel = self.add_weight(
                name="Kernel",
                shape=proj_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            # PReLU alpha
            self.proj_alpha = self.add_weight(
                name="Alpha",
                shape=[proj_shape[-1]],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            with tf.variable_scope(self._proj_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.proj_mean = self.add_weight(
                    name="Mean",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.proj_variance = self.add_weight(
                    name="Variance",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.proj_gamma = self.add_weight(
                    name="Gamma",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=True
                )
                self.proj_beta = self.add_weight(
                    name="Beta",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )
        with tf.variable_scope(self._conv_scope, reuse=tf.AUTO_REUSE):
            self.conv_kernel = self.add_weight(
                name="Kernel",
                shape=conv_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            # PReLU alpha
            self.conv_alpha = self.add_weight(
                name="Alpha",
                shape=[conv_shape[-1]],
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            with tf.variable_scope(self._conv_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.conv_mean = self.add_weight(
                    name="Mean",
                    shape=[conv_shape[-1]],
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.conv_variance = self.add_weight(
                    name="Variance",
                    shape=[conv_shape[-1]],
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.conv_gamma = self.add_weight(
                    name="Gamma",
                    shape=[conv_shape[-1]],
                    initializer=tf.initializers.ones(),
                    trainable=True
                )
                self.conv_beta = self.add_weight(
                    name="Beta",
                    shape=[conv_shape[-1]],
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )

        with tf.variable_scope(self._exp_scope, reuse=tf.AUTO_REUSE):
            self.exp_kernel = self.add_weight(
                name="Kernel",
                shape=exp_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            with tf.variable_scope(self._exp_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.exp_mean = self.add_weight(
                    name="Mean",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.exp_variance = self.add_weight(
                    name="Variance",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.exp_gamma = self.add_weight(
                    name="Gamma",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=True
                )
                self.exp_beta = self.add_weight(
                    name="Beta",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )

        with tf.variable_scope(self._res_scope, reuse=tf.AUTO_REUSE):
            self.residual_alpha = self.add_weight(
                name="Alpha",
                shape=[exp_shape[-1]],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
        self.built = True

    def call(self, inputs, training, **kwargs):
        """
        :param inputs: input tensor
        :returns: output tensor
        :rtype:   tf.Tensor
        """
        # Convert parameters to legal shapes
        dilations = self.dilation_rate if len(self.dilation_rate) == 4 \
            else (1, *self.dilation_rate, 1)

        # Main branch:
        with tf.name_scope(self._proj_scope):
            conv_out = tf.nn.conv2d(inputs, self.proj_kernel,
                                    strides=[1,2,2,1],
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._proj_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.proj_mean,
                    self.proj_variance, self.proj_gamma,
                    self.proj_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            conv_out = xops.prelu(conv_out, self.proj_alpha)

        with tf.name_scope(self._conv_scope):
            conv_out = tf.nn.conv2d(conv_out, self.conv_kernel,
                                    strides=[1,1,1,1],
                                    dilations=dilations,
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._conv_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.conv_mean,
                    self.conv_variance, self.conv_gamma,
                    self.conv_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            conv_out = xops.prelu(conv_out, self.conv_alpha)

        with tf.name_scope(self._exp_scope):
            conv_out = tf.nn.conv2d(conv_out, self.exp_kernel,
                                    strides=[1,1,1,1],
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._exp_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.exp_mean,
                    self.exp_variance, self.exp_gamma,
                    self.exp_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
                if self.drop_rate > 0.0:
                    conv_out = xops.spatial_dropout(conv_out, self.drop_rate)

        # Residual branch
        res_out, max_pool_argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1,2,2,1], strides=[1,2,2,1],
            Targmax=tf.int64, padding=self.padding, name="MaxPool")
        # tf.tile() ?
        res_out = tf.pad(res_out,
                         paddings=self.zero_padding,
                         name="ZeroPad")
        # Residual connection
        with tf.name_scope(self._res_scope):
            output = tf.math.add(conv_out, res_out, name="Residual")
            output = xops.prelu(output, self.residual_alpha)
        return output, max_pool_argmax

class BottleneckUpsample(Layer):

    def __init__(self,
                 output_channels,
                 kernel_size=(3,3),
                 padding="SAME",
                 projection_rate=4,
                 dilation_rate=(1,1),
                 kernel_initializer=tf.initializers.glorot_uniform(),
                 kernel_regularizer=None,
                 alpha_initializer = tf.initializers.constant(0.25),
                 drop_rate=0.1,
                 batch_norm_momentum=0.90,
                 name="BottleneckUpsample",
                 **kwargs):
        """
        Implements the plain bottleneck module with upsample in ENet,
        including possibility of dilated convolution and asymmetric
        (spatially separable) convolution.
                     +-------+
                     | Input |
                     +-------+
                         ||
            +------------++----------+
            |                        |
            |           +-------------------------+
            |           |        1x1 conv         |
            |           |  x(input_ch/proj_rate)  |
            |           |  -> BatchNorm -> PReLU  |
            V           +-------------------------+
    +----------------+               | Projection
    |    1x1 conv    |               V
    | x(input_ch/2)  |  +-------------------------+
    +----------------+  |  3x3/s2 conv_transpose  |
            |           |  x(input_ch/proj_rate)  |
            V           |  -> BatchNorm -> PReLU  |
    +----------------+  +-------------------------+
    | 2x2 max_unpool |               | Convolution
    +----------------+               V
            |           +-------------------------+
            |           |        1x1 conv         |
            |           |      x(input_ch/2)      |
            |           |     -> BatchNorm        |
            |           +-------------------------+
            |                        | Expansion
            +------------++----------+
                         \/ Residual connection
                   +-----------+
                   |    Add    |
                   | -> PReLU  |
                   +-----------+

        :param kernel_size:         Kernel size for the convolution operation.
        :param projection_rate:     How much to compress the channel dimension.
                                    in projection stage.
        :param padding:             Either "VALID" or "SAME".
        :param dilation_rate:       Optional dilation rate for conv op.
        :param kernel_initializer:  Optional kernel initializer.
        :param alpha_initializer:   Optional PReLU initializer
        :param kernel_regularizer:  Optional regularizer.
        :param drop_rate:           Optional dropout rate.
        :param batch_norm_momentum: BatchNorm moving average momentum.
        :param name:                Scope name for the block
        """

        super(BottleneckUpsample, self).__init__(name=name, dtype=tf.float32)

        self.output_channels = output_channels
        self.kernel_size     = kernel_size
        self.projection_rate = projection_rate
        self.padding         = padding
        self.dilation_rate   = dilation_rate

        self.alpha_initializer   = tf.keras.initializers.get(alpha_initializer)
        self.kernel_initializer  = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer  = tf.keras.regularizers.get(kernel_regularizer)
        self.drop_rate           = drop_rate
        self.batch_norm_momentum = batch_norm_momentum

    def build(self, input_shape):
        """
        :param input_shape: input shape NHWC
        """
        if self.built:
            return
        # Save absolute name_scopes
        with tf.name_scope("Projection") as scope:
            self._proj_scope = scope
        with tf.name_scope("Convolution") as scope:
            self._conv_scope = scope
        with tf.name_scope("Expansion") as scope:
            self._exp_scope = scope
        with tf.name_scope("Residual") as scope:
            self._res_scope = scope

        # Compute shapes
        input_channels  = int(input_shape[-1])
        proj_filters    = input_channels // self.projection_rate
        conv_filters    = proj_filters // 2

        proj_shape = (1, 1, input_channels, proj_filters)
        conv_shape = self.kernel_size + (conv_filters, proj_filters) \
                     if len(self.kernel_size) == 2 else self.kernel_size
        exp_shape  = (1, 1, conv_filters, self.output_channels)
        res_shape  = (1, 1, input_channels, self.output_channels)
        # Create weights for each sub-layer (conv kernel, PReLU weights and
        # batch norm parameters)
        with tf.variable_scope(self._proj_scope, reuse=tf.AUTO_REUSE):
            self.proj_kernel = self.add_weight(
                name="Kernel",
                shape=proj_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            # PReLU alpha
            self.proj_alpha = self.add_weight(
                name="Alpha",
                shape=[proj_shape[-1]],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            with tf.variable_scope(self._proj_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.proj_mean = self.add_weight(
                    name="Mean",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.proj_variance = self.add_weight(
                    name="Variance",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.proj_gamma = self.add_weight(
                    name="Gamma",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=True
                )
                self.proj_beta = self.add_weight(
                    name="Beta",
                    shape=[proj_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )
        with tf.variable_scope(self._conv_scope, reuse=tf.AUTO_REUSE):
            self.conv_kernel = self.add_weight(
                name="Kernel",
                shape=conv_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            # PReLU alpha
            self.conv_alpha = self.add_weight(
                name="Alpha",
                shape=[conv_filters],
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            with tf.variable_scope(self._conv_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.conv_mean = self.add_weight(
                    name="Mean",
                    shape=[conv_filters],
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.conv_variance = self.add_weight(
                    name="Variance",
                    shape=[conv_filters],
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.conv_gamma = self.add_weight(
                    name="Gamma",
                    shape=[conv_filters],
                    initializer=tf.initializers.ones(),
                    trainable=True
                )
                self.conv_beta = self.add_weight(
                    name="Beta",
                    shape=[conv_filters],
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )

        with tf.variable_scope(self._exp_scope, reuse=tf.AUTO_REUSE):
            self.exp_kernel = self.add_weight(
                name="Kernel",
                shape=exp_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            with tf.variable_scope(self._exp_scope+"BatchNorm/",
                                   reuse=tf.AUTO_REUSE):
                self.exp_mean = self.add_weight(
                    name="Mean",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=False
                )
                self.exp_variance = self.add_weight(
                    name="Variance",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=False
                )
                self.exp_gamma = self.add_weight(
                    name="Gamma",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.ones(),
                    trainable=True
                )
                self.exp_beta = self.add_weight(
                    name="Beta",
                    shape=[exp_shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )
        with tf.variable_scope(self._res_scope, reuse=tf.AUTO_REUSE):
            self.res_kernel = self.add_weight(
                name="Kernel",
                shape=res_shape,
                dtype=tf.float32,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            self.residual_alpha = self.add_weight(
                name="Alpha",
                shape=[exp_shape[-1]],
                dtype=tf.float32,
                initializer=self.alpha_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
        self.built = True

    def call(self, inputs, unpool_argmax, training, **kwargs):
        """
        :param inputs: input tensor
        :returns: output tensor
        :rtype:   tf.Tensor
        """
        # Convert parameters to legal shapes
        dilations = self.dilation_rate if len(self.dilation_rate) == 4 \
                    else (1, *self.dilation_rate, 1)
        input_shape = inputs.shape.as_list()
        batch_size  = input_shape[0] if input_shape[0] is not None \
                      else tf.shape(inputs)[0]
        conv_filters = (input_shape[-1] // 2) // self.projection_rate
        conv_out_shape = tf.stack([batch_size, 2*input_shape[1],
                                   2*input_shape[2], conv_filters],
                                  name="ConvTsposeOutShape")

        # Main branch:
        with tf.name_scope(self._proj_scope):
            conv_out = tf.nn.conv2d(inputs, self.proj_kernel,
                                    strides=[1,1,1,1],
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._proj_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.proj_mean,
                    self.proj_variance, self.proj_gamma,
                    self.proj_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            conv_out = xops.prelu(conv_out, self.proj_alpha)

        with tf.name_scope(self._conv_scope):
            conv_out = tf.nn.conv2d_transpose(conv_out, self.conv_kernel,
                                              output_shape=conv_out_shape,
                                              strides=[1,2,2,1],
                                              padding=self.padding,
                                              name="Conv2DTranspose")
            with tf.name_scope(self._conv_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.conv_mean,
                    self.conv_variance, self.conv_gamma,
                    self.conv_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
            conv_out = xops.prelu(conv_out, self.conv_alpha)

        with tf.name_scope(self._exp_scope):
            conv_out = tf.nn.conv2d(conv_out, self.exp_kernel,
                                    strides=[1,1,1,1],
                                    padding=self.padding,
                                    name="Conv2D")
            with tf.name_scope(self._exp_scope+"BatchNorm/"):
                conv_out, update_mean, update_var = xops.batch_norm(
                    conv_out, self.exp_mean,
                    self.exp_variance, self.exp_gamma,
                    self.exp_beta, training=training,
                    decay=self.batch_norm_momentum)
            if training:
                self.add_update([update_mean, update_var])
                if self.drop_rate > 0.0:
                    conv_out = xops.spatial_dropout(conv_out, self.drop_rate)

        # Residual connection
        with tf.name_scope(self._res_scope):
            # Residual branch
            res_out = tf.nn.conv2d(inputs, self.res_kernel,
                                   strides=[1,1,1,1],
                                   padding="SAME", name="Conv2D")
            res_out = xops.unpool_2d(res_out, unpool_argmax,
                                     strides=[1,2,2,1])
            output = tf.math.add(conv_out, res_out, name="Residual")
            output = xops.prelu(output, self.residual_alpha)
        return output

class Final(Layer):

    def __init__(self, classes,
                 kernel_size=(3,3),
                 padding="SAME",
                 dilation_rate=(1,1),
                 kernel_initializer=tf.initializers.glorot_uniform(),
                 kernel_regularizer=None,
                 name="Final",
                 **kwargs):
        """
        ENet final block (transposed convolution).
        :param classes:             Number of output classes (channels).
        :param kernel_size:         Kernel size for the convolution operation.
        :param padding:             Either "VALID" or "SAME".
        :param dilation_rate:       Optional dilation rate for conv op.
        :param kernel_initializer:  Optional initializer.
        :param kernel_regularizer:  Optional regularizer.
        """

        super(Final, self).__init__(name=name)

        self.classes       = classes
        self.kernel_size   = kernel_size
        self.padding       = padding
        self.dilation_rate = dilation_rate

        self.kernel_initializer  = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer  = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        """
        :param input_shape: input shape NHWC
        """
        if self.built:
            return

        with tf.name_scope("ConvTransposed") as scope:
            self._conv_scope = scope
        # Compute shapes
        input_channels = int(input_shape[-1])
        kernel_shape = self.kernel_size + (self.classes, input_channels)
        # Create weights
        with tf.variable_scope(self._conv_scope, reuse=tf.AUTO_REUSE):
            self.kernel = self.add_weight(
                name="Kernel",
                dtype=tf.float32,
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: input tensor
        :returns: output tensor
        :rtype:   tf.Tensor
        """
        output = None
        with tf.name_scope(self._conv_scope):
            dilations = self.dilation_rate if len(self.dilation_rate) == 4 \
                else (1, *self.dilation_rate, 1)
            input_shape = inputs.shape.as_list()
            conv_out_shape = None
            with tf.name_scope("ShapeOps"):
                batch_size  = input_shape[0] if input_shape[0] is not None \
                    else tf.shape(inputs)[0]
                conv_out_shape = tf.stack([batch_size, 2*input_shape[1],
                                           2*input_shape[2], self.classes])
            output = tf.nn.conv2d_transpose(inputs, self.kernel,
                                            output_shape=conv_out_shape,
                                            strides=[1,2,2,1],
                                            padding=self.padding,
                                            name="Conv2DTranspose")
        return output
