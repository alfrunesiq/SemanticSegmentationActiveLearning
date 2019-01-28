import tensorflow as tf
import extra_ops as xops

def block_initial(inputs, is_training, \
                  padding="SAME", \
                  output_width=16, \
                  name="initial"):
    """FIXME! briefly describe function
                 +-------+
                 | Input |
                 +-------+
                    / \
                  /     \
                /         \
    +--------------+  +----------------+
    | 3x3 conv x16 |  | 2x2/s2 MaxPool |
    +--------------+  +----------------+
                \         /
                  \     /
                    \ /
                +--------+
                | Concat |
                +--------+

    :param inputs:       Input tensor
    :param padding:      Padding for the conv operation
    :param output_width: Channels of the output tensor
    :param name:         Name of the scope for the block

    :returns: output tensor, trainable variables
    :rtype:   Tensor, list
    """
    params = {}
    with tf.variable_scope(name):
        with tf.name_scope("ShapeOps"):
            # shape(inputs)=[N,H,W,C]
            # Get input channel count
            input_ch = inputs.shape[-1]
            # output width is concatenation of max pool and conv
            conv_width = output_width - input_ch
            conv_kern_shape = [3,3,input_ch,conv_width]
        # Get conv. kernel
        kern     = tf.get_variable(name="Kernel", \
                                   shape=conv_kern_shape, \
                                   initializer=tf.glorot_uniform_initializer(), \
                                   trainable=True)
        out_conv = tf.nn.conv2d(inputs, kern, \
                                strides=[1,2,2,1], \
                                padding=padding, \
                                name="Conv2D")
        out_mp   = tf.nn.max_pool(inputs, \
                                  ksize=[1,2,2,1], \
                                  strides=[1,2,2,1], \
                                  padding=padding, \
                                  name="MaxPool")
        out      = tf.concat([out_conv, out_mp], \
                              axis=3, \
                              name="Concat")
    params["Kernel"] = kern
    return out, params

def block_bottleneck(inputs, \
                      padding="SAME", \
                      projection_rate=4, \
                      dilations=[1,1,1,1], \
                      is_training=True, \
                      bn_decay=0.90, \
                      asymmetric=False, \
                      kernel_initializer=tf.initializers.glorot_uniform(), \
                      alpha_initializer=tf.initializers.constant(0.25), \
                      name="Bottleneck"):
    """
    Implements the plain bottleneck module in ENet, including possibility
    of dilated convolution and asymmetric (spatially separable) convolution.
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

    :param inputs:
    :param padding:
    :param projection_rate:
    :param dilations:
    :param name:
    :returns:
    :rtype:

    """
    variables = {}   # Dict with Variables in the block
    out       = None
    with tf.variable_scope(name):

        with tf.name_scope("ShapeOps"):
            # Get input channel count
            input_ch = inputs.shape[-1]
            # Number of filters in the bottleneck are reduced by a factor of
            # @projection_rate
            bneck_filters = input_ch / projection_rate
            # Get conv. kernels' shape
            proj_kern_shape = [1,1,input_ch,bneck_filters]
            if asymmetric:
                conv_kern_shape = [ \
                        [5,1,bneck_filters,bneck_filters], \
                        [1,5,bneck_filters,bneck_filters], \
                ]
            else:
                conv_kern_shape = [3,3,bneck_filters,bneck_filters]
            exp_kern_shape = [1,1,bneck_filters,input_ch]
            # END scope ShapeOps

        ############ Main Branch ############
        with tf.variable_scope("DownProject"):
            # Bottleneck projection operation
            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    shape=[bneck_filters], \
                                    trainable=True)
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=proj_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d(inputs, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                      decay=bn_decay)
            xops.prelu(out, alpha, name="PReLU")
            variables["DownProject"] = {}
            variables["DownProject"]["Kernel"] = kern
            variables["DownProject"]["Alpha"] = alpha
            variables["DownProject"]["BatchNorm"] = bn_params

        with tf.variable_scope("Conv"):
            # Main convolution operation
            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    shape=[bneck_filters], \
                                    trainable=True)
            if asymmetric:

                kern = [ \
                         tf.get_variable(name="KernelCol", \
                                         dtype=tf.float32, \
                                         initializer=kernel_initializer, \
                                         shape=conv_kern_shape[0], \
                                         trainable=True),
                         tf.get_variable(name="KernelRow", \
                                         dtype=tf.float32, \
                                         initializer=kernel_initializer, \
                                         shape=conv_kern_shape[1], \
                                         trainable=True)
                ]
                out = tf.nn.conv2d(out, kern[0], \
                                   strides=[1,1,1,1], \
                                   padding=padding, \
                                   dilations=dilations, \
                                   name="Conv2D")
                out = tf.nn.conv2d(out, kern[1], \
                                   strides=[1,1,1,1], \
                                   padding=padding, \
                                   dilations=dilations, \
                                   name="Conv2D")
            else:
                kern = tf.get_variable(name="Kernel", \
                                       dtype=tf.float32, \
                                       initializer=kernel_initializer, \
                                       shape=conv_kern_shape, \
                                       trainable=True)
                out = tf.nn.conv2d(out, kern, \
                                   strides=[1,1,1,1], \
                                   padding=padding, \
                                   dilations=dilations, \
                                   name="Conv2D")

            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            out = xops.prelu(out, alpha, name="PReLU")

            variables["Conv"] = {}
            variables["Conv"]["Kernel"] = kern
            variables["Conv"]["Alpha"]  = alpha
            variables["Conv"]["BatchNorm"] = bn_params
        # END scope Conv

        with tf.variable_scope("Expansion"):
            # Feature expansion operation
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=exp_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d(out, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            variables["Expansion"] = {}
            variables["Expansion"]["Kernel"] = kern
            variables["Expansion"]["BatchNorm"] = bn_params
            # NOTE: no prelu here
        # TODO: add spatial dropout here if is_training == True
        #####################################

        alpha = tf.get_variable(name="Alpha", \
                                shape=[input_ch], \
                                dtype=tf.float32, \
                                initializer=alpha_initializer, \
                                trainable=True)
        # NOTE: out comes from main branch
        out = tf.add(inputs, out, name="Residual")
        out = xops.prelu(out, alpha, name="PReLU")
        variables["Alpha"] = alpha

        # end tf.variable_scope(name)

    return out, variables
# END def block_bottleneck


def block_bottleneck_upsample(inputs, unpool_argmax, is_training, \
                              padding="SAME", \
                              projection_rate=4, \
                              dilations=[1,1,1,1], \
                              bn_decay=0.90, \
                              kernel_initializer=tf.initializers.glorot_uniform(), \
                              alpha_initializer=tf.initializers.constant(0.25), \
                              name="BottleneckUpsample"):
    """
                 +-------+
                 | Input |
                 +-------+
                     ||
        +------------++----------+
        |                        |
        |           +-------------------------+
        |           |       2x2/s2 conv       |
        |           |  x(input_ch/proj_rate)  |
        |           |  -> BatchNorm -> PReLU  |
        V           +-------------------------+
+----------------+               | Projection
|    1x1 conv    |               V
| x(input_ch/2)  |  +-------------------------+
+----------------+  |         3x3 conv        |
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
    :param inputs:
    :param padding:
    :param projection_rate:
    :param name:
    :returns:
    :rtype:

    """

    variables = {}   # Dict with Variables in the block
    out       = None
    with tf.variable_scope(name):

        with tf.name_scope("ShapeOps"):
            # Get input channel count
            input_shape = tf.shape(inputs, name="InputShape")
            input_ch    = inputs.shape[3]
            batch_sz    = input_shape[0]
            # Number of filters in the bottleneck are reduced by a factor of
            # @projection_rate
            bneck_filters = input_ch / projection_rate
            # Get conv. kernels' shape
            proj_kern_shape = [1,1,input_ch,bneck_filters]
            conv_kern_shape = [3,3,bneck_filters,bneck_filters]
            conv_out_shape  = tf.stack([batch_sz, 2*input_shape[1], \
                                        2*input_shape[2], bneck_filters], \
                                       name="ConvTsposeOutShape")
            # NOTE: upsampling halves the number of output channels following
            #       VGG-philosophy of preserving computational complexity
            exp_kern_shape = [1,1,bneck_filters,input_ch/2]
            # TODO: check if 1x1 of 3x3 is actually used
            res_kern_shape = [1,1,input_ch,input_ch/2]
            # END scope ShapeOps

        ############ Main Branch ############
        with tf.variable_scope("DownProject"):
            # Bottleneck projection operation
            alpha = tf.get_variable(name="Alpha", \
                                    shape=[bneck_filters], \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    trainable=True)
            kern = tf.get_variable(name="Kernel", \
                                   shape=proj_kern_shape, \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   trainable=True)
            out = tf.nn.conv2d(inputs, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            xops.prelu(out, alpha, name="PReLU")

            variables["DownProject"] = {}
            variables["DownProject"]["Kernel"] = kern
            variables["DownProject"]["Alpha"] = alpha
            variables["DownProject"]["BatchNorm"] = bn_params
            # END scope DownProject

        with tf.variable_scope("Conv"):
            # Main convolution operation
            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    shape=[bneck_filters], \
                                    trainable=True)
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=conv_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d_transpose(out, kern, conv_out_shape, \
                                         strides=[1,2,2,1], \
                                         name="Conv2DTranspose")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            out = xops.prelu(out, alpha, name="PReLU")

            variables["Conv"] = {}
            variables["Conv"]["Kernel"] = kern
            variables["Conv"]["Alpha"]  = alpha
            variables["Conv"]["BatchNorm"] = bn_params
        # END scope Conv

        # TODO: did they really use Expansion in upsampling blocks?
        with tf.variable_scope("Expansion"):
            # Feature expansion operation
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=exp_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d(out, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            variables["Expansion"] = {}
            variables["Expansion"]["Kernel"] = kern
            variables["Expansion"]["BatchNorm"] = bn_params
            # NOTE: no prelu here
            # END scope Expansion
        #####################################

        ########## Residual Branch ##########
        #TODO: remove this and replace with upsampling stuff
        kern = tf.get_variable(name="Kernel", \
                               dtype=tf.float32, \
                               initializer=kernel_initializer, \
                               shape=res_kern_shape, \
                               trainable=True)
        res_out = tf.nn.conv2d(inputs, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               name="Conv2D")
        res_out = xops.unpool_2d(res_out, unpool_argmax, \
                                 strides=[1,2,2,1])
        variables["Kernel"] = kern
        #####################################

        alpha = tf.get_variable(name="Alpha", \
                                shape=[input_ch/2], \
                                dtype=tf.float32, \
                                initializer=alpha_initializer, \
                                trainable=True)
        # NOTE: out comes from main branch
        out = tf.add(res_out, out, name="Residual")
        out = xops.prelu(out, alpha, name="PReLU")
        variables["Alpha"] = alpha

    # end with tf.variable_scope(name)

    return out, variables
# END def block_bottleneck

def block_bottleneck_downsample(inputs, is_training, \
                                padding="SAME", \
                                projection_rate=4, \
                                bn_decay=0.90, \
                                dilations=[1,1,1,1], \
                                kernel_initializer=tf.initializers.glorot_uniform(), \
                                alpha_initializer=tf.initializers.constant(0.25), \
                                name="BottleneckDownsample"):
    """
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
    :param inputs:
    :param padding:
    :param projection_rate:
    :param name:
    :returns: operation output, parameters, max pool argmax
    :rtype:   (tf.Tensor, dict, tf.Tensor)
    """
    variables = {}   # Dict with Variables in the block
    out       = None
    with tf.variable_scope(name):
        with tf.name_scope("ShapeOps"):
            # Get input channel count
            input_ch = inputs.shape[-1]
            # Number of filters in the bottleneck are reduced by a factor of
            # @projection_rate
            bneck_filters = input_ch / projection_rate
            # Get conv. kernels' shape
            proj_kern_shape = [2,2,input_ch,bneck_filters]
            conv_kern_shape = [3,3,bneck_filters,bneck_filters]
            # NOTE: downsampling doubles the output channel count following
            #       VGG-philosophy of preserving computational complexity
            exp_kern_shape = [1,1,bneck_filters,2*input_ch]
            zero_padding  = [[0,0],[0,0],[0,0],[0,input_ch]]
            # END scope ShapeOps

        ############ Main Branch ############
        with tf.variable_scope("DownProject"):
            # Bottleneck projection operation
            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    shape=[bneck_filters], \
                                    trainable=True)
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=proj_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d(inputs, kern, \
                               strides=[1,2,2,1], \
                               padding=padding, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            xops.prelu(out, alpha, name="PReLU")

            variables["DownProject"] = {}
            variables["DownProject"]["Kernel"] = kern
            variables["DownProject"]["Alpha"] = alpha
            variables["DownProject"]["BatchNorm"] = bn_params
            # END scope DownProject

        with tf.variable_scope("Conv"):
            # Main convolution operation
            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    shape=[bneck_filters], \
                                    trainable=True)
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=conv_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d(out, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               dilations=dilations, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            out = xops.prelu(out, alpha, name="PReLU")

            variables["Conv"] = {}
            variables["Conv"]["Kernel"] = kern
            variables["Conv"]["Alpha"]  = alpha
            variables["Conv"]["BatchNorm"] = bn_params
        # END scope Conv

        with tf.variable_scope("Expansion"):
            # Feature expansion operation
            kern = tf.get_variable(name="Kernel", \
                                   dtype=tf.float32, \
                                   initializer=kernel_initializer, \
                                   shape=exp_kern_shape, \
                                   trainable=True)
            out = tf.nn.conv2d(out, kern, \
                               strides=[1,1,1,1], \
                               padding=padding, \
                               name="Conv2D")
            out, bn_params = xops.batch_normalization(out, is_training, \
                                                       decay=bn_decay)
            variables["Expansion"] = {}
            variables["Expansion"]["Kernel"] = kern
            variables["Expansion"]["BatchNorm"] = bn_params
            # NOTE: no prelu here
            # END scope Expansion
        #####################################

        ########## Residual Branch ##########
        # MAYBE: add Targmax=tf.int32 below? (can still address 4GB)
        # BUG: max_pool_with_argmax apparently doesn't support
        #      Targmax=tf.int32
        res_out, max_pool_argmax = \
            tf.nn.max_pool_with_argmax(inputs, \
                                       ksize=[1,2,2,1], \
                                       strides=[1,2,2,1], \
                                       Targmax=tf.int64, \
                                       padding=padding, \
                                       name="MaxPool")
        # tf.tile() ?
        res_out = tf.pad(res_out, \
                         paddings=zero_padding, \
                         name="ZeroPad")
        #####################################

        alpha = tf.get_variable(name="Alpha", \
                                shape=[2*input_ch], \
                                dtype=tf.float32, \
                                initializer=alpha_initializer, \
                                trainable=True)
        print(alpha)
        # NOTE: out comes from main branch
        out = tf.add(res_out, out, name="Residual")
        out = xops.prelu(out, alpha, name="PReLU")
        variables["Alpha"] = alpha
    # end with tf.variable_scope(name)

    return out, variables, max_pool_argmax
# END def block_bottleneck
