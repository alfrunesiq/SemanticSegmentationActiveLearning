#import numpy as np
import tensorflow as tf
import extra_ops as xops

class ENet:
    """
    https://arxiv.org/pdf/1606.02147.pdf
    """
    self._bottleneck_num = 0
    self._stage_num      = 0

    def __init__(self):


    def _block_initial(self, inputs, \
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
        with tf.variable_scope(name):
            with tf.name_scope("ShapeOps"):
                # shape(inputs)=[N,H,W,C]
                # Get input shape
                input_sh = tf.shape(inputs, name="InputShape")
                # Get input channel count
                input_ch = input_sh[-1]
                # output width is concatenation of max pool and conv
                conv_width = output_width - input_ch
            # Get conv. kernel
            kern     = tf.get_variable(name="Kernel", \
                                       initializer=tf.glorot_uniform_initializer(), \
                                       shape=[conv_width,3,3,input_ch], \
                                       trainable=True)
            out_conv = tf.nn.conv2d(inputs, kern, \
                                    strides=[1,2,2,1], \
                                    padding=padding, \
                                    name="Conv2D")
            out_mp   = tf.nn.max_pool(inputs, \
                                      ksize=[1,2,2,1], \
                                      strides=[1,2,2,1], \
                                      name="MaxPool")
            out      = tf.concat([out_conv, out_mp], \
                                  axis=3, \
                                  name="Concat")
        return out, [kern]

    def _batch_normalization(self, inputs, is_training=True, momentum=0.9):
        params = {}
        with tf.variable_scope("BatchNorm"):
            with tf.name_scope("ShapeOps"):
                input_ch = tf.shape(inputs)[-1]
            params["Mean"] = tf.get_variable(shape=[input_ch], \
                                             initializer=tf.zeros_initializer(), \
                                             trainable=False, \
                                             dtype=tf.float32, \
                                             name="Mean")
            params["Variance"] = tf.get_variable(shape=[input_ch], \
                                                 initializer=tf.ones_initializer(), \
                                                 dtype=tf.float32, \
                                                 trainable=False, \
                                                 name="Variance")
            params["Beta"] = tf.get_variable(shape=[input_ch],
                                             initializer=tf.zeros_initializer(), \
                                             trainable=True, \
                                             dtype=tf.float32, \
                                             name="Beta")
            params["Gamma"] = tf.get_variable(shape=[input_ch],
                                             initializer=tf.ones_initializer(), \
                                             trainable=True, \
                                             dtype=tf.float32, \
                                             name="Gamma")
            out, batch_mean, batch_var = tf.nn.fused_batch_norm(inputs, \
                                                                scale=params["Gamma"], \
                                                                offset=params["Beta"], \
                                                                mean=params["Mean"], \
                                                                variance=params["Variance"], \
                                                                is_training=is_training)
            if is_training:
                update_mean = params["Mean"].assign(momentum*params["Mean"] + \
                                                    (1 - momentum)*batch_mean)
                update_var = params["Variance"].assign(momentum*params["Variance"] + \
                                                       (1 - momentum)*batch_mean)
                with tf.control_dependencies([update_mean, update_var]):
                    out = tf.identity(out)

        return out, params

    def _block_bottleneck(self, inputs, input_shape, \
                          padding="SAME", \
                          projection_rate=4, \
                          dilations=[1,1,1,1], \
                          is_training=True, \
                          bn_momentum=0.90, \
                          kernel_initializer=tf.initializers.glorot_uniform(), \
                          alpha_initializer=tf.initializers.constant(0.25), \
                          name="Bottleneck"):
        """
        Implements the plain bottleneck module in ENet, including possibility
        of dilated convolution.
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
        :param input_shape:
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
                input_ch = tf.shape(inputs, name="InputShape")[-1]
                # Number of filters in the bottleneck are reduced by a factor of
                # @projection_rate
                bneck_filters = input_ch / projection_rate
                # Get conv. kernels' shape
                proj_kern_shape = tf.stack([bneck_filters,1,1,input_ch], \
                                           name="DownProjectShape")
                conv_kern_shape  = tf.stack([bneck_filters,3,3,bneck_filters], \
                                       name="ConvShape")
                exp_kern_shape = tf.stack([input_ch,1,1,bneck_filters], \
                                           name="ExpansionShape")
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
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

                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
                variables["Expansion"] = {}
                variables["Expansion"]["Kernel"] = kern
                variables["Expansion"]["BatchNorm"] = bn_params
                # NOTE: no prelu here
            # TODO: add spatial dropout here if is_training == True
            #####################################

            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    variables["Conv"]["BatchNorm"] = bn_params
                                    shape=[bneck_filters], \
                                    trainable=True)
            # NOTE: out comes from main branch
            out = tf.add(inputs, out, name="Residual")
            out = xops.prelu(out, alpha, name="PReLU")
            variables["Alpha"] = alpha

            # end tf.variable_scope(name)

        return out, variables
    # END _block_bottleneck


    def _block_bottleneck_upsample(self, inputs, unpool_argmax, \
                                   padding="SAME", \
                                   projection_rate=4, \
                                   dilations=[1,1,1,1], \
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
        :returns: 
        :rtype: 

        """

        variables = {}   # Dict with Variables in the block
        out       = None
        with tf.variable_scope(name):

            with tf.name_scope("ShapeOps"):
                # Get input channel count
                input_shape = tf.shape(inputs, name="InputShape")
                input_ch    = input_shape[3]
                # Number of filters in the bottleneck are reduced by a factor of
                # @projection_rate
                bneck_filters = input_ch / projection_rate
                # Get conv. kernels' shape
                proj_kern_shape = tf.stack([bneck_filters,1,1,input_ch], \
                                           name="DownProjectShape")
                conv_kern_shape = tf.stack([bneck_filters,3,3,bneck_filters], \
                                           name="ConvShape")
                conv_out_shape  = tf.stack([input_shape[0], 2*input_shape[1], \
                                            2*input_shape[2], bneck_filters], \
                                           name="ConvTsposeOutShape")
                # NOTE: upsampling halves the number of output channels following
                #       VGG-philosophy of preserving computational complexity
                exp_kern_shape = tf.stack([input_ch/2,1,1,bneck_filters], \
                                           name="ExpansionShape")
                # TODO: check if 1x1 of 3x3 is actually used
                res_kern_shape = tf.stack([input_ch/2,1,1,input_ch], \
                                          name="ResidualKernelShape")
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
                variables["Expansion"] = {}
                variables["Expansion"]["Kernel"] = kern
                variables["Expansion"]["BatchNorm"] = bn_params
                # NOTE: no prelu here
                # END scope Expansion
            #####################################

            ########## Residual Branch ##########
            #TODO: remove this and replace with upsampling stuff
            res_out, max_pool_argmax = \
                tf.nn.max_pool_with_argmax(inputs, \
                                           ksize=[1,2,2,1], \
                                           strides=[1,2,2,1], \
                                           padding=padding, \
                                           name="MaxPool")
            # tf.tile() ?
            res_out = tf.pad(res_out, \
                             paddings=zero_padding, \
                             name="ZeroPad")
            #####################################

            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    variables["Conv"]["BatchNorm"] = bn_params
                                    shape=input_ch, \
                                    trainable=True)
            # NOTE: out comes from main branch
            out = tf.add(inputs, out, name="Residual")
            out = xops.prelu(out, alpha, name="PReLU")
            variables["Alpha"] = alpha

        # end with tf.variable_scope(name)

        return out, variables, max_pool_argmax
    # END _block_bottleneck
        return out, variables

    def _block_bottleneck_asymmetric(self, inputs, \
                                     receptive_field=[5,5], \
                                     padding="SAME", \
                                     projection_rate=4, \
                                     name="BottleneckAsymmetric"):

        return out, variables






    def _block_bottleneck_downsample(self, inputs, \
                                     padding="SAME", \
                                     projection_rate=4, \
                                     is_training=True, \
                                     bn_momentum=0.90, \
                                     kernel_initializer=tf.initializers.glorot_uniform(), \
                                     alpha_initializer=tf.initializers.constant(0.25), \
                                     name="BottleneckDownsample"):
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
        :returns: 
        :rtype: 

        """

        #NOTE: need to return max-pool argmax
        variables = {}   # Dict with Variables in the block
        out       = None
        with tf.variable_scope(name):

            with tf.name_scope("ShapeOps"):
                # Get input channel count
                input_ch = tf.shape(inputs, name="InputShape")[3]
                # Number of filters in the bottleneck are reduced by a factor of
                # @projection_rate
                bneck_filters = input_ch / projection_rate
                # Get conv. kernels' shape
                proj_kern_shape = tf.stack([bneck_filters,2,2,input_ch], \
                                           name="DownProjectShape")
                conv_kern_shape  = tf.stack([bneck_filters,3,3,bneck_filters], \
                                       name="ConvShape")
                # NOTE: downsampling doubles the output channel count following
                #       VGG-philosophy of preserving computational complexity
                exp_kern_shape = tf.stack([2*input_ch,1,1,bneck_filters], \
                                           name="ExpansionShape")
                zero_paddings   = tf.stack([0,0,0,input_ch])
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
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
                out, bn_params = self._batch_normalization(out, is_training, \
                                                           momentum=bn_momentum)
                variables["Expansion"] = {}
                variables["Expansion"]["Kernel"] = kern
                variables["Expansion"]["BatchNorm"] = bn_params
                # NOTE: no prelu here
                # END scope Expansion
            #####################################

            ########## Residual Branch ##########
            # MAYBE: add Targmax=tf.int32 below? (can still address 4GB)
            res_out, max_pool_argmax = \
                tf.nn.max_pool_with_argmax(inputs, \
                                           ksize=[1,2,2,1], \
                                           strides=[1,2,2,1], \
                                           padding=padding, \
                                           name="MaxPool")
            # tf.tile() ?
            res_out = tf.pad(res_out, \
                             paddings=zero_padding, \
                             name="ZeroPad")
            #####################################

            alpha = tf.get_variable(name="Alpha", \
                                    dtype=tf.float32, \
                                    initializer=alpha_initializer, \
                                    variables["Conv"]["BatchNorm"] = bn_params
                                    shape=input_ch, \
                                    trainable=True)
            # NOTE: out comes from main branch
            out = tf.add(inputs, out, name="Residual")
            out = xops.prelu(out, alpha, name="PReLU")
            variables["Alpha"] = alpha

        # end with tf.variable_scope(name)

        return out, variables, max_pool_argmax
    # END _block_bottleneck

    def _build(self):

        return inputs, outputs
