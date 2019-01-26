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

    def _max_unpool(self, inputs, max_pool_indices, name="MaxUnpool"):
        """FIXME! briefly describe function

        :param inputs: 
        :param max_pool_indices: 
        :param name: 
        :returns: 
        :rtype: (tf.Tensor, dict)

        """
        with tf.name_scope("MaxUnpool"):
            with tf.name_scope("ShapeOps"):
                in_shape = tf.shape(inputs)
                out_shape = in_shape * tf.constant([1,2,2,1])
            out = tf.manip.scatter_nd(max_pool_indices, inputs, \
                                      out_shape, name="MaxUnpool")
        return out

    def _batch_normalization(self, inputs, is_training=True, momentum=0.9):
        params = {}
        with tf.variable_scope("BatchNorm"):
            params["Mean"] = tf.get_variable(shape=[filters], \
                                             initializer=tf.zeros_initializer(), \
                                             trainable=False, \
                                             dtype=tf.float32, \
                                             name="Mean")
            params["Variance"] = tf.get_variable(shape=[filters], \
                                                 initializer=tf.ones_initializer(), \
                                                 dtype=tf.float32, \
                                                 trainable=False, \
                                                 name="Variance")
            params["Beta"] = tf.get_variable(shape=[filters],
                                             initializer=tf.zeros_initializer(), \
                                             trainable=True, \
                                             dtype=tf.float32, \
                                             name="Beta")
            params["Gamma"] = tf.get_variable(shape=[filters],
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


    def _conv2d(self, inputs, kernel_shape, \
                strides=[1,1,1,1], \
                padding="SAME", \
                use_bias=False, \
                name="Conv"):
        """FIXME! briefly describe function

        :param inputs:
        :param kernel_shape:
        :param strides:
        :param padding:
        :param use_bn:
        :param use_bias:
        :param name:
        :returns:
        :rtype: (tf.Tensor, dict)
        """
        params = {}
        with tf.variable_scope(name):
            kern = tf.get_variable(name="Kernel", \
                                   shape=kernel_shape, \
                                   initializer= \
                                   tf.initializers.glorot_uniform(dtype=tf.float32), \
                                   trainable=True)
            out = tf.nn.conv2d(inputs, kern, \
                               strides=strides, \
                               padding=padding, \
                               name="Conv2D")
            params["Kernel"] = kern
            if use_bias:
                bias = tf.get_variable("Bias", shape=kernel_shape[0])
                out = tf.nn.bias_add(out, bias, name="BiasAdd")
                params["Bias"] = bias
        return out, params


    #TODO implement this using the functions above
    #TODO also create separate bottleneck blocks for down- and upsampling respectively
    #      - the latter uses tf.nn.conv2d_transpose and tf.manip.scatter_nd respectively
    def _block_bottleneck(self, inputs, input_shape, filters, \
                          padding="SAME", \
                          separable_conv=False, \
                          bottleneck_factor=4, \
                          dilations=[1,1,1,1], \
                          name="Bottleneck"):
        """FIXME! briefly describe function

        :param inputs: 
        :param input_shape: 
        :param filters: 
        :param padding: 
        :param separable_conv: 
        :param bottleneck_factor: 
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
                input_ch = tf.shape(inputs, name="InputShape")[3]
                # Number of filters in the bottleneck are reduced by a factor of
                # @bottleneck_factor
                bneck_filters = filters / bottleneck_factor
                # Get conv. kernels
                if downsampling:
                    down_proj_shape = tf.stack([bneck_filters,2,2,input_ch], \
                                               name="DownProjectShape")
                    zero_paddings   = tf.stack([0,0,0,filters - input_ch])
                else:
                    down_proj_shape = tf.stack([bneck_filters,1,1,input_ch], \
                                               name="DownProjectShape")
                if separable_conv:
                    conv_shape = [tf.stack([bneck_filters,1,5,bneck_filters], \
                                           name="ConvRowShape"), \
                                  tf.stack([bneck_filters,5,1,bneck_filters], \
                                           name="ConvColShape")]
                else: #(default)
                    conv_shape  = tf.stack([bneck_filters,3,3,bneck_filters], \
                                           name="ConvShape")
                expansion_shape = tf.stack([filters,1,1,bneck_filters], \
                                           name="ExpansionShape")
                # END ShapeOps

            ############ Main Branch ############
            with tf.variable_scope("DownProject"):
                # Bottleneck projection operation
                alpha = tf.get_variable(name="Alpha", \
                                        initializer= \
                                        tf.initializers.constant(0.25, \
                                                                 dtype=tf.float32)
                                        shape=bneck_filters, \
                                        trainable=True)
                kern = tf.get_variable(name="Kernel", \
                                       initializer= \
                                       tf.initializers.glorot_uniform(dtype=tf.float32), \
                                       shape=down_proj_shape, \
                                       trainable=True)
                variables["DownProject"] = {}
                variables["DownProject"]["Kernel"] = kern
                variables["DownProject"]["Alpha"] = alpha
                if downsampling:
                    out = tf.nn.conv2d(inputs, kern, \
                                       strides=[1,2,2,1], \
                                       padding=padding, \
                                       name="Conv2D")
                else: #(default)
                    out = tf.nn.conv2d(inputs, kern, \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       name="Conv2D")
                xops.prelu(out, alpha, name="PReLU")

            with tf.variable_scope("Conv"):
                # Main convolution operation
                alpha = tf.get_variable(name="Alpha", \
                                        initializer= \
                                        tf.initializers.constant(0.25, \
                                                                 dtype=tf.float32)
                                        shape=bneck_filters, \
                                        trainable=True)
                bias  = tf.get_variable(name="BiasCol", \
                                        initializer= \
                                        tf.initializers.zeros(dtype=tf.float32), \
                                        shape=bneck_filters, \
                                        trainable=True) \
                if separable_conv:
                    kern = [ \
                        tf.get_variable(name="KernelRow", \
                                        initializer= \
                                        tf.initializers.glorot_uniform(dtype=tf.float32), \
                                        shape=conv_shape[0], \
                                        trainable=True), \
                        tf.get_variable(name="KernelCol", \
                                        initializer= \
                                        tf.initializers.glorot_uniform(dtype=tf.float32), \
                                        shape=conv_shape[1], \
                                        trainable=True) \
                    ]
                    bias = tf.get_variable(name="BiasCol", \
                                           initializer= \
                                           tf.initializers.zeros(dtype=tf.float32), \
                                           shape=bneck_filters, \
                                           trainable=True)
                    # Separable convolution:
                    out = tf.nn.conv2d(out, kern[0], \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       dilations=dilations, \
                                       name="ConvRow")
                    out = tf.nn.conv2d(out, kern[1], \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       dilations=dilations, \
                                       name="ConvCol")

                else: #(defalut)
                    kern = tf.get_variable(name="Kernel", \
                                           initializer= \
                                           tf.initializers.glorot_uniform(dtype=tf.float32), \
                                           shape=conv_shape, \
                                           trainable=True)
                    out = tf.nn.conv2d(out, kern, \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       dilations=dilations, \
                                       name="Conv2D")

                out = tf.nn.bias_add(out, bias, name="BiasAdd")

                out = xops.prelu(out, alpha, name="PReLU")

                variables["Conv"] = {}
                variables["Conv"]["Kernel"] = kern
                variables["Conv"]["Bias"]   = bias
                variables["Conv"]["Alpha"]  = alpha
            # END scope Conv

            with tf.variable_scope("Expansion"):
                # Feature expansion operation
                kern = tf.get_variable(name="Kernel", \
                                       initializer= \
                                       tf.initializers.glorot_uniform(dtype=tf.float32), \
                                       shape=expansion_shape, \
                                       trainable=True)
                variables["Expansion"] = {}
                variables["Expansion"]["Kernel"] = kern
                out = tf.nn.conv2d(out, kern, \
                                   strides=[1,1,1,1], \
                                   padding=padding, \
                                   name="Conv2D")
                # NOTE: no prelu here
            #####################################

            ########## Residual Branch ##########
            if downsampling:
                res_out = tf.nn.max_pool(inputs, \
                                         ksize=[1,2,2,1], \
                                         strides=[1,2,2,1], \
                                         padding=padding, \
                                         name="MaxPool")
                # tf.tile() ?
                res_out = tf.pad(res_out, \
                                 paddings=zero_padding, \
                                 name="ZeroPad")
            else:
                res_out = inputs
            #####################################

            out = tf.add(res_out, out, name="Residual")
            # TODO: insert xops.PReLU here

        # end with tf.variable_scope(name)
        return out, variables
    # END _block_bottleneck

    def _block_bottleneck_downsample(self, inputs, filters, \
                                     padding="SAME", \
                                     separable_conv=False, \
                                     bottleneck_factor=4, \
                                     dilations=[1,1,1,1], \
                                     name="BottleneckDownsample"):
        variables = {}   # Dict with Variables in the block
        out       = None
        with tf.variable_scope(name):

            with tf.name_scope("ShapeOps"):
                # Get input channel count
                input_ch = tf.shape(inputs, name="InputShape")[3]
                # Number of filters in the bottleneck are reduced by a factor of
                # @bottleneck_factor
                bneck_filters = filters / bottleneck_factor
                # Get conv. kernels
                down_proj_shape = tf.stack([bneck_filters,2,2,input_ch], \
                                           name="DownProjectShape")
                zero_paddings   = tf.stack([0,0,0,filters - input_ch])
                if separable_conv:
                    conv_shape = [tf.stack([bneck_filters,1,5,bneck_filters], \
                                           name="ConvRowShape"), \
                                  tf.stack([bneck_filters,5,1,bneck_filters], \
                                           name="ConvColShape")]
                else: #(default)
                    conv_shape  = tf.stack([bneck_filters,3,3,bneck_filters], \
                                           name="ConvShape")
                expansion_shape = tf.stack([filters,1,1,bneck_filters], \
                                           name="ExpansionShape")
                # END ShapeOps

            ############ Main Branch ############
            with tf.variable_scope("DownProject"):
                # Bottleneck projection operation
                alpha = tf.get_variable(name="Alpha", \
                                        initializer= \
                                        tf.initializers.constant(0.25, \
                                                                 dtype=tf.float32)
                                        shape=bneck_filters, \
                                        trainable=True)
                kern = tf.get_variable(name="Kernel", \
                                       initializer= \
                                       tf.initializers.glorot_uniform(dtype=tf.float32), \
                                       shape=down_proj_shape, \
                                       trainable=True)
                variables["DownProject"] = {}
                variables["DownProject"]["Kernel"] = kern
                variables["DownProject"]["Alpha"] = alpha
                out = tf.nn.conv2d(inputs, kern, \
                                   strides=[1,2,2,1], \
                                   padding=padding, \
                                   name="Conv2D")
                xops.prelu(out, alpha, name="PReLU")

            with tf.variable_scope("Conv"):
                # Main convolution operation
                alpha = tf.get_variable(name="Alpha", \
                                        initializer= \
                                        tf.initializers.constant(0.25, \
                                                                 dtype=tf.float32)
                                        shape=bneck_filters, \
                                        trainable=True)
                bias  = tf.get_variable(name="BiasCol", \
                                        initializer= \
                                        tf.initializers.zeros(dtype=tf.float32), \
                                        shape=bneck_filters, \
                                        trainable=True) \
                if separable_conv:
                    kern = [ \
                        tf.get_variable(name="KernelRow", \
                                        initializer= \
                                        tf.initializers.glorot_uniform(dtype=tf.float32), \
                                        shape=conv_shape[0], \
                                        trainable=True), \
                        tf.get_variable(name="KernelCol", \
                                        initializer= \
                                        tf.initializers.glorot_uniform(dtype=tf.float32), \
                                        shape=conv_shape[1], \
                                        trainable=True) \
                    ]
                    bias = tf.get_variable(name="BiasCol", \
                                           initializer= \
                                           tf.initializers.zeros(dtype=tf.float32), \
                                           shape=bneck_filters, \
                                           trainable=True)
                    # Separable convolution:
                    out = tf.nn.conv2d(out, kern[0], \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       dilations=dilations, \
                                       name="ConvRow")
                    out = tf.nn.conv2d(out, kern[1], \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       dilations=dilations, \
                                       name="ConvCol")

                else: #(defalut)
                    kern = tf.get_variable(name="Kernel", \
                                           initializer= \
                                           tf.initializers.glorot_uniform(dtype=tf.float32), \
                                           shape=conv_shape, \
                                           trainable=True)
                    out = tf.nn.conv2d(out, kern, \
                                       strides=[1,1,1,1], \
                                       padding=padding, \
                                       dilations=dilations, \
                                       name="Conv2D")

                out = tf.nn.bias_add(out, bias, name="BiasAdd")

                out = xops.prelu(out, alpha, name="PReLU")

                variables["Conv"] = {}
                variables["Conv"]["Kernel"] = kern
                variables["Conv"]["Bias"]   = bias
                variables["Conv"]["Alpha"]  = alpha
            # END scope Conv

            with tf.variable_scope("Expansion"):
                # Feature expansion operation
                kern = tf.get_variable(name="Kernel", \
                                       initializer= \
                                       tf.initializers.glorot_uniform(dtype=tf.float32), \
                                       shape=expansion_shape, \
                                       trainable=True)
                variables["Expansion"] = {}
                variables["Expansion"]["Kernel"] = kern
                out = tf.nn.conv2d(out, kern, \
                                   strides=[1,1,1,1], \
                                   padding=padding, \
                                   name="Conv2D")
                # NOTE: no prelu here
            #####################################

            ########## Residual Branch ##########
            res_out = tf.nn.max_pool(inputs, \
                                     ksize=[1,2,2,1], \
                                     strides=[1,2,2,1], \
                                     padding=padding, \
                                     name="MaxPool")
            # tf.tile() ?
            res_out = tf.pad(res_out, \
                             paddings=zero_padding, \
                             name="ZeroPad")
            #####################################

            out = tf.add(res_out, out, name="Residual")
            # TODO: insert xops.PReLU here

        # end with tf.variable_scope(name)
        return out, variables
    # END _block_bottleneck

        return out, variables, maxPoolIdx

    def _block_bottleneck_upsample(self, inputs, filters, \
                                   padding="SAME", \
                                   separable_conv=False, \
                                   bottleneck_factor=4, \
                                   dilations=[1,1,1,1], \
                                   name="BottleneckUpsample"):

        return out, variables

    def _build(self):

        return inputs, outputs
