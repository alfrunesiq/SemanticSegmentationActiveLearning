from tensorflow.compat import v1 as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.client   import device_lib
_LOCAL_DEVICE_PROTOS = device_lib.list_local_devices()
def _get_available_gpus():
    return [dev.name for dev in _LOCAL_DEVICE_PROTOS if dev.device_type == "GPU"]


def prelu(x, alpha, name="PReLU"):
    """
    Parametrized Rectified Linear Unit
    https://arxiv.org/pdf/1502.01852.pdf
    :param x:     input features
    :param alpha: parameter (Variable)
                  NOTE: should match channel depth
    :param name:  Name scope of the operations
    :returns: Rectified feature map
    :rtype:   tf.Tensor

    """
    with tf.name_scope(name):
        pos = tf.nn.relu(x, name="Pos")
        neg = tf.nn.relu(-x, name="Neg")
        neg_scaled = tf.multiply(neg, alpha, name="Scale")
        ret = tf.subtract(pos, neg_scaled, name="Sub")
        return ret

def unpool_2d(inputs,
              idx,
              strides=[1, 2, 2, 1],
              name='Unpool2D'):
    """
    2D unpooling layer as described in:
    https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
    The unpooling is realized using the scatter operation (witch is what unpool
    operation really is). In order to utilize the scatter function along with the
    flattened indices of tf.nn.max_pool_with_argmax the indeces needs to be
    manually scaled in the batch dimension, and the inputs flattened.
    NOTE: this layer only works with indeces from a matching
          tf.nn.max_pool_with_argmax with padding="SAME"
    MAYBE: add padding parameter if needed
    NOTE: currently only works with GPU, see remark below

    :param inputs:  input feature map to be upsampled
    :param idx:     indices from tf.nn.max_pool_with_argmax
    :param strides: strides that matches the corresponding max_pool in
                    the encoder
    :param name:    name of the outer scope of the operation
    :returns:       output tensor
    :rtype:         tf.Tensor

    """
    with tf.variable_scope(name):
        with tf.name_scope("ShapeOps"):
            input_shape     = tf.shape(inputs, out_type=tf.int32)
            output_shape    = [input_shape[0], input_shape[1] * strides[1], \
                               input_shape[2] * strides[2], input_shape[3]]
            flat_input_size = tf.reduce_prod(input_shape)
            out_img_size    = output_shape[1] * output_shape[2] * output_shape[3]
            flat_out_size   = output_shape[0]*out_img_size

        inputs_ = tf.reshape(inputs, [flat_input_size])
        # NOTE there is an ERROR in the docs:
        # https://www.tensorflow.org/api_docs/python/tf/nn/max_pool_with_argmax
        # The batch dimension is not counted for in the argmax tensor, i.e. the
        # actual argmax is: (y * width + x) * channels + c
        # for each image in batch dimension
        # UPDATE: there is a BUG in Tensorflow behaviour, apparently the above
        #         remark only applies to CPU implementation of the op.
        #         https://github.com/tensorflow/tensorflow/pull/23993
        # We're just going to assume that the ops are running on gpu if one is
        # available
        if len(_get_available_gpus()) > 0:
            if idx.dtype == tf.int64:
                idx = tf.cast(idx, tf.int32)
            # scale indeces for GPU bug
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32),
                                     shape=[input_shape[0], 1, 1, 1])*out_img_size
            idx_sc = idx + batch_range
        else:
            idx_sc = idx
        idx_sc_ = tf.reshape(idx_sc, [flat_input_size, 1])
        ret = tf.scatter_nd(idx_sc_, inputs_, shape=[flat_out_size])
        # Restore the output shape
        ret = tf.reshape(ret, output_shape)
    return ret

def batch_normalization(inputs, training=True, decay=0.9):
    """
    In good Tensorflow spirit this function is DEPRECATED.
    """
    params = {}
    with tf.variable_scope("BatchNorm"):
        with tf.name_scope("ShapeOps"):
            input_ch = inputs.shape[-1]
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
        if training:
            out, batch_mean, batch_var = \
                        tf.nn.fused_batch_norm(inputs, \
                                               scale=params["Gamma"], \
                                               offset=params["Beta"], \
                                               is_training=training)
            update_mean = tf.assign_sub(params["Mean"], \
                                        (1-decay)*(params["Mean"]-batch_mean))
            update_var  = tf.assign_sub(params["Variance"], \
                                        (1-decay)*(params["Variance"]-batch_var))
            with tf.control_dependencies([update_mean, update_var]):
                out = tf.identity(out)
        else:
            out, _, _ = tf.nn.fused_batch_norm(inputs, \
                                               scale=params["Gamma"], \
                                               offset=params["Beta"], \
                                               mean=params["Mean"], \
                                               variance=params["Variance"], \
                                               is_training=training)
    return out, params

def spatial_dropout(inputs, drop_rate, name="SpatialDropout"):
    """
    Applies channelwise spatial (activation) dropout.
    :param inputs:    input tensor (4D tf.Tensor - NHWC)
    :param drop_rate: dropout rate
    :param name:      name of the operation scope
    :returns: output tensor (same dims as @inputs)
    :rtype:   tf.Tensor
    """
    with tf.name_scope(name):
        input_shape = tf.shape(inputs, name="InputShape")
        noise_shape = tf.stack([input_shape[0], 1, 1, input_shape[-1]])
        out = tf.nn.dropout(inputs, noise_shape=noise_shape,
                            rate=drop_rate, name="Dropout")
    return out


def batch_norm(inputs, mean, var, gamma, beta, training=True, decay=0.9):
    """
    Wrapper around batch normalization
    :param inputs:   input tensor (4D tf.Tensor - NHWC)
    :param mean:     mean variable
    :param var:      variance variable
    :param gamma:    scale variable
    :param beta:     offset variable
    :param training: whether accumulate mean and compute batch statistics
    :param decay:    moving average decay parameter
    :returns: normalized tensor (same dims as @inputs)
    :rtype: tf.Tensor

    """
    update_mean = None
    update_var  = None
    if training:
        out, batch_mean, batch_var = tf.nn.fused_batch_norm(
            inputs, scale=gamma,
            offset=beta, is_training=training)
        with tf.name_scope("MovingAverages"):
            update_mean = moving_averages.assign_moving_average(mean, batch_mean,
                                                                decay=decay,
                                                                name="MovingMean")
            update_var = moving_averages.assign_moving_average(var, batch_var,
                                                               decay=decay,
                                                               name="MovingVar")
    else:
        out, _, _ = tf.nn.fused_batch_norm(inputs, scale=gamma,
                                           offset=beta, mean=mean,
                                           variance=var, is_training=training)
    return out, update_mean, update_var
