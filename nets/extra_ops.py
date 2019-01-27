import tensorflow as tf

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
            input_shape     = tf.shape(inputs, out_type=tf.int64)
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
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=idx.dtype),
                                 shape=[input_shape[0], 1, 1, 1])*out_img_size
        idx_sc = idx + batch_range
        idx_sc_ = tf.reshape(idx_sc, [flat_input_size, 1])
        ret = tf.scatter_nd(idx_sc_, inputs_, shape=[flat_out_size])
        # Restore the output shape
        ret = tf.reshape(ret, output_shape)
    return ret
