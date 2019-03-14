import tensorflow as tf

def masked_softmax_cross_entropy(
        labels,
        logits,
        mask,
        num_classes,
        weight=0.0,
        label_smoothing=0.0,
        scope="XEntropy"):
    """
    Evaluates the softmax cross-entropy loss, and masks out labels using
    @mask parameter. Optionally if @weight is greater than zero, the
    loss also punishes missclassified labels by a factor (1+weights),
    this is to help cope with sparse classes.
    :param labels:          ground truth labels (tf.uint8)
    :param logits:          segmentation output logits
    :param mask:            mask for labels should be ignored
    :param num_classes:     number of valid calsses
    :param weight:          [optional] weight parameter to punish errors
    :param label_smoothing: [optional] label smoothing on per pixel
                            classification
    :param scope:           [optional] scope for the operations
    :returns: The mean cross entropy loss
    :rtype:   tf.Tensor: @logits.dtype (tf.float32)
    """
    with tf.name_scope(scope):
        # Squeeze out that singleton dimension
        _labels = labels
        if labels.shape.ndims == 4:
            _labels = tf.squeeze(_labels, axis=3)
        # Apply label smoothing
        on_value  = 1.0 - label_smoothing
        off_value = label_smoothing / (num_classes - 1.0)
        # Generate one-hot labels
        _labels_oh   = tf.one_hot(_labels, num_classes,
                                  on_value, off_value,
                                  axis=3,
                                  dtype=tf.float32,
                                  name="LabelsOneHot")
        _labels_oh = tf.stop_gradient(_labels_oh)

        # Create mask to ignore @mask_index labels
        _mask = mask
        if mask.dtype != tf.float32:
            _mask = tf.cast(mask, dtype=tf.float32)
        if weight > 1.0: # ENet type mask weighting
            p_class = tf.reduce_sum(tf.nn.softmax(logits) * _labels_oh, axis=-1)
            w_class = tf.math.divide(1.0, tf.math.log(weight + p_class))
            _mask = _mask * w_class

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=_labels_oh, logits=logits, axis=-1,
            name="SoftmaxCrossEntropy")
        # Apply mask / weighting
        loss = tf.math.multiply(loss, _mask)
        # Do the mean in two takes: first in batch dimension,
        # then spatial with higher precission
        loss = tf.reduce_mean(loss, axis=0, name="BatchMeanCrossEntropy")
        # Cast to float64, spatial dimensions can make the numeric
        # errors can be severe for high resolution images
        loss = tf.cast(loss, tf.float64)
        # Compute scalar mean loss
        loss = tf.math.reduce_mean(loss, name="MeanCrossEntropy")
    return loss

def multiscale_masked_softmax_cross_entropy(
        labels,
        logits,
        mask,
        num_classes,
        weight=0.0,
        label_smoothing=0.0,
        scope="MultiXEntropy"):
    """
    Evaluates the softmax cross-entropy loss, and masks out labels using
    @mask parameter. Optionally if @weight is greater than zero, the
    loss also punishes missclassified labels by a factor (1+weights),
    this is to help cope with sparse classes.
    :param labels:          ground truth labels (tf.uint8).
    :param logits:          list of segmentation output logits in
                            decrementing scale.
    :param mask:            mask for labels should be ignored.
    :param num_classes:     number of valid calsses.
    :param weight:          [optional] weight parameter to punish errors.
    :param label_smoothing: [optional] label smoothing on per pixel.
                            classification.
    :param scope:           [optional] scope for the operations.
    :returns: The mean cross entropy loss
    :rtype:   tf.Tensor: @logits.dtype (tf.float32)
    """
    with tf.name_scope(scope) as name_scope:
        # Squeeze out that singleton dimension
        _labels = labels
        _labels4 = labels
        if labels.shape.ndims == 4:
            _labels = tf.squeeze(_labels, axis=3)
        else:
            _labels4 = tf.expand_dims(labels, axis=-1)

        _mask = mask
        if mask.dtype != tf.float32:
            _mask = tf.cast(mask, dtype=tf.float32)

        losses = [masked_softmax_cross_entropy(_labels, logits[0],
                                               mask, num_classes,
                                               weight, label_smoothing)]
        weights = []
        var_count = 0
        for _logits in logits[1:]:
            # Create endpoint prediction perceptron weights
            with tf.variable_scope("Weights", reuse=tf.AUTO_REUSE):
                tmp_krnl = tf.get_variable("Kernel_" + str(var_count),
                                           shape=[1, 1,
                                                  _logits.shape[-1],
                                                  num_classes],
                                           trainable=True)
                var_count += 1
            # Get actual "logits"
            _logits_ = tf.nn.conv2d(_logits, tmp_krnl, 
                                    strides=[1,1,1,1], 
                                    padding="VALID")
            logits_shape = _logits_.shape.as_list()
            # Append weights to returned list
            weights.append(tmp_krnl)
            # Resize mask (careful not to interpolate)
            _mask_ = tf.image.resize_nearest_neighbor(tf.expand_dims(_mask,
                                                                     axis=-1),
                                                      logits_shape[1:3])
            _mask_ = tf.squeeze(_mask_, axis=3)
            # Resize labels (again, careful not to interpolate)
            _labels_ = tf.image.resize_nearest_neighbor(_labels4,
                                                        logits_shape[1:3])
            _labels_ = tf.squeeze(_labels_)
            loss = masked_softmax_cross_entropy(_labels_, _logits_, 
                                                _mask_, num_classes,
                                                weight, label_smoothing)
            # Append loss to overall loss
            losses.append(loss)
        # Sum losses and normalize
        loss = tf.math.add_n(losses) / float(len(losses))
    # Also make sure to return the weights so they can be saved
    return loss, weights


def L2_regularization(kernels, weight, scope=None):
    """
    Creates scalar L2_regularization loss over the list of @kernels
    :param kernels: list of tf.Tensor kernels
    :param weight:  weight parameter
    :param name:    name / scope of the operation
    :returns: l2 loss over the kernels
    :rtype:   tf.Tensor scalar
    """
    l2_losses = []
    for krnl in kernels:
        # Create name under kernel variable scope
        _name = krnl.name + "/L2Regularization"
        # Append loss (@tf.nn.l2_loss returns a scalar)
        l2_losses.append(tf.nn.l2_loss(krnl, name=_name))
    # Sum over all scalars
    _weight = weight / float(len(kernels))
    with tf.name_scope(scope):
        l2_loss = tf.math.add_n(l2_losses)
        l2_loss = tf.math.multiply(l2_loss, _weight)
    return l2_loss

__all__ = ["masked_softmax_cross_entropy", "L2_regularization"]
