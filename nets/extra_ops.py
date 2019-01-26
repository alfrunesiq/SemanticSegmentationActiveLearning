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
