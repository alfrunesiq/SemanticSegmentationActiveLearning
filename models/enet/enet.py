import tensorflow as tf

from . import enet_modules as eblk
from .._model import Model

class ENet(Model):
    """
    https://arxiv.org/pdf/1606.02147.pdf
    """
    def __init__(self, num_classes, is_training):
        self.num_classes = num_classes
        self.is_training = is_training
        # Initialize network class members
        self.parameters  = {}
        self.logits      = None
        self.outputs     = None

    def build(self, inputs):
        # Build the graph
        logits = None
        is_training = self.is_training
        # initialize parameters dict
        self.parameters["Initial"] = {}
        for i in range(9):
            if i < 5:
                self.parameters["Bottleneck1_%d" % i] = {} #[0,4]
            self.parameters["Bottleneck2_%d" % i] = {}     #[0,8]
            if i > 0:
                self.parameters["Bottleneck3_%d" % i] = {} #[1,8]
            if i < 3:
                self.parameters["Bottleneck4_%d" % i] = {} #[0,2]
            if i < 2:
                self.parameters["Bottleneck5_%d" % i] = {} #[0,1]
        self.parameters["Fullconv"] = {}

        # Below follows specs from Tab. 1 in (Paszke et. al 2016)
        with tf.variable_scope("Initial"):
            logits, self.parameters["Initial"] = \
                    eblk.block_initial(inputs, is_training=is_training)
        with tf.variable_scope("Stage1"):
            logits, self.parameters["Bottleneck1_0"], amax1 = \
                    eblk.block_bottleneck_downsample(logits,
                                                     is_training=is_training, \
                                                     name="Bottleneck1_0")
            logits, self.parameters["Bottleneck1_1"] = \
                    eblk.block_bottleneck(logits, is_training=is_training, \
                                          name="Bottleneck1_1")
            logits, self.parameters["Bottleneck1_2"] = \
                    eblk.block_bottleneck(logits, is_training=is_training, \
                                          name="Bottleneck1_2")
            logits, self.parameters["Bottleneck1_3"] = \
                    eblk.block_bottleneck(logits, is_training=is_training, \
                                          name="Bottleneck1_3")
            logits, self.parameters["Bottleneck1_4"] = \
                    eblk.block_bottleneck(logits, is_training=is_training, \
                                          name="Bottleneck1_4")
        with tf.variable_scope("Stage2"):
            logits, self.parameters["Bottleneck2_0"], amax2 = \
                    eblk.block_bottleneck_downsample(logits, is_training=is_training, \
                                                     name="Bottleneck2_0")
            logits, self.parameters["Bottleneck2_1"] = \
                    eblk.block_bottleneck(logits, is_training=is_training,
                                          name="Bottleneck2_1")
            logits, self.parameters["Bottleneck2_2"] = \
                    eblk.block_bottleneck(logits, dilations=[1,2,2,1],
                                          is_training=is_training,
                                          name="Bottleneck2_2")
            logits, self.parameters["Bottleneck2_3"] = \
                    eblk.block_bottleneck(logits, asymmetric=True,
                                          is_training=is_training,
                                          name="Bottleneck2_3")
            logits, self.parameters["Bottleneck2_4"] = \
                    eblk.block_bottleneck(logits, dilations=[1,4,4,1],
                                          is_training=is_training,
                                          name="Bottleneck2_4")
            logits, self.parameters["Bottleneck2_5"] = \
                    eblk.block_bottleneck(logits, is_training=is_training,
                                          name="Bottleneck2_5")
            logits, self.parameters["Bottleneck2_6"] = \
                    eblk.block_bottleneck(logits, dilations=[1,8,8,1],
                                          is_training=is_training,
                                          name="Bottleneck2_6")
            logits, self.parameters["Bottleneck2_7"] = \
                    eblk.block_bottleneck(logits, asymmetric=True,
                                          is_training=is_training,
                                          name="Bottleneck2_7")
            logits, self.parameters["Bottleneck2_8"] = \
                    eblk.block_bottleneck(logits, dilations=[1,16,16,1],
                                          is_training=is_training,
                                          name="Bottleneck2_8")

        with tf.variable_scope("Stage3"):
            logits, self.parameters["Bottleneck3_1"] = \
                    eblk.block_bottleneck(logits, is_training=is_training,
                                          name="Bottleneck3_1")
            logits, self.parameters["Bottleneck3_2"] = \
                    eblk.block_bottleneck(logits, dilations=[1,2,2,1],
                                          is_training=is_training,
                                           name="Bottleneck3_2")
            logits, self.parameters["Bottleneck3_3"] = \
                    eblk.block_bottleneck(logits, asymmetric=True,
                                          is_training=is_training,
                                          name="Bottleneck3_3")
            logits, self.parameters["Bottleneck3_4"] = \
                    eblk.block_bottleneck(logits, dilations=[1,4,4,1],
                                          is_training=is_training,
                                          name="Bottleneck3_4")
            logits, self.parameters["Bottleneck3_5"] = \
                    eblk.block_bottleneck(logits, is_training=is_training,
                                          name="Bottleneck3_5")
            logits, self.parameters["Bottleneck3_6"] = \
                    eblk.block_bottleneck(logits, dilations=[1,8,8,1],
                                          is_training=is_training,
                                          name="Bottleneck3_6")
            logits, self.parameters["Bottleneck3_7"] = \
                    eblk.block_bottleneck(logits, asymmetric=True,
                                          is_training=is_training,
                                          name="Bottleneck3_7")
            logits, self.parameters["Bottleneck3_8"] = \
                    eblk.block_bottleneck(logits, dilations=[1,16,16,1],
                                          is_training=is_training,
                                          name="Bottleneck3_8")

        with tf.variable_scope("Stage4"):
            logits, self.parameters["Bottleneck4_0"] = \
                    eblk.block_bottleneck_upsample(logits, amax2,
                                                   is_training=is_training,
                                                   name="Bottleneck4_0")
            logits, self.parameters["Bottleneck4_1"] = \
                    eblk.block_bottleneck(logits, is_training=is_training,
                                          name="Bottleneck4_1")
            logits, self.parameters["Bottleneck4_2"] = \
                    eblk.block_bottleneck(logits, is_training=is_training,
                                          name="Bottleneck4_2")

        with tf.variable_scope("Stage5"):
            logits, self.parameters["Bottleneck5_0"] = \
                    eblk.block_bottleneck_upsample(logits, amax1,
                                                   is_training=is_training,
                                                   name="Bottleneck5_0")
            logits, self.parameters["Bottleneck5_1"] = \
                    eblk.block_bottleneck(logits,
                                          is_training=is_training,
                                          name="Bottleneck5_1")
        logits, self.parameters["Fullconv"] = eblk.block_final(logits,
                                                               self.num_classes,
                                                               name="Fullconv")
        self.logits = logits
        self.output = tf.nn.softmax(logits, name="Predictions")
        return self.output, self.parameters
    # END def _build

    def get_vars(self):
        # Recursively walk through parameter dict, and add the leaf nodes to a
        # variable list. NOTE: the dict and hence the list is not ordered.
        def dict_iterator(d):
            var_list = []
            for k in d:
                if isinstance(d[k], dict):
                    var_list.extend(dict_iterator(d[k]))
                elif isinstance(d[k], list):
                    var_list.extend(d[k])
                elif isinstance(d[k], tf.Variable):
                    var_list.append(d[k])
            return var_list
        return dict_iterator(self.parameters)

    def get_logits(self):
        return self.logits

    def get_output(self):
        return self.output

