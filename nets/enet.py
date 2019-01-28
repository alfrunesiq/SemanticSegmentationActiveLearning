#import numpy as np
import tensorflow as tf
import extra_ops as xops
import enet_modules as eblk

class ENet:
    """
    https://arxiv.org/pdf/1606.02147.pdf
    """
    def __init__(self, inputs, num_classes, is_training):
        self.outputs = self._build(inputs, num_classes, is_training)

    def _build(self, inputs, num_classes, is_training):
        net = None
        # initialize parameters dict
        self.parameters = {}
        self.parameters["Initial"] = {}
        for i in range(9):
            if i < 5:
                self.parameters["Bottleneck1_%d" % i] = {}
            self.parameters["Bottleneck2_%d" % i] = {}
            if i > 0:
                self.parameters["Bottleneck3_%d" % i] = {}
            if i < 3:
                self.parameters["Bottleneck4_%d" % i] = {}
            if i < 2:
                self.parameters["Bottleneck5_%d" % i] = {}
        self.parameters["Fullconv"] = {}

        # Below follows specs from Tab. 1 in (Paszke et. al 2016)
        with tf.variable_scope("Initial"):
            net, self.parameters["Initial"] = \
                    eblk.block_initial(inputs, is_training=is_training)
        with tf.variable_scope("Stage1"):
            net, self.parameters["Bottleneck1_0"], amax1 = \
                    eblk.block_bottleneck_downsample(net, name="Bottleneck1_0")
            net, self.parameters["Bottleneck1_1"] = \
                    eblk.block_bottleneck(net, name="Bottleneck1_1")
            net, self.parameters["Bottleneck1_2"] = \
                    eblk.block_bottleneck(net, name="Bottleneck1_2")
            net, self.parameters["Bottleneck1_3"] = \
                    eblk.block_bottleneck(net, name="Bottleneck1_3")
            net, self.parameters["Bottleneck1_4"] = \
                    eblk.block_bottleneck(net, name="Bottleneck1_4")
        with tf.variable_scope("Stage2"):
            net, self.parameters["Bottleneck2_0"], amax2 = \
                    eblk.block_bottleneck_downsample(net, name="Bottleneck2_0")
            net, self.parameters["Bottleneck2_1"] = \
                    eblk.block_bottleneck(net, name="Bottleneck2_1")
            net, self.parameters["Bottleneck2_2"] = \
                    eblk.block_bottleneck(net, dilations=[1,2,2,1], \
                                           name="Bottleneck2_2")
            net, self.parameters["Bottleneck2_3"] = \
                    eblk.block_bottleneck(net, asymmetric=True, \
                                           name="Bottleneck2_3")
            net, self.parameters["Bottleneck2_4"] = \
                    eblk.block_bottleneck(net, dilations=[1,4,4,1], \
                                           name="Bottleneck2_4")
            net, self.parameters["Bottleneck2_5"] = \
                    eblk.block_bottleneck(net, name="Bottleneck2_5")
            net, self.parameters["Bottleneck2_6"] = \
                    eblk.block_bottleneck(net, dilations=[1,8,8,1], \
                                           name="Bottleneck2_6")
            net, self.parameters["Bottleneck2_7"] = \
                    eblk.block_bottleneck(net, asymmetric=True, \
                                           name="Bottleneck2_7")
            net, self.parameters["Bottleneck2_8"] = \
                    eblk.block_bottleneck(net, dilations=[1,16,16,1], \
                                           name="Bottleneck2_8")

        with tf.variable_scope("Stage3"):
            net, self.parameters["Bottleneck3_1"] = \
                    eblk.block_bottleneck(net, name="Bottleneck3_1")
            net, self.parameters["Bottleneck3_2"] = \
                    eblk.block_bottleneck(net, dilations=[1,2,2,1], \
                                           name="Bottleneck3_2")
            net, self.parameters["Bottleneck3_3"] = \
                    eblk.block_bottleneck(net, asymmetric=True, \
                                           name="Bottleneck3_3")
            net, self.parameters["Bottleneck3_4"] = \
                    eblk.block_bottleneck(net, dilations=[1,4,4,1], \
                                           name="Bottleneck3_4")
            net, self.parameters["Bottleneck3_5"] = \
                    eblk.block_bottleneck(net, name="Bottleneck3_5")
            net, self.parameters["Bottleneck3_6"] = \
                    eblk.block_bottleneck(net, dilations=[1,8,8,1], \
                                           name="Bottleneck3_6")
            net, self.parameters["Bottleneck3_7"] = \
                    eblk.block_bottleneck(net, asymmetric=True, \
                                           name="Bottleneck3_7")
            net, self.parameters["Bottleneck3_8"] = \
                    eblk.block_bottleneck(net, dilations=[1,16,16,1], \
                                           name="Bottleneck3_8")

        with tf.variable_scope("Stage4"):
            net, self.parameters["Bottleneck4_0"] = \
                    eblk.block_bottleneck_upsample(net, amax2, \
                                                    name="Bottleneck4_0")
            net, self.parameters["Bottleneck4_1"] = \
                    eblk.block_bottleneck(net, name="Bottleneck4_1")
            net, self.parameters["Bottleneck4_2"] = \
                    eblk.block_bottleneck(net, name="Bottleneck4_2")

        with tf.variable_scope("Stage5"):
            net, self.parameters["Bottleneck5_0"] = \
                    eblk.block_bottleneck_upsample(net, amax1, \
                                                    name="Bottleneck5_0")
            net, self.parameters["Bottleneck5_1"] = \
                    eblk.block_bottleneck(net, name="Bottleneck5_1")

        # TODO: insert transpose convolution here

        return inputs, net
    # END def _build
