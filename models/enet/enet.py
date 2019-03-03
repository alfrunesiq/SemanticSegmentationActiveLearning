import tensorflow as tf

from tensorflow.keras import Model
from . import enet_modules as mod

class ENet(Model):
    """
    https://arxiv.org/pdf/1606.02147.pdf
    """
    def __init__(self, classes, name="ENet"):
        """

        :param classes: number of output classes
        """

        self.classes = classes
        super(ENet, self).__init__(name=name)

        # Define all layers as in the paper
        self.Initial = mod.Initial(16)

        # Stage 1
        self.Bottleneck1_0 = mod.BottleneckDownsample(64, name="Bottleneck1_0",
                                                      drop_rate=0.01)
        self.Bottleneck1_1 = mod.Bottleneck(64, name="Bottleneck1_1",
                                            drop_rate=0.01)
        self.Bottleneck1_2 = mod.Bottleneck(64, name="Bottleneck1_2",
                                            drop_rate=0.01)
        self.Bottleneck1_3 = mod.Bottleneck(64, name="Bottleneck1_3",
                                            drop_rate=0.01)
        self.Bottleneck1_4 = mod.Bottleneck(64, name="Bottleneck1_4",
                                            drop_rate=0.01)

        # Stage 2
        self.Bottleneck2_0 = mod.BottleneckDownsample(128, name="Bottleneck2_0",
                                                      drop_rate=0.1)
        self.Bottleneck2_1 = mod.Bottleneck(128, name="Bottleneck2_1")
        self.Bottleneck2_2 = mod.Bottleneck(128, name="Bottleneck2_2",
                                            drop_rate=0.1,
                                            dilation_rate=(2,2))
        self.Bottleneck2_3 = mod.Bottleneck(128, name="Bottleneck2_3",
                                            drop_rate=0.1,
                                            asymmetric=True,
                                            kernel_size=(5,5))
        self.Bottleneck2_4 = mod.Bottleneck(128, name="Bottleneck2_4",
                                            drop_rate=0.1,
                                            dilation_rate=(4,4))
        self.Bottleneck2_5 = mod.Bottleneck(128, name="Bottleneck2_5",
                                            drop_rate=0.1)
        self.Bottleneck2_6 = mod.Bottleneck(128, name="Bottleneck2_6",
                                            drop_rate=0.1,
                                            dilation_rate=(8,8))
        self.Bottleneck2_7 = mod.Bottleneck(128, name="Bottleneck2_7",
                                            drop_rate=0.1,
                                            asymmetric=True,
                                            kernel_size=(5,5))
        self.Bottleneck2_8 = mod.Bottleneck(128, name="Bottleneck2_8",
                                            drop_rate=0.1,
                                            dilation_rate=(8,8))

        # Stage 3
        self.Bottleneck3_1 = mod.Bottleneck(128, name="Bottleneck3_1",
                                            drop_rate=0.1)
        self.Bottleneck3_2 = mod.Bottleneck(128, name="Bottleneck3_2",
                                            drop_rate=0.1,
                                            dilation_rate=(2,2))
        self.Bottleneck3_3 = mod.Bottleneck(128, name="Bottleneck3_3",
                                            drop_rate=0.1,
                                            asymmetric=True,
                                            kernel_size=(5,5))
        self.Bottleneck3_4 = mod.Bottleneck(128, name="Bottleneck3_4",
                                            drop_rate=0.1,
                                            dilation_rate=(4,4))
        self.Bottleneck3_5 = mod.Bottleneck(128, name="Bottleneck3_5",
                                            drop_rate=0.1)
        self.Bottleneck3_6 = mod.Bottleneck(128, name="Bottleneck3_6",
                                            drop_rate=0.1,
                                            dilation_rate=(8,8))
        self.Bottleneck3_7 = mod.Bottleneck(128, name="Bottleneck3_7",
                                            drop_rate=0.1,
                                            asymmetric=True,
                                            kernel_size=(5,5))
        self.Bottleneck3_8 = mod.Bottleneck(128, name="Bottleneck3_8",
                                            drop_rate=0.1,
                                            dilation_rate=(8,8))

        # Stage 4
        self.Bottleneck4_0 = mod.BottleneckUpsample(64, name="Bottleneck4_0",
                                                    drop_rate=0.1)
        self.Bottleneck4_1 = mod.Bottleneck(64, name="Bottleneck4_1",
                                            drop_rate=0.1)
        self.Bottleneck4_2 = mod.Bottleneck(64, name="Bottleneck4_2",
                                            drop_rate=0.1)

        # Stage 5
        self.Bottleneck5_0 = mod.BottleneckUpsample(16, name="Bottleneck5_0",
                                                    drop_rate=0.1)
        self.Bottleneck5_1 = mod.Bottleneck(16, name="Bottleneck5_1",
                                            drop_rate=0.1)

        # Final UpConv
        self.Final = mod.Final(self.classes)

        # Output lists
        self.initial = []

        self.bottleneck1_0 = []
        self.bottleneck1_1 = []
        self.bottleneck1_2 = []
        self.bottleneck1_3 = []
        self.bottleneck1_4 = []

        self.bottleneck2_0 = []
        self.bottleneck2_1 = []
        self.bottleneck2_2 = []
        self.bottleneck2_3 = []
        self.bottleneck2_4 = []
        self.bottleneck2_5 = []
        self.bottleneck2_6 = []
        self.bottleneck2_7 = []
        self.bottleneck2_8 = []

        self.bottleneck3_1 = []
        self.bottleneck3_2 = []
        self.bottleneck3_3 = []
        self.bottleneck3_4 = []
        self.bottleneck3_5 = []
        self.bottleneck3_6 = []
        self.bottleneck3_7 = []
        self.bottleneck3_8 = []

        self.bottleneck4_0 = []
        self.bottleneck4_1 = []
        self.bottleneck4_2 = []

        self.bottleneck5_0 = []
        self.bottleneck5_1 = []

        self.final = []

    def build(self, input_shape):
        """
        Store the absolute name scopes used in @call to
        enable scope reuse.
        """
        if self.built:
            return
        # Save name scopes
        with tf.name_scope("Stage1") as scope:
            self._stage1_scope = scope
        with tf.name_scope("Stage2") as scope:
            self._stage2_scope = scope
        with tf.name_scope("Stage3") as scope:
            self._stage3_scope = scope
        with tf.name_scope("Stage4") as scope:
            self._stage4_scope = scope
        with tf.name_scope("Stage5") as scope:
            self._stage5_scope = scope
        self.built = True

    def call(self, inputs, training):
        """
        Implements the __call__ building functionality, interconnecting
        the network modules.
        :param inputs:   input tensor (4D tf.Tensor - NHWC)
        :param training: build for training of inference
        :returns: network logits
        :rtype:   tf.Tensor
        """
        initial = self.Initial(inputs, training)
        with tf.name_scope(self._stage1_scope): # Stage 1
            bottleneck1_0, argmax1 = self.Bottleneck1_0(initial, training)
            bottleneck1_1 = self.Bottleneck1_1(bottleneck1_0, training)
            bottleneck1_2 = self.Bottleneck1_2(bottleneck1_1, training)
            bottleneck1_3 = self.Bottleneck1_3(bottleneck1_2, training)
            bottleneck1_4 = self.Bottleneck1_4(bottleneck1_3, training)

        with tf.name_scope(self._stage2_scope): # Stage 2
            bottleneck2_0, argmax2 = self.Bottleneck2_0(bottleneck1_4, training)
            bottleneck2_1 = self.Bottleneck2_1(bottleneck2_0, training)
            bottleneck2_2 = self.Bottleneck2_2(bottleneck2_1, training)
            bottleneck2_3 = self.Bottleneck2_3(bottleneck2_2, training)
            bottleneck2_4 = self.Bottleneck2_4(bottleneck2_3, training)
            bottleneck2_5 = self.Bottleneck2_5(bottleneck2_4, training)
            bottleneck2_6 = self.Bottleneck2_6(bottleneck2_5, training)
            bottleneck2_7 = self.Bottleneck2_7(bottleneck2_6, training)
            bottleneck2_8 = self.Bottleneck2_8(bottleneck2_7, training)

        with tf.name_scope(self._stage3_scope): # Stage 3
            bottleneck3_1 = self.Bottleneck3_1(bottleneck2_8, training)
            bottleneck3_2 = self.Bottleneck3_2(bottleneck3_1, training)
            bottleneck3_3 = self.Bottleneck3_3(bottleneck3_2, training)
            bottleneck3_4 = self.Bottleneck3_4(bottleneck3_3, training)
            bottleneck3_5 = self.Bottleneck3_5(bottleneck3_4, training)
            bottleneck3_6 = self.Bottleneck3_6(bottleneck3_5, training)
            bottleneck3_7 = self.Bottleneck3_7(bottleneck3_6, training)
            bottleneck3_8 = self.Bottleneck3_8(bottleneck3_7, training)

        with tf.name_scope(self._stage4_scope): # Stage 4
            bottleneck4_0 = self.Bottleneck4_0(bottleneck3_8, argmax2, training)
            bottleneck4_1 = self.Bottleneck4_1(bottleneck4_0, training)
            bottleneck4_2 = self.Bottleneck4_2(bottleneck4_1, training)

        with tf.name_scope(self._stage5_scope): # Stage 5
            bottleneck5_0 = self.Bottleneck5_0(bottleneck4_2, argmax1, training)
            bottleneck5_1 = self.Bottleneck5_1(bottleneck5_0, training)

        final  = self.Final(bottleneck5_1)

        # Add layer outputs to respective lists
        self.initial.append(initial)

        self.bottleneck1_0.append(bottleneck1_0)
        self.bottleneck1_1.append(bottleneck1_1)
        self.bottleneck1_2.append(bottleneck1_2)
        self.bottleneck1_3.append(bottleneck1_3)
        self.bottleneck1_4.append(bottleneck1_4)

        self.bottleneck2_0.append(bottleneck2_0)
        self.bottleneck2_1.append(bottleneck2_1)
        self.bottleneck2_2.append(bottleneck2_2)
        self.bottleneck2_3.append(bottleneck2_3)
        self.bottleneck2_4.append(bottleneck2_4)
        self.bottleneck2_5.append(bottleneck2_5)
        self.bottleneck2_6.append(bottleneck2_6)
        self.bottleneck2_7.append(bottleneck2_7)
        self.bottleneck2_8.append(bottleneck2_8)

        self.bottleneck3_1.append(bottleneck3_1)
        self.bottleneck3_2.append(bottleneck3_2)
        self.bottleneck3_3.append(bottleneck3_3)
        self.bottleneck3_4.append(bottleneck3_4)
        self.bottleneck3_5.append(bottleneck3_5)
        self.bottleneck3_6.append(bottleneck3_6)
        self.bottleneck3_7.append(bottleneck3_7)
        self.bottleneck3_8.append(bottleneck3_8)

        self.bottleneck4_0.append(bottleneck4_0)
        self.bottleneck4_1.append(bottleneck4_1)
        self.bottleneck4_2.append(bottleneck4_2)

        self.bottleneck5_0.append(bottleneck5_0)
        self.bottleneck5_1.append(bottleneck5_1)

        self.final.append(final)
        self.outputs.append(final)

        return final
