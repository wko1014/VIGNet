import tensorflow as tf
import numpy as np

class vignet(tf.keras.Model):
    def __init__(self, mode):
        tf.keras.backend.set_floatx("float64")
        super(vignet, self).__init__()

        self.mode = mode

        self.regularizer = tf.keras.regularizers.L1L2(l1=0.05, l2=0.005)
        self.activation = tf.nn.leaky_relu
        
        # Define convolution layers
        self.conv1 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv2 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv3 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv4 = tf.keras.layers.Conv2D(20, (17, 1), kernel_regularizer=self.regularizer, activation=self.activation)

        self.flatten = tf.keras.layers.Flatten()

        if self.mode == "CLF":
            self.dense = tf.keras.layers.Dense(3)
        elif self.mode == "RGS":
            self.dense = tf.keras.layers.Dense(1)

    def MHRSSA(self, x, out_filter, num_channel=17):
        for i in range(out_filter):
            tmp = tf.keras.layers.Conv2D(num_channel, (num_channel, 1), kernel_regularizer=self.regularizer, activation=None)(x)
            if i == 0: MHRSSA = tmp
            else: MHRSSA = tf.concat((MHRSSA, tmp), 1)

        MHRSSA = tf.transpose(MHRSSA, perm=[0, 3, 2, 1])

        MHRSSA = tf.keras.layers.DepthwiseConv2D((1, 5), kernel_regularizer=self.regularizer, activation=None)(MHRSSA)
        MHRSSA = tf.keras.activations.softmax(MHRSSA)
        return MHRSSA


    def call(self, x):
        att1 = self.MHRSSA(x, 10)
        hidden = self.conv1(x)
        hidden *= att1

        att2 = self.MHRSSA(hidden, 10)
        hidden = self.conv2(hidden)
        hidden *= att2

        att3 = self.MHRSSA(hidden, 10)
        hidden = self.conv3(hidden)
        hidden *= att3

        hidden = self.conv4(hidden)

        hidden = self.flatten(hidden)
        hidden = self.dense(hidden)

        if self.mode == "CLF":
            y_hat = tf.keras.activations.softmax(hidden)
        elif self.mode == "RGS":
            y_hat = hidden
        return y_hat

