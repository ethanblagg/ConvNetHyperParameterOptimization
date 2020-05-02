# subclass for keras Model class for image classification
import tensorflow as tf
import tensorflow.keras.layers as kl

class Ethanet(tf.keras.Model):

    conv1_filters = 12
    conv2_filters = 24
    conv3_filters = 48
    kernel_size = 4
    stride = 2
    padding = 'same'
    activation_relu = "relu"
    activation_sig = "sigmoid"
    pool_size = 2
    fc1_size = 60
    fc2_size = 5                            # default. need to set to num classes
    dropout_rate = 0.05

    def __init__(self, filt_size=12, num_logits=5):
        super(Ethanet, self).__init__()     # init superclass

        # conv  pooling  relu
        # conv  pooling  relu
        # conv  pooling  relu
        # flatten ?????
        # fully connected  relu
        # dropout
        # fully connected  sigmoid
        self.conv1_filters = filt_size
        self.conv2_filters = filt_size * 2
        self.conv3_filters = filt_size * 4
        self.fc1_size = num_logits * 2
        self.fc2_size = num_logits

        self.conv1 = kl.Conv2D(self.conv1_filters, self.kernel_size, self.stride, self.padding, activation=self.activation_relu)
        self.conv2 = kl.Conv2D(self.conv2_filters, self.kernel_size, self.stride, self.padding, activation=self.activation_relu)
        self.conv3 = kl.Conv2D(self.conv3_filters, self.kernel_size, self.stride, self.padding, activation=self.activation_relu)
        self.maxpooling = kl.MaxPooling2D(self.pool_size)
        self.flatten = kl.Flatten()
        self.fc1 = kl.Dense(self.fc1_size, activation=self.activation_relu)
        self.fc2 = kl.Dense(self.fc2_size, activation=None)#, activation=self.activation_sig)
        self.dropout = kl.Dropout(self.dropout_rate)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.conv3(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

