import tensorflow as tf
from keras import layers
from keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D


class ChannelAttention(Layer):
    def __init__(self, ratio=0.25, kernel_initializer='he_normal', **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        channel = input_shape[-1]

        # 共享的全连接层
        self.shared_dense1 = layers.Dense(
            int(channel * self.ratio),
            activation='relu',
            kernel_initializer=self.kernel_initializer
        )
        self.shared_dense2 = layers.Dense(
            channel,
            kernel_initializer=self.kernel_initializer
        )

        # 全局池化层
        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()
        self.reshape = layers.Reshape((1, 1, channel))

        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        # 全局平均池化和最大池化
        x_avg = self.reshape(self.gap(inputs))
        x_max = self.reshape(self.gmp(inputs))

        # 共享全连接层处理
        x_avg = self.shared_dense2(self.shared_dense1(x_avg))
        x_max = self.shared_dense2(self.shared_dense1(x_max))

        # 相加并通过sigmoid
        x = layers.Add()([x_avg, x_max])
        x = layers.Activation('sigmoid')(x)

        return layers.Multiply()([inputs, x])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'ratio': self.ratio,
            'kernel_initializer': self.kernel_initializer
        })
        return config


class SpatialAttention(Layer):
    def __init__(self, kernel_size=(7, 7), kernel_initializer='he_normal', **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # 通道维度上的最大池化和平均池化
        x_max = tf.reduce_max(inputs, axis=3, keepdims=True)
        x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)

        # 拼接并卷积
        x = tf.concat([x_max, x_avg], axis=3)
        x = self.conv(x)

        return inputs * x

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer
        })
        return config


class CBAMAttention(Layer):
    def __init__(self, channel_ratio=0.25, spatial_kernel=(7, 7), kernel_initializer='he_normal', **kwargs):
        super(CBAMAttention, self).__init__(**kwargs)
        self.channel_ratio = channel_ratio
        self.spatial_kernel = spatial_kernel
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.channel_att = ChannelAttention(
            ratio=self.channel_ratio,
            kernel_initializer=self.kernel_initializer
        )
        self.spatial_att = SpatialAttention(
            kernel_size=self.spatial_kernel,
            kernel_initializer=self.kernel_initializer
        )
        super(CBAMAttention, self).build(input_shape)

    def call(self, inputs):
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x

    def get_config(self):
        config = super(CBAMAttention, self).get_config()
        config.update({
            'channel_ratio': self.channel_ratio,
            'spatial_kernel': self.spatial_kernel,
            'kernel_initializer': self.kernel_initializer
        })
        return config