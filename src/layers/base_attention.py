from keras import layers

import tensorflow as tf


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()

    def compute_output_shape(self, input_shape):
        attn_output_shape = self.mha.compute_output_shape(
            input_shape, input_shape, input_shape
        )
        add_output_shape = self.add.compute_output_shape(
            [input_shape, attn_output_shape]
        )
        layernorm_output_shape = self.layernorm.compute_output_shape(add_output_shape)
        return layernorm_output_shape
