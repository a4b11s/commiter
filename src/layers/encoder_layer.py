import keras
from .global_self_attention import GlobalSelfAttention
from .feed_forward import FeedForward


class EncoderLayer(keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    def build(self, input_shape):
        self.self_attention.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        attn_output_shape = self.self_attention.compute_output_shape(input_shape)
        ffn_shape = self.ffn.compute_output_shape(attn_output_shape)
        return ffn_shape
