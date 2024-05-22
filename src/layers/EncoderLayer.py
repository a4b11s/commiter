import keras
from layers import GlobalSelfAttention, FeedForward


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

    def compute_mask(self, inputs, mask=None):
        # Passes the mask through the self-attention layer
        return self.self_attention.compute_mask(inputs, mask)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

    def compute_output_spec(self, input_spec):
        # Output spec is the same as input spec
        return input_spec
