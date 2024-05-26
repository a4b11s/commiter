import keras
from .causal_self_attention import CausalSelfAttention
from .cross_attention import CrossAttention
from .feed_forward import FeedForward


class DecoderLayer(keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

    def build(self, input_shape):
        self.causal_self_attention.build(input_shape)
        self.cross_attention.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # The causal self-attention layer should receive the mask
        return self.causal_self_attention.compute_mask(inputs, mask)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

    def compute_output_spec(self, input_spec):
        # Output spec is the same as input spec
        return input_spec
