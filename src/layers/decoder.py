import keras
from .decoder_layer import DecoderLayer
from .positional_embedding import PositionalEmbedding


class Decoder(keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.pos_embedding.build(input_shape)
        for dec_l in self.dec_layers:
            dec_l.build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Passes the mask through the last decoder layer
        return self.dec_layers[-1].compute_mask(inputs, mask)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

    def compute_output_spec(self, input_spec):
        # Output spec is the same as input spec
        return input_spec