import keras
from .encoder_layer import EncoderLayer
from .positional_embedding import PositionalEmbedding

class Encoder(keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.

    def build(self, input_shape):
        super().build(input_shape)
        self.pos_embedding.build(input_shape)
        for enc_l in self.enc_layers:
            enc_l.build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Passes the mask through the last encoder layer
        return self.enc_layers[-1].compute_mask(inputs, mask)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

    def compute_output_spec(self, input_spec):
        # Output spec is the same as input spec
        return input_spec
