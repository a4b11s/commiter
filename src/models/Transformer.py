from keras import Model, layers
from layers.Encoder import Encoder
from layers.Decoder import Decoder
import tensorflow as tf


class Transformer(Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate=0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = layers.Dense(target_vocab_size)

    @tf.function
    def call(self, inputs):
        context = inputs["encoder_input"]
        x = inputs["decoder_input"]

        context = self.encoder(context)  # (batch_size, context_len, d_model)
        tf.print("Encoder output shape:", tf.shape(context))

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        tf.print("Decoder output shape:", tf.shape(x))

        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        tf.print("Final layer output shape:", tf.shape(logits))

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
