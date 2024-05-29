from keras import Model, layers
from layers import Encoder, Decoder
import tensorflow as tf


class Transformer(Model):
    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the Transformer model.

        Args:
        num_layers (int): The number of layers in the encoder and decoder.
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        dff (int): The dimension of the feed-forward network.
        input_vocab_size (int): The size of the input vocabulary.
        target_vocab_size (int): The size of the target vocabulary.
        dropout_rate (float, optional): The dropout rate for the model. Defaults to 0.1.

        Attributes:
        encoder (Encoder): The initialized encoder layer.
        decoder (Decoder): The initialized decoder layer.
        final_layer (layers.Dense): The initialized final dense layer.
        d_model (int): The dimension of the model.
        target_vocab_size (int): The size of the target vocabulary.
        """
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

        self.d_model = d_model
        self.target_vocab_size = target_vocab_size

    def call(self, inputs):
        """
        This method is the main entry point for the Transformer model.
        It takes the input tensor and processes it through the encoder and decoder layers,
        finally passing the decoder output through the final dense layer.

        Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, 2, seq_len).

        Returns:
        tf.Tensor: Output tensor of shape (batch_size, target_len, target_vocab_size).

        Raises:
        AttributeError: If the `_keras_mask` attribute is not found in the logits tensor.

        Note:
        - The input tensor is split into two parts: 'context' and 'x'.
        - The 'context' tensor is passed through the encoder layer.
        - The 'x' tensor is passed through the decoder layer, with the 'context' tensor as additional input.
        - The output tensor from the decoder layer is passed through the final dense layer.
        - The '_keras_mask' attribute is removed from the logits tensor.
        """

        # Assume inputs have shape (batch_size, 2, seq_len)
        # Split the inputs into context and x along the second axis
        context, x = tf.split(inputs, num_or_size_splits=2, axis=1)

        # Remove the extra dimension after the split
        context = tf.squeeze(context, axis=1)
        x = tf.squeeze(x, axis=1)

        # Pass the context and x through the encoder and decoder, respectively.
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Pass the decoder output through the final dense layer.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        # Remove the extra dimension after the final dense layer.
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the logits.
        return logits  # logits
