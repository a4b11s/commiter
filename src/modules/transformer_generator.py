import tensorflow as tf


class TransformerGenerator(tf.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer_model = transformer_model

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int64),
            tf.TensorSpec(shape=[None], dtype=tf.int64),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        ]
    )
    def generate(self, context, start_token, max_length):
        """
        Generate a sequence given the initial context and start token.

        Args:
            context: Tensor of shape (batch_size, context_len) - the input context.
            start_token: Tensor of shape (batch_size,) - the initial token to start the generation.
            max_length: int - the maximum length of the generated sequence.

        Returns:
            Tensor of shape (batch_size, max_length) - the generated sequence.
        """
        # Initialize the generated sequence with the start token
        generated_sequence = tf.TensorArray(dtype=tf.int64, size=max_length)
        generated_sequence = generated_sequence.write(0, start_token)

        # Initialize input_sequence with start_token
        input_sequence = tf.expand_dims(start_token, 1)  # (batch_size, 1)

        for i in tf.range(1, max_length):
            # Combine encoded_context and input_sequence for the call method
            inputs = tf.concat(
                [
                    tf.expand_dims(context, axis=1),
                    tf.expand_dims(input_sequence, axis=1),
                ],
                axis=1,
            )

            # Pass the combined input through the transformer
            logits = self.transformer_model.call(inputs)

            # Get the logits for the last position
            next_token_logits = logits[:, -1, :]  # (batch_size, target_vocab_size)

            # Predict the next token
            next_token = tf.argmax(
                next_token_logits, axis=-1, output_type=tf.int64
            )  # (batch_size,)

            # Append the next token to the input sequence
            input_sequence = tf.concat(
                [input_sequence, tf.expand_dims(next_token, axis=1)], axis=-1
            )  # (batch_size, i+1)

            # Write the next token to the generated sequence
            generated_sequence = generated_sequence.write(i, next_token)

        # Convert the generated sequence to a tensor
        generated_sequence = tf.transpose(
            generated_sequence.stack()
        )  # (batch_size, max_length)

        return generated_sequence
