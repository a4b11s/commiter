import tensorflow as tf


class TransformerGenerator(tf.Module):
    def __init__(
        self, transformer_model, start_token_idx, end_token_idx, max_sequence_length=100
    ):
        """
        Initialize the Sequence Generator module.

        Args:
        transformer_model (Transformer): The trained Transformer model.
        start_token_idx (int): Index of the start token in the vocabulary.
        end_token_idx (int): Index of the end token in the vocabulary.
        max_sequence_length (int): Maximum length of the generated sequence. Defaults to 100.
        """
        super().__init__()
        self.transformer_model = transformer_model
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.max_sequence_length = max_sequence_length

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def __call__(self, seed_input):
        """
        Generate a sequence given a seed input.

        Args:
        seed_input (tf.Tensor): Seed input tensor of shape (batch_size, seq_len).

        Returns:
        tf.Tensor: Generated sequence tensor of shape (batch_size, seq_len).
        """
        batch_size = tf.shape(seed_input)[0]
        current_input = seed_input

        for _ in tf.range(self.max_sequence_length):
            # Pass the current input through the transformer model
            output = self.transformer_model(current_input)

            # Take the last token from the output
            last_token = output[:, -1:, :]

            # Sample the next token probabilities
            next_token_probs = tf.squeeze(last_token, axis=1)

            # Sample the next token indices
            next_token_idx = tf.random.categorical(
                next_token_probs, num_samples=1, dtype=tf.int32
            )

            # Concatenate the next token indices to the current input
            current_input = tf.concat([current_input, next_token_idx], axis=-1)

            # Check if the end token is generated for all sequences
            if tf.reduce_all(tf.equal(next_token_idx, self.end_token_idx)):
                break

        # Remove the start token from the generated sequences
        generated_sequences = current_input[:, 1:]

        return generated_sequences
