import tensorflow as tf


@tf.function
def zeros_pad_to_maxlen(val, maxlen):
    padding = tf.zeros([maxlen - tf.shape(val)[0]], dtype=tf.float32)
    return tf.concat([val, padding], axis=0)


@tf.function
def seq_to_input(x, y, maxlen):
    # Cast x and y to float32
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    # Ensure x and y are tensors
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # Create padded encoder input
    padded_encoder_batch = zeros_pad_to_maxlen(x, maxlen)

    # Create sequences for decoder input and target batches
    decoder_batch = tf.TensorArray(dtype=tf.float32, size=maxlen)
    target_batch = tf.TensorArray(dtype=tf.float32, size=maxlen)

    for i in tf.range(maxlen):
        decoder_seq = zeros_pad_to_maxlen(y[:i], maxlen)
        target_seq = zeros_pad_to_maxlen(y[: i + 1], maxlen)
        decoder_batch = decoder_batch.write(i, decoder_seq)
        target_batch = target_batch.write(i, target_seq)

    padded_decoder_batch = decoder_batch.stack()
    padded_target_batch = target_batch.stack()

    # Expand dimensions of encoder input to match the batch size
    padded_encoder_batch = tf.expand_dims(padded_encoder_batch, axis=0)
    padded_encoder_batch = tf.tile(padded_encoder_batch, [maxlen, 1])

    # Combine encoder input with decoder input for the dataset
    y_batch = tf.stack([padded_encoder_batch, padded_decoder_batch], axis=1)

    # Convert to tensors
    y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
    padded_target_batch = tf.convert_to_tensor(padded_target_batch, dtype=tf.float32)

    # Combine batches into a dataset
    dataset = tf.data.Dataset.from_tensor_slices((y_batch, padded_target_batch))

    return dataset
