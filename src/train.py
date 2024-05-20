import tensorflow as tf
import keras
import numpy as np

from tensorflow import data as tf_data

from models.Transformer import Transformer
from data.DataProcessor import BertProcessor, TextVectorizationProcessor
from utils.CustomSchedule import CustomSchedule
from callbacks.TextGenerator import TextGenerator

from config import Config


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def create_subsequences(x, y):
    # Створення послідовностей для декодера та міток
    decoder_inputs = [tf.concat([y[:i]], axis=0) for i in range(1, y.shape[0] + 1)]
    targets = [y[:i] for i in range(1, y.shape[0] + 1)]

    # Знаходимо максимальну довжину
    max_len_dec = max(len(seq) for seq in decoder_inputs)
    max_len_tgt = max(len(seq) for seq in targets)

    # Виконуємо паддінг послідовностей до максимального розміру
    decoder_inputs = [
        tf.pad(seq, [[0, max_len_dec - tf.shape(seq)[0]]]) for seq in decoder_inputs
    ]
    targets = [tf.pad(seq, [[0, max_len_tgt - tf.shape(seq)[0]]]) for seq in targets]

    # Перетворення списків у датасет і додавання вхідного тензора для енкодера до кожного елементу
    subsequences = tf.data.Dataset.from_tensor_slices((decoder_inputs, targets))
    subsequences = subsequences.map(
        lambda dec_in, tgt: ({"encoder_input": x, "decoder_input": dec_in}, tgt)
    )
    return subsequences


def transformer_map_fn(dataset):
    return dataset.flat_map(lambda x, y: create_subsequences(x, y))


# context, x


def train_model():
    (
        embed_dim,
        num_heads,
        feed_forward_dim,
        batch_size,
        dataset_path,
        pipline_buffer,
        path_to_vocab,
        input_sequence_length,
        input_vocab_size,
        adapt_steps,
        output_sequence_length,
    ) = Config.config.values()

    # Prepare dataset
    dataset = tf_data.experimental.CsvDataset(
        dataset_path,
        record_defaults=[tf.string, tf.string],
        buffer_size=pipline_buffer,
        select_cols=[0, 1],
    )
    input_dataset = dataset.map(lambda x, _: x, num_parallel_calls=tf_data.AUTOTUNE)

    input_processor = TextVectorizationProcessor(
        sequence_length=input_sequence_length,
        vocab_size=input_vocab_size,
        vocab_path=path_to_vocab,
        dataset=input_dataset,
        adapt_steps=adapt_steps,
        verbose=True,
    )
    output_processor = BertProcessor(
        sequence_length=input_sequence_length,
    )

    def tokenize(x, y):
        x = input_processor.tokenize(x)
        y = output_processor.preprocess(y)
        y = output_processor.tokenize(y)

        return x, y

    dataset = dataset.map(tokenize, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.apply(transformer_map_fn)
    callback_dataset = dataset.take(10).map(lambda x, _: x)
    dataset = dataset.batch(
        batch_size, drop_remainder=True, num_parallel_calls=tf_data.AUTOTUNE
    )
    dataset = dataset.repeat(100000)
    dataset = dataset.prefetch(buffer_size=pipline_buffer)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=input_vocab_size,
        dropout_rate=dropout_rate,
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    callbacks = [
        TextGenerator(output_processor, callback_dataset),
        keras.callbacks.ModelCheckpoint(
            "/ds/checkpoint.weights.h5", save_weights_only=True, verbose=1
        ),
    ]

    learning_rate = CustomSchedule(d_model)

    optimizer = keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer(next(dataset.take(1).as_numpy_iterator())[0])

    transformer.summary()

    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    print(len(transformer.trainable_variables))

    transformer.fit(
        dataset,
        verbose=1,
        steps_per_epoch=5000,
        epochs=1000,
        callbacks=callbacks,
    )

    print("Training finished")
