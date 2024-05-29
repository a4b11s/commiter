import tensorflow as tf
import tensorboard
import keras

from tensorflow import data as tf_data

from data import BertProcessor, TextVectorizationProcessor, DataProcessor
from models import Transformer
from losses import masked_loss
from metrics import masked_accuracy
from utils import CustomSchedule, seq_to_input
from config import Config


def dataset_prepare(
    dataset: tf_data.Dataset, pipline_buffer, batch_size, repeat_count=100000
):
    return (
        dataset.batch(
            batch_size, drop_remainder=True, num_parallel_calls=tf_data.AUTOTUNE
        )
        .repeat(repeat_count)
        .prefetch(buffer_size=pipline_buffer)
    )


def train_model():
    num_heads = Config.config["num_heads"]
    num_layers = Config.config["num_layers"]
    d_model = Config.config["d_model"]
    dff = Config.config["dff"]
    dropout_rate = Config.config["dropout_rate"]

    batch_size = Config.config["batch_size"]
    element_per_epoch = Config.config["element_per_epoch"]
    dataset_path = Config.config["dataset_path"]
    pipline_buffer = Config.config["pipline_buffer"]

    path_to_vocab = Config.config["path_to_vocab"]
    input_vocab_size = Config.config["input_vocab_size"]
    adapt_steps = Config.config["adapt_steps"]
    sequence_length = Config.config["sequence_length"]

    steps_per_epoch = int(element_per_epoch / batch_size)

    # Prepare dataset
    dataset = tf_data.experimental.CsvDataset(
        dataset_path,
        record_defaults=[tf.string, tf.string],
        buffer_size=pipline_buffer,
        select_cols=[0, 1],
    )
    dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf_data.AUTOTUNE)
    input_dataset = dataset.map(lambda x, _: x, num_parallel_calls=tf_data.AUTOTUNE)
    input_processor: DataProcessor = TextVectorizationProcessor(
        sequence_length=sequence_length,
        vocab_size=input_vocab_size,
        vocab_path=path_to_vocab,
        dataset=input_dataset,
        adapt_steps=adapt_steps,
        verbose=True,
    )
    output_processor: DataProcessor = BertProcessor()

    @tf.function
    def tokenize(x, y):
        x = input_processor.tokenize(x)
        y = output_processor.preprocess(y)
        y = output_processor.tokenize(y)

        return x, y

    dataset = dataset.map(tokenize, num_parallel_calls=tf_data.AUTOTUNE).cache()
    dataset = dataset.flat_map(lambda x, y: seq_to_input(x, y, sequence_length)).cache()

    dataset = dataset_prepare(dataset, pipline_buffer, batch_size)

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=output_processor.vocab_size,
        dropout_rate=dropout_rate,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            Config.config["chp_path"], save_weights_only=True, verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=Config.config["tb_log_dir"],
            profile_batch=32,
            histogram_freq=1,
            write_images=True,
            update_freq="batch",
            embeddings_freq=1,
        ),
    ]

    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    dummy_inputs = tf.random.uniform((16, 2, 1000), dtype=tf.float32)

    lg = transformer(dummy_inputs)

    print(lg.shape)

    transformer.summary(expand_nested=True)

    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    transformer.fit(
        dataset,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        epochs=1000,
        callbacks=callbacks,
    )

    print("Training finished")
