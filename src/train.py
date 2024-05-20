import tensorflow as tf
import keras

from tensorflow import data as tf_data

from data import BertProcessor, TextVectorizationProcessor, DataProcessor
from models import Transformer
from losses import masked_loss
from metrics import masked_accuracy
from utils import CustomSchedule
from config import Config

def train_model():
    batch_size = Config.config['batch_size']
    dataset_path = Config.config['dataset_path']
    pipline_buffer = Config.config['pipline_buffer']
    path_to_vocab = Config.config['path_to_vocab']
    input_vocab_size = Config.config['input_vocab_size']
    adapt_steps = Config.config['adapt_steps']
    input_sequence_length = Config.config['input_sequence_length']

    # Prepare dataset
    dataset = tf_data.experimental.CsvDataset(
        dataset_path,
        record_defaults=[tf.string, tf.string],
        buffer_size=pipline_buffer,
        select_cols=[0, 1],
    )
    input_dataset = dataset.map(lambda x, _: x, num_parallel_calls=tf_data.AUTOTUNE)
    input_processor: DataProcessor = TextVectorizationProcessor(
        sequence_length=input_sequence_length,
        vocab_size=input_vocab_size,
        vocab_path=path_to_vocab,
        dataset=input_dataset,
        adapt_steps=adapt_steps,
        verbose=True,
    )
    output_processor: DataProcessor = BertProcessor(
        sequence_length=input_sequence_length,
    )

    def tokenize(x, y):
        x = input_processor.tokenize(x)
        y = output_processor.preprocess(y)
        y = output_processor.tokenize(y)

        return x, y

    dataset = dataset.map(tokenize, num_parallel_calls=tf_data.AUTOTUNE)
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

    callbacks = [
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

    print(f'trainable_variables: {len(transformer.trainable_variables)}')

    transformer.fit(
        dataset,
        verbose=1,
        steps_per_epoch=5000,
        epochs=1000,
        callbacks=callbacks,
    )

    print("Training finished")
