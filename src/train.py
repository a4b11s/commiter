import tensorflow as tf
import keras

from data import DataPipline
from models import Transformer
from losses import masked_loss
from metrics import masked_accuracy
from utils import CustomSchedule
from config import Config


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

    data_pipline = DataPipline(
        dataset_path=dataset_path,
        batch_size=batch_size,
        pipline_buffer=pipline_buffer,
        repeat_count=1000,
        sequence_length=sequence_length,
        input_vocab_size=input_vocab_size,
        path_to_vocab=path_to_vocab,
        adapt_steps=adapt_steps,
    )

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=data_pipline.output_processor.vocab_size,
        dropout_rate=dropout_rate,
    )

    try:
        transformer.load_weights(Config.config["chp_path"])
    except:
        print("Model not found")

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

    trian_ds, val_ds = data_pipline.get_dataset(validation_size=500)

    transformer.fit(
        trian_ds,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        epochs=1000,
        callbacks=callbacks,
        validation_data=val_ds,
    )
