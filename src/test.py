import tensorflow as tf

from data import DataPipline
from models import Transformer
from modules import TransformerGenerator
from config import Config


def test():
    num_heads = Config.config["num_heads"]
    num_layers = Config.config["num_layers"]
    d_model = Config.config["d_model"]
    dff = Config.config["dff"]
    dropout_rate = Config.config["dropout_rate"]

    batch_size = Config.config["batch_size"]
    dataset_path = Config.config["dataset_path"]
    pipline_buffer = Config.config["pipline_buffer"]

    path_to_vocab = Config.config["path_to_vocab"]
    input_vocab_size = Config.config["input_vocab_size"]
    adapt_steps = Config.config["adapt_steps"]
    sequence_length = Config.config["sequence_length"]

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
    
    transformer.load_weights(Config.config["chp_path"])

    dataset = data_pipline.get_dataset()
    x, _ = next(dataset.as_numpy_iterator())
    transformer_generator: TransformerGenerator = TransformerGenerator(transformer, 0, 0, 300)

    start_token = tf.zeros(shape=(batch_size,), dtype=tf.int64)

    output = transformer_generator(x[0], start_token, 300)
    
    for seq in output:
        print(data_pipline.output_processor.detokenize(seq))
        print("***")
