from utils import ConfigCreator

config_field = [
    "num_heads",
    "num_layers",
    "d_model",
    "dff",
    "dropout_rate",
    "batch_size",
    "element_per_epoch",
    "dataset_path",
    "pipline_buffer",
    "path_to_vocab",
    "sequence_length",
    "input_vocab_size",
    "adapt_steps",
    "chp_path",
    "tb_log_dir",
]

default_config = {
    "num_heads": 8,
    "num_layers": 2,
    "d_model": 128,
    "dff": 256,
    "dropout_rate": 0.1,
    "batch_size": 8,
    "element_per_epoch": 100_000,
    "pipline_buffer": 1_000,
    "sequence_length": 1000,
    "input_vocab_size": 300,
    "adapt_steps": 100000,
    "dataset_path": "/data/dataset/cleaned_output.csv",
    "path_to_vocab": "/data/configs/input_vocab.pkl",
    "chp_path": "/data/chp/checkpoint.weights.h5",
    "tb_log_dir": "/data/tb_logs/",
}

Config = ConfigCreator(config_field=config_field, default_config=default_config)
