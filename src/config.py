from utils.ConfigCreator import ConfigCreator

vectorizer_config = ["vocab_size", "vocab_path", "maxlen", "adapt_steps"]


config_field = [
    "embed_dim",
    "num_heads",
    "feed_forward_dim",
    "batch_size",
    "dataset_path",
    "pipline_buffer",
    "path_to_vocab",
    "input_sequence_length",
    "input_vocab_size",
    "adapt_steps",
    "output_sequence_length",
]

default_config = {
    "embed_dim": 256,
    "num_heads": 16,
    "feed_forward_dim": 512,
    "batch_size": 16,
    "dataset_path": "/ds/cleaned_output.csv",
    "pipline_buffer": 64,
    "path_to_vocab": "/t_cfg/input_vocab.pkl",
    "input_sequence_length": 1000,
    "input_vocab_size": 300,
    "adapt_steps": 100000,
    "output_sequence_length": 100,
}

Config = ConfigCreator(config_field=config_field, default_config=default_config)
