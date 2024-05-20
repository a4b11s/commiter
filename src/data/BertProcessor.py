import tensorflow as tf
import keras_nlp
import re

from tensorflow import strings as tf_strings

from data import DataProcessor


class BertProcessor(DataProcessor):
    def __init__(
        self,
        sequence_length: int,
        vocab_size: int | None = None,
        bert_tokenizer_preset: str = "bert_small_en_uncased",
    ):
        self.bert_tokenizer = keras_nlp.models.BertTokenizer.from_preset(
            bert_tokenizer_preset
        )

        self.bert_tokenizer.sequence_length = sequence_length

        self.vocab_size: int = self.bert_tokenizer.vocabulary_size()

        self.tokens: list[str] = ["[PAD]"]

    def preprocess(self, text: str | bytes | tf.Tensor) -> str:
        if tf.is_symbolic_tensor(text):
            return tf_strings.lower(text)

        if isinstance(text, bytes):
            text = str(text, "utf-8")

        return str.lower(text)

    def postprocess(self, text: str | bytes) -> str:
        if not isinstance(text, str):
            text = str(text, "utf-8")

        tokens_regex = re.compile("|".join(self.tokens))

        removed_tokens = re.sub(tokens_regex, "", text)

        return str.strip(removed_tokens)

    def tokenize(self, text: str | bytes) -> tf.Tensor:
        return self.bert_tokenizer.tokenize(text)

    def detokenize(self, sequence: tf.Tensor) -> str:
        detokenized: tf.Tensor = self.bert_tokenizer.detokenize(sequence)

        return detokenized.numpy().decode("utf-8")
