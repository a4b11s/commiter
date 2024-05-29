import tensorflow as tf
import numpy as np
import pickle
import time
import os

from tensorflow import strings as tf_strings
from keras import layers

from .data_processor import DataProcessor


class TextVectorizationProcessor(DataProcessor):
    def __init__(
        self,
        sequence_length: int,
        vocab_size: int | None = None,
        vocab_path: str | None = None,
        dataset: tf.data.Dataset | None = None,
        adapt_steps: int = 5000,
        own_tokens: list[str] = [],
        verbose: bool = False,
    ):
        self.tokenizer: layers.TextVectorization = layers.TextVectorization(
            name="input_vectorizer",
            standardize=self._standardization,
            split="character",
            output_mode="int",
            output_sequence_length=sequence_length,
            max_tokens=vocab_size,
            pad_to_max_tokens=False,
        )

        self.verbose = verbose
        self.dataset: tf.data.Dataset | None = dataset
        self.adapt_steps: int = adapt_steps
        self.vocab_loaded: bool = False
        self.vocab_path: str | None = vocab_path

        if vocab_path is not None:
            try:
                self.vocabulary_from_file()
                self.vocab_loaded = True
                if self.verbose:
                    print(f"Vocabulary loaded from {vocab_path}")
            except FileNotFoundError:
                if self.verbose:
                    print(f"File not found: {vocab_path}. Vocabulary not loaded")

        if not self.vocab_loaded:
            self.adapt()

        try:
            self.vocabulary_to_file()
        except FileNotFoundError:
            if self.verbose:
                print(f"Vocabulary not saved to {self.vocab_path}")

        if len(own_tokens) == 0:
            own_tokens = ["[BOS]", "[EOS]"]

        self.tokenizer.set_vocabulary(self.tokenizer.get_vocabulary()[:-len(own_tokens)] + own_tokens)

        self.dict: list[str] = self.tokenizer.get_vocabulary()

    def tokenize(self, text):
        return self.tokenizer(text)

    def detokenize(self, sequence: tf.Tensor | np.ndarray) -> str:
        if isinstance(sequence, tf.Tensor):
            sequence = sequence.numpy()

        text = ""

        for number in sequence:
            text += self.dict[number] + " "

        return text.strip()

    def adapt(self):
        if self.dataset is None:
            raise ValueError("Dataset is None")

        if self.verbose:
            print(f"Vectorizer adaptation started")
            print(f"Vectorizer adaptation steps: {self.adapt_steps}")

        adapt_start_time = time.time()

        self.tokenizer.adapt(self.dataset, batch_size=1024, steps=self.adapt_steps)

        if self.verbose:
            print(f"Vectorizer adaptation finished")
            print(f"Vectorizer adaptation time: {time.time() - adapt_start_time}")
            print(f"Vocabulary size: {len(self.tokenizer.get_vocabulary())}")

    def vocabulary_to_file(self):
        if not os.path.exists(os.path.dirname(self.vocab_path)):
            raise FileNotFoundError(
                f"Directory not found: {os.path.dirname(self.vocab_path)}"
            )

        vocabulary: list[str] = self.tokenizer.get_vocabulary()

        with open(self.vocab_path, "wb") as f:
            pickle.dump(vocabulary, f)

    def vocabulary_from_file(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"File not found: {self.vocab_path}")

        with open(self.vocab_path, "rb") as f:
            self.tokenizer.set_vocabulary(pickle.load(f))

    @staticmethod
    def _standardization(text):
        linebrake_tokinazed_string = tf_strings.regex_replace(text, "\n", "[NEWLINE]")
        return linebrake_tokinazed_string

    @staticmethod
    def _split(input_string):
        tokens = ["[NEWLINE]"]
        chars = []
        splitted = list(input_string)

        i = 0

        while i < len(splitted):
            cursor = splitted[i]

            if cursor == "[":
                try:
                    token_end_index = splitted.index("]", i)

                    token_candidate = "".join(splitted[i : token_end_index + 1])

                    if token_candidate in tokens:
                        chars.append(token_candidate)
                        i = token_end_index + 1
                        continue

                except ValueError:
                    print("Token not found")
                    pass

            chars.append(splitted[i])

            i += 1

        return chars
