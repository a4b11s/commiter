import tensorflow as tf

from tensorflow import data as tf_data
from typing import Tuple

from .data_processor import DataProcessor
from .bert_processor import BertProcessor
from .text_vectorization_processor import TextVectorizationProcessor

from utils import seq_to_input


class DataPipline:
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        pipline_buffer: int,
        repeat_count: int,
        sequence_length: int,
        input_vocab_size: int | None = None,
        path_to_vocab: str | None = None,
        adapt_steps: int = 5000,
    ):
        """
        Initializes the DataPipeline object.

        Args:
            dataset_path (str): The path to the dataset file.
            batch_size (int): The number of samples per batch.
            pipline_buffer (int): The buffer size for the pipeline.
            repeat_count (int): The number of times to repeat the dataset.
            sequence_length (int): The maximum length of the input sequence.
            input_vocab_size (int | None): The size of the input vocabulary. If None, it will be automatically determined.
            path_to_vocab (str | None): The path to the input vocabulary file. If None, it will be automatically determined.
            adapt_steps (int): The number of steps for adaptive tokenization.

        Attributes:
            dataset_path (str): The path to the dataset file.
            batch_size (int): The number of samples per batch.
            pipline_buffer (int): The buffer size for the pipeline.
            repeat_count (int): The number of times to repeat the dataset.
            sequence_length (int): The maximum length of the input sequence.
            input_vocab_size (int | None): The size of the input vocabulary. If None, it will be automatically determined.
            path_to_vocab (str | None): The path to the input vocabulary file. If None, it will be automatically determined.
            adapt_steps (int): The number of steps for adaptive tokenization.
            input_processor (DataProcessor): The input data processor.
            output_processor (DataProcessor): The output data processor.
            dataset (tf_data.Dataset): The prepared dataset.
        """

        self.dataset_path: str = dataset_path
        self.batch_size: int = batch_size
        self.pipline_buffer: int = pipline_buffer
        self.repeat_count: int = repeat_count

        self.sequence_length: int = sequence_length
        self.input_vocab_size: int = input_vocab_size
        self.path_to_vocab: str = path_to_vocab
        self.adapt_steps: int = adapt_steps

        self._create_preprocessors()

        self.dataset = self._prepare_dataset()

    def get_dataset(
        self,
        validation_size: int | None = None,
    ) -> Tuple[tf_data.Dataset, tf_data.Dataset] | tf_data.Dataset:
        """
        Returns the dataset for training and validation.

        Args:
            validation_size (int | None): The size of the validation set. If None, the dataset will be repeated for training.

        Returns:
            Tuple[tf_data.Dataset, tf_data.Dataset] | tf_data.Dataset: A tuple containing the training and validation datasets, or a single dataset for training if validation_size is None.
        """
        if validation_size is None:
            return self.dataset.repeat(self.repeat_count).prefetch(self.pipline_buffer)

        val_data = self.dataset.take(validation_size).prefetch(self.pipline_buffer)

        train_data = (
            self.dataset.skip(validation_size)
            .repeat(self.repeat_count)
            .prefetch(self.pipline_buffer)
        )

        return (train_data, val_data)

    def _create_preprocessors(self):
        input_dataset = tf_data.experimental.CsvDataset(
            self.dataset_path,
            record_defaults=[tf.string],
            buffer_size=self.pipline_buffer,
            select_cols=[0],
        )

        input_processor: DataProcessor = TextVectorizationProcessor(
            sequence_length=self.sequence_length,
            vocab_size=self.input_vocab_size,
            vocab_path=self.path_to_vocab,
            dataset=input_dataset,
            adapt_steps=self.adapt_steps,
            verbose=True,
        )
        output_processor: DataProcessor = BertProcessor()

        self.input_processor: DataProcessor = input_processor
        self.output_processor: DataProcessor = output_processor

    def _prepare_dataset(self) -> tf_data.Dataset:
        @tf.function
        def tokenize(x, y):
            x = self.input_processor.preprocess(x)
            x = self.input_processor.tokenize(x)
            y = self.output_processor.preprocess(y)
            y = self.output_processor.tokenize(y)

            return x, y

        dataset = tf_data.experimental.CsvDataset(
            self.dataset_path,
            record_defaults=[tf.string, tf.string],
            buffer_size=self.pipline_buffer,
            select_cols=[0, 1],
        )

        dataset = dataset.map(tokenize, num_parallel_calls=tf_data.AUTOTUNE)

        dataset = dataset.flat_map(
            lambda x, y: seq_to_input(x, y, self.sequence_length)
        )

        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf_data.AUTOTUNE,
        )

        return dataset
