from tensorflow import keras
from tensorflow import data as tf_data
from data.DataProcessor import DataProcessor
import numpy as np


class TextGenerator(keras.callbacks.Callback):
    def __init__(
        self,
        data_processor: DataProcessor,
        batch,
        output_method="cmd",
        print_freq=1,
    ):
        self.data_processor: DataProcessor = data_processor

        self.batch = batch
        self.output_method = output_method
        self.print_freq = print_freq

    def output(self, prefix, text):
        if self.output_method == "cmd":
            print(f"{prefix}, generated: {text}")

    def tokenize(self, text):
        return self.x_tokenizer(text)

    def on_epoch_end(self, epoch, logs=None):
        if self.print_freq <= 0 or epoch % self.print_freq != 0:
            return

        for batch in self.batch.as_numpy_iterator():

            out = self.model.predict(
                [[0][batch]],
                verbose=0,
            )

            print(out)

            print(out.shape)
        # out = np.argmax(out, axis=2)

        # for batch in out:
        #     detokenized = self.data_processor.detokenize(batch)
        #     postprocessed = self.data_processor.postprocess(detokenized)
        #     self.output(f"Epoch {epoch+1}", postprocessed)
