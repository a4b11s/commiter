class DataProcessor:
    def __init__(self, sequence_length, vocab_size):
        pass

    def preprocess(self, text):
        return text

    def postprocess(self, text):
        return text

    def tokenize(self, text):
        raise NotImplementedError()

    def detokenize(self, text):
        raise NotImplementedError()
