from keras import layers, Sequential


class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = Sequential(
            [
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model),
                layers.Dropout(dropout_rate),
            ]
        )
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

    def compute_mask(self, inputs, mask=None):
        seq_mask = self.seq.compute_mask(inputs, mask)
        add_mask = self.add.compute_mask(inputs, seq_mask)
        norm_mask = self.layer_norm.compute_mask(inputs, add_mask)
        return norm_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, input_spec):
        return input_spec
