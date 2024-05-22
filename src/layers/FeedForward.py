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

    def build(self, input_shape):
        self.seq.build(input_shape)
        self.add.build([input_shape, input_shape])
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, input_spec):
        return input_spec
