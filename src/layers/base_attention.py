from keras import layers


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()

    def build(self, input_shape):
        # The input_shape should be a list or tuple of shapes: [query_shape, key_shape, value_shape]
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape

        self.mha.build(query_shape, key_shape, value_shape)
        self.layernorm.build(query_shape)
        self.add.build([query_shape, query_shape])
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # The default behavior for this base class is to pass through the mask unchanged.
        # This can be overridden in subclasses if different mask handling is required.
        return mask

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def compute_output_spec(self, input_spec):
        if isinstance(input_spec, list):
            return input_spec[0]
        return input_spec
