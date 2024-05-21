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


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

    def compute_mask(self, inputs, mask=None):
        # If mask is provided for the context, we need to propagate it through the network.
        # Typically, this means returning the mask for the query tensor.
        if mask is not None and isinstance(mask, list) and len(mask) == 2:
            return mask[0]
        return mask


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
