from .base_attention import BaseAttention


class CrossAttention(BaseAttention):
    def call(self, x, context, mask=None):
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

    def compute_output_shape(self, x_shape, context_shape):
        attn_output_shape = self.mha.compute_output_shape(x_shape, context_shape, context_shape)
        add_output_shape = self.add.compute_output_shape([x_shape, attn_output_shape])
        layernorm_output_shape = self.layernorm.compute_output_shape(add_output_shape)
        return layernorm_output_shape