import tensorflow as tf
from keras import layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs[0] + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def create_encoder(maxlen, vocab_size, embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    x = transformer_block((x, x, x))
    return tf.keras.Model(inputs=inputs, outputs=x, name="encoder")


def create_decoder(maxlen, vocab_size, embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = layers.Input(shape=(maxlen,))
    enc_outputs = layers.Input(shape=(maxlen, embed_dim))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    x = transformer_block((x, enc_outputs, enc_outputs))
    return tf.keras.Model([inputs, enc_outputs], x, name="decoder")


# Параметри моделі
vocab_size = 20000  # Розмір словника
maxlen = 200  # Максимальна довжина вхідної послідовності
embed_dim = 32  # Розмірність простору векторних уявлень
num_heads = 2  # Кількість голів в багатоголовому механізмі уваги
ff_dim = 32  # Розмірність повнозв'язної (feed-forward) мережі в блоці трансформера

encoder = create_encoder(maxlen, vocab_size, embed_dim, num_heads, ff_dim)
decoder = create_decoder(maxlen, vocab_size, embed_dim, num_heads, ff_dim)

encoder_inputs = layers.Input(shape=(maxlen,))
decoder_inputs = layers.Input(shape=(maxlen,))

enc_outputs = encoder(encoder_inputs)
dec_outputs = decoder([decoder_inputs, enc_outputs])

outputs = layers.Dense(vocab_size, activation="softmax")(dec_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()
