import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class Embedding(Layer):
    def __init__(self, model_dim, vocab_size, **kwargs):
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        super(Embedding, self).__init__(**kwargs)

    # 采用随机初始化方法初始化一个和输入大小一样的embedding
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.vocab_size, self.model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)

    # 将输入的token转换成embedding，同时做scale
    def call(self, token, scale=0.5):
        # 转换类型
        if K.dtype(token) != "int32":
            token = K.cast(token, "int32")
        # 按token取embedding对应行
        print('embedding:')
        print(self.embeddings)
        # print(token)
        embedding = K.gather(self.embeddings, token)
        embedding = embedding * (self.model_dim ** scale)
        print(embedding)
        return embedding

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model_dim': self.model_dim,
            'vocab_size': self.vocab_size
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape + (self._model_dim,)


class attentionLayer(Layer):
    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(attentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(attentionLayer, self).get_config().copy()
        config.update(
            {'_masking': self._masking,
             '_future': self._future,
             '_dropout_rate': self._dropout_rate,
             '_masking_num': self._masking_num
             }
        )
        return config

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs

    def padding_mask(self, QK):
        padding = tf.cast(tf.equal(QK, 0), tf.float32)
        padding *= self._masking_num
        return QK + padding

    # sequence mask(传说中的下三角)
    def sequence_mask(self, QK):
        # 初始化下三角矩阵
        seq_mask = 1 - tf.linalg.band_part(tf.ones_like(QK), -1, 0)
        seq_mask *= self._masking_num
        return QK + seq_mask

    def call(self, inputs, **kwargs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5
        if self._masking:
            scaled_matmul = self.padding_mask(scaled_matmul)
        if self._future:
            scaled_matmul = self.sequence_mask(scaled_matmul)
        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        out = K.dropout(softmax_out, self._dropout_rate)
        outputs = K.batch_dot(out, values)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):
    def __init__(self, head, head_dim, dropout_rate=0.1, masking=False, future=False, trainable=True, **kwargs):
        self.head = head
        self.head_dim = head_dim
        self.masking = masking
        self._dropout_rate = dropout_rate
        self._future = future
        self.trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config().copy()
        config.update({
            'head': self.head,
            'head_dim': self.head_dim,
            'masking': self.masking,
            '_dropout_rate': self._dropout_rate,
            '_future': self._future,
            'trainable': self.trainable
        })
        return config

    def build(self, input_shape):
        self.weight_query = self.add_weight(shape=(input_shape[0][-1], self.head * self.head_dim),
                                            initializer='glorot_uniform', trainable=self.trainable,
                                            name='weights_querys')
        self.weight_key = self.add_weight(shape=(input_shape[1][-1], self.head * self.head_dim),
                                          initializer='glorot_uniform', trainable=self.trainable, name='weights_keys')
        self.weight_value = self.add_weight(shape=(input_shape[2][-1], self.head * self.head_dim),
                                            initializer='glorot_uniform', trainable=self.trainable,
                                            name='weights_values')

    def call(self, inputs, *args, **kwargs):
        if self.masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs
        queries_linear = K.dot(queries, self.weight_query)
        keys_linear = K.dot(keys, self.weight_key)
        values_linear = K.dot(values, self.weight_value)
        queries_multi_head = tf.concat(tf.split(queries_linear, self.head, axis=2), axis=0)
        keys_multi_head = tf.concat(tf.split(keys_linear, self.head, axis=2), axis=0)
        values_multi_head = tf.concat(tf.split(values_linear, self.head, axis=2), axis=0)
        if self.masking:
            att_inputs = [queries_multi_head, keys_multi_head, values_multi_head, masks]
        else:
            att_inputs = [queries_multi_head, keys_multi_head, values_multi_head]
        attention = attentionLayer(
            masking=self.masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)
        outputs = tf.concat(tf.split(att_out, self.head, axis=0), axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
