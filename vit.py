import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class Embedding(Layer):
    def __init__(self, input_size, patch_size, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.n_patchs = int((input_size[0] * input_size[1]) / (patch_size * patch_size))
        self.embed_dim = patch_size * patch_size
        self.patch = Conv2D(filters=self.embed_dim, kernel_size=(patch_size, patch_size), strides=patch_size,
                            padding='same')
        self.reshape = Reshape((-1, self.embed_dim))
        # self.flatten = Flatten((2))
        # self.transpose = Permute((2,1))
        self.ds = Dense(self.embed_dim)

    def build(self, input_shape):
        self.embedding = self.add_weight(name='embedding',
                                         shape=(self.n_patchs, self.embed_dim),
                                         initializer='random_normal',
                                         trainable=True)
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        x = self.patch(inputs)
        # x = self.flatten(x)
        x = self.reshape(x)
        x = self.ds(x)
        # x = self.transpose(x)
        x = x + self.embedding
        return x


def transpose_layer(inputs, num_heads, head_size):
    shape = list(inputs.shape[:-1] + (num_heads, head_size))
    shape[0] = -1
    y = tf.reshape(inputs, shape)
    return tf.transpose(y, [0, 2, 1, 3])


class SelfAttention(Layer):
    def __init__(self, num_heads, embed_dim, dropout_rate, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_heads * self.head_size
        self.q = Dense(self.all_head_size)
        self.k = Dense(self.all_head_size)
        self.v = Dense(self.all_head_size)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)
        self.fc = Dense(self.all_head_size)

    def call(self, inputs):
        q = self.q(inputs)
        k = self.k(inputs)
        v = self.v(inputs)
        qw = transpose_layer(q, self.num_heads, self.head_size)
        kw = transpose_layer(k, self.num_heads, self.head_size)
        vw = transpose_layer(v, self.num_heads, self.head_size)
        attention_score = tf.matmul(qw, kw, transpose_b=True) / np.sqrt(self.head_size)
        attention_score = tf.nn.softmax(attention_score, axis=-1)
        attention_probe = self.drop1(attention_score)
        attention_probe = tf.transpose(tf.matmul(attention_probe, vw), perm=[0, 2, 1, 3])
        new_shape = list(attention_probe.shape[:-2] + [self.all_head_size])
        new_shape[0] = -1
        context_layer = tf.reshape(attention_probe, new_shape)
        output = self.fc(context_layer)
        output = self.drop2(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.all_head_size)


def fc_layer(inputs, embed_dim, hidden_dim, dropout_rate):
    x = Dense(hidden_dim, activation=tf.nn.gelu)(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(embed_dim)(x)
    return x


class FC_layer(Layer):
    def __init__(self, embed_dim, hidden_dim, dropout_rate, **kwargs):
        super(FC_layer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dense = Dense(hidden_dim, activation=tf.nn.gelu)
        self.dropout = Dropout(dropout_rate)
        self.dense2 = Dense(embed_dim)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class TransformerLayer(Layer):
    def __init__(self, num_heads, embed_dim, dropout_rate, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.atten = SelfAttention(num_heads, embed_dim, dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.fc = FC_layer(embed_dim=self.embed_dim, hidden_dim=2048, dropout_rate=self.dropout_rate)

    def call(self, inputs):
        h = inputs
        h = self.norm1(h)
        h = self.atten(h)
        h = inputs + h
        inputs = h
        h = self.norm2(h)
        h = self.fc(h)
        h = inputs + h
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_dim)


class VisionTransformer(Layer):
    def __init__(self, num_heads, embed_dim, dropout_rate, n_layers=8, patch_size=16,
                 patch_dim=768, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.emb = Embedding(input_size=(embed_dim, embed_dim, 1), patch_size=patch_size)
        self.patch_dim = patch_dim
        self.transformer = [TransformerLayer(num_heads, self.patch_dim, dropout_rate) for _ in range(self.n_layers)]

    def call(self, inputs, *args, **kwargs):
        x = self.emb(inputs)
        for layer in self.transformer:
            x = layer(x)
        return x


    def get_config(self):
        return super().get_config().update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'extract_layers': self.extract_layers,
            'dropout_rate': self.dropout_rate,
            'n_layers': self.n_layers,
            'patch_dim': self.patch_dim
        })