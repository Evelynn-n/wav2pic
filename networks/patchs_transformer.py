import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout,Add,Layer,LayerNormalization,\
    Conv2D,LeakyReLU,BatchNormalization,Input,Reshape
class attentionLayer(Layer):
    def __init__(self, dropout_rate=0.2, **kwargs):
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(attentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(attentionLayer, self).get_config().copy()
        config.update(
            {
             '_dropout_rate': self._dropout_rate,
             '_masking_num': self._masking_num
             }
        )
        return config

    def call(self, inputs, **kwargs):

        queries, keys, values = inputs
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5
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
        self.queriesConv = Conv2D(head*head_dim,kernel_size=1)
        self.keysConv = Conv2D(head*head_dim, kernel_size=1)
        self.valuesConv = Conv2D(head*head_dim, kernel_size=1)
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

    def call(self, inputs, *args, **kwargs):
        queries, keys, values = inputs
        queries_linear = self.queriesConv(queries)
        keys_linear = self.keysConv(keys)
        values_linear = self.valuesConv(values)
        queries = Reshape((queries_linear.shape[1] * queries_linear.shape[2], self.head*self.head_dim))(queries_linear)
        keys = Reshape((keys_linear.shape[1] * keys_linear.shape[2], self.head*self.head_dim))(keys_linear)
        values = Reshape((values_linear.shape[1] * values_linear.shape[2], self.head*self.head_dim))(values_linear)
        queries_multi_head = tf.concat(tf.split(queries, self.head, axis=2), axis=0)
        keys_multi_head = tf.concat(tf.split(keys, self.head, axis=2), axis=0)
        values_multi_head = tf.concat(tf.split(values, self.head, axis=2), axis=0)
        att_inputs = [queries_multi_head, keys_multi_head, values_multi_head]
        attention = attentionLayer(dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)
        att_out = Reshape((values_linear.shape[1], values_linear.shape[2],self.head_dim))(att_out)
        outputs = tf.concat(tf.split(att_out, self.head, axis=0), axis=3)
        # outputs = tf.concat(tf.split(att_out, self.head, axis=0), axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

def FeedForwardNetwork(inputs,units_dim, model_dim):
    output = Conv2D(units_dim,1)(inputs)
    output = LeakyReLU()(output)
    output = Dense(model_dim)(output)
    return output
class EncoderLayer(Layer):
    def __init__(self,n_heads,d_model,dropout_rate=0.2,**kwargs):
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.mattnlayer = MultiHeadAttention(self.n_heads,self.d_model//self.n_heads,masking=True)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
        super(EncoderLayer, self).__init__(**kwargs)
    def call(self, inputs, *args, **kwargs):
        # masks = K.equal(inputs, 0)
        attn_output = self.mattnlayer([inputs,inputs,inputs])
        attn_output = self.dropout1(attn_output)
        attn_output = self.norm1(attn_output+inputs)
        ffn_output = FeedForwardNetwork(attn_output,self.d_model,self.d_model)
        out = self.dropout2(ffn_output)
        output = self.norm2(out+attn_output)
        output = self.dropout3(output)
        return output
    def get_config(self):
        config = super(EncoderLayer, self).get_config().copy()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads
        })
    def compute_output_shape(self, input_shape):
        return input_shape
class Encoder(Layer):
    def __init__(self,d_model,n_heads,patch_size,n_layers,**kwargs):
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.enc_layers = [
            EncoderLayer(d_model=self.d_model, n_heads=self.n_heads)
            for _ in range(self.n_layers)]
        super(Encoder, self).__init__(**kwargs)
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches
    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({
            'n_heads':self.n_heads,
            'd_model':self.d_model,
            'n_layers':self.n_layers,
        })
        return config
    def call(self, inputs, *args, **kwargs):
        des1 = Dense(self.d_model)(inputs)
        outputs = Dropout(0.2)(des1)
        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs)
        return outputs
def cbl(inputs,input_size,kernel_size,stride=1):
    convx = Conv2D(filters=input_size,kernel_size=kernel_size,strides=stride,padding="SAME")(inputs)
    bn = BatchNormalization()(convx)
    lkru = LeakyReLU()(bn)
    return lkru
def convBlock(inputs,input_size,output_size):
    if input_size==output_size:
        cbl1 = cbl(inputs,input_size/2,1)
        cbl2 = cbl(cbl1,input_size/2,3)
        cbl3 = cbl(cbl2,output_size,1)
        output = Add()([cbl3,inputs])
    else:
        cbl1 = cbl(inputs,input_size/2,1)
        cbl2 = cbl(cbl1,input_size/2,3,2)
        cbl3 = cbl(cbl2,output_size,1)
        conv = Conv2D(output_size,1,2)(inputs)
        output = Add()([cbl3,conv])
    return output

if __name__ == '__main__':
    inputs = Input((45,))
    # output = EncoderLayer(3, 45)(inputs)
    output = Encoder(118,3,45,3)(inputs)
    print(output)