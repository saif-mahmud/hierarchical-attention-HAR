import tensorflow as tf
from model.multiheaded_self_attention import MultiHeadAttention


class AggregateAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(AggregateAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.d_model = d_model
        self.query = self.add_weight("learned_query", shape=[1, 1, self.d_model], initializer=tf.keras.initializers.Orthogonal())

    def call(self, v, k):
        batched_query = tf.tile(self.query, [tf.shape(v)[0], 1, 1])
        output, attention_weights = self.mha(v, k, batched_query, mask=None)
        output = tf.squeeze(output, axis=1)
        attention_weights = tf.squeeze(attention_weights, axis=2)
        return output, attention_weights
