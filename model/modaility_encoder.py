import tensorflow as tf
from model.positional_encoding import PositionalEncoding
from model.multiheaded_self_attention import SelfAttentionBlock


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class DataTransform(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(DataTransform, self).__init__()
        self.conv_1d = tf.keras.layers.Conv1D(d_model, 1, activation='relu')

    def call(self, x):
        x = self.conv_1d(x)
        return x


class ModalityEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_timesteps, d_model, num_heads, dff, num_sa_blocks=2, dropout_rate=0.1):
        super(ModalityEncoderBlock, self).__init__()
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        self.look_ahead_mask = create_look_ahead_mask(n_timesteps)
        self.data_transform = DataTransform(d_model)
        self.pe = PositionalEncoding(
            n_timesteps, d_model, dropout_rate=dropout_rate)
        self.num_sa_blocks = num_sa_blocks
        self.self_attn_blocks = [SelfAttentionBlock(
            d_model, num_heads, dff, dropout_rate) for _ in range(self.num_sa_blocks)]

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = self.data_transform(x)
        x = self.pe(x)
        for i in range(self.num_sa_blocks):
            x = self.self_attn_blocks[i](x)

        return x
