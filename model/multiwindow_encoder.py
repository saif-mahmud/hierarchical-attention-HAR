import tensorflow as tf
from model.modality_encoder import ModalityEncoderBlock
from model.combined_sensor_attention import CombinedSensorSelfAttention


def get_modality_encoder(modality_indices, n_timesteps, n_features, dff=512, d_model=128, num_heads=4, dropout_rate=0.2):
    num_modality = len(modality_indices)-1
    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features,))
    attn_scores = []
    modality_outputs = []
    for i in range(num_modality):
        modality_x = inputs[:, :, modality_indices[i]:modality_indices[i+1]]
        modality_x = ModalityEncoderBlock(n_timesteps=n_timesteps, d_model=d_model,
                                          num_heads=num_heads, dff=dff, num_sa_blocks=2, dropout_rate=dropout_rate)(modality_x)
        modality_outputs.append(modality_x)
    model = tf.keras.Model(inputs=inputs, outputs=modality_outputs)
    return model


class MultiWindowEncoder(tf.keras.layers.Layer):
    def __init__(self, modality_indices, n_window, n_timesteps, n_features, d_model, num_heads, dff, dropout_rate=0.1):
        super(MultiWindowEncoder, self).__init__()
        self.n_window = n_window
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        self.window_encoder = get_modality_encoder(
            modality_indices, n_timesteps=n_timesteps, n_features=n_features, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
        self.combined_sensor_attn = CombinedSensorSelfAttention(
            d_model, 1, dff, dropout_rate, concat=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        feature_dim = tf.shape(x)[-1]
        # all sessions in batch dim
        x = tf.reshape(x, (-1, self.n_timesteps, feature_dim))
        x = self.window_encoder(x)
        x, attn_scores = self.combined_sensor_attn(x)
        x = tf.reshape(x, (batch_size, -1, self.d_model))
        attn_scores = tf.reshape(
            attn_scores, (batch_size, -1, self.n_timesteps))
        return x, attn_scores
