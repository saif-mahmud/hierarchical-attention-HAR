import tensorflow as tf
from model.modaility_encoder import ModalityEncoderBlock
from model.combined_sensor_attention import CombinedSensorSelfAttention
from model.multiwindow_encoder import MultiWindowEncoder



class HSA_model():
    def __init__(self, modality_indices, n_window, n_timesteps, n_features, n_outputs, dff, d_model, num_heads, dropout_rate):
        super().__init__()
        self.modality_indices = modality_indices
        self.n_window, self.n_timesteps, self.n_features, self.n_outputs = n_window, n_timesteps, n_features, n_outputs
        self.dff, self.d_model, self.num_heads, self.dropout_rate = dff, d_model, num_heads, dropout_rate


    def get_model(self):
        inputs = tf.keras.layers.Input(shape=(self.n_window, self.n_timesteps, self.n_features,))
        x, _ = MultiWindowEncoder(self.modality_indices, self.n_window, self.n_timesteps,
                                self.n_features, self.d_model, self.num_heads, self.dff, dropout_rate=0.1)(inputs)
        x = ModalityEncoderBlock(n_timesteps=self.n_window, d_model=self.d_model, num_heads=self.num_heads,
                                dff=self.dff, num_sa_blocks=2, dropout_rate=self.dropout_rate)(x)
        x, _ = CombinedSensorSelfAttention(self.d_model, 1, self.dff, self.dropout_rate, concat=False)(x)
        
        predictions = tf.keras.layers.Dense(self.n_outputs, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        
        return model
