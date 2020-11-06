import tensorflow as tf
from model.modality_encoder import ModalityEncoderBlock
from model.combined_sensor_attention import CombinedSensorSelfAttention
from model.multiwindow_encoder import MultiWindowEncoder
from model.variational_autoencoder import VariationalAutoEncoder


class HSA_model():
    def __init__(self, modality_indices, n_window, n_timesteps, n_features, n_outputs, dff, d_model, num_heads, dropout_rate):
        super().__init__()
        self.modality_indices = modality_indices
        self.n_window, self.n_timesteps, self.n_features, self.n_outputs = n_window, n_timesteps, n_features, n_outputs
        self.dff, self.d_model, self.num_heads, self.dropout_rate = dff, d_model, num_heads, dropout_rate

    def get_model(self):
        inputs = tf.keras.layers.Input(
            shape=(self.n_window, self.n_timesteps, self.n_features,))
        x, _ = MultiWindowEncoder(self.modality_indices, self.n_window, self.n_timesteps,
                                  self.n_features, self.d_model, self.num_heads, self.dff, dropout_rate=0.1)(inputs)
        x = ModalityEncoderBlock(n_timesteps=self.n_window, d_model=self.d_model, num_heads=self.num_heads,
                                 dff=self.dff, num_sa_blocks=2, dropout_rate=self.dropout_rate)(x)
        x, _ = CombinedSensorSelfAttention(
            self.d_model, 1, self.dff, self.dropout_rate, concat=False)(x)

        predictions = tf.keras.layers.Dense(
            self.n_outputs, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model


class HSA_model_two_level():
    def __init__(self, modality_indices, n_window, n_timesteps, n_features, n_outputs, dff, d_model, num_heads, dropout_rate, n_outputs_window=18):
        super().__init__()
        self.modality_indices = modality_indices
        self.n_window, self.n_timesteps, self.n_features = n_window, n_timesteps, n_features
        self.n_outputs, self.n_outputs_window = n_outputs, n_outputs_window
        self.dff, self.d_model, self.num_heads, self.dropout_rate = dff, d_model, num_heads, dropout_rate

    def get_model(self):
        inputs = tf.keras.layers.Input(
            shape=(self.n_window, self.n_timesteps, self.n_features,))
        x, _ = MultiWindowEncoder(self.modality_indices, self.n_window, self.n_timesteps,
                                  self.n_features, self.d_model, self.num_heads, self.dff, dropout_rate=0.1)(inputs)
        window_prediction = tf.keras.layers.Dense(
            self.n_outputs_window, activation='softmax', name='window_pred')(x)
        x = ModalityEncoderBlock(n_timesteps=self.n_window, d_model=self.d_model, num_heads=self.num_heads,
                                 dff=self.dff, num_sa_blocks=2, dropout_rate=self.dropout_rate)(x)
        x, _ = CombinedSensorSelfAttention(
            self.d_model, 1, self.dff, self.dropout_rate, concat=False)(x)

        predictions = tf.keras.layers.Dense(
            self.n_outputs, activation='softmax', name='session_pred')(x)
        model = tf.keras.Model(inputs=inputs, outputs=[window_prediction, predictions])
        return model

    def get_compiled_model(self, lr=0.001):
        model = self.get_model()
        losses = {'session_pred': tf.keras.losses.CategoricalCrossentropy() , 'window_pred':tf.keras.losses.CategoricalCrossentropy() }
        model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
        return model


class HSA_model_session_guided_window():
    def __init__(self, modality_indices, n_window, n_timesteps, n_features, n_outputs, dff, d_model, num_heads, dropout_rate, n_outputs_window=18):
        super().__init__()
        self.modality_indices = modality_indices
        self.n_window, self.n_timesteps, self.n_features = n_window, n_timesteps, n_features
        self.n_outputs, self.n_outputs_window = n_outputs, n_outputs_window
        self.dff, self.d_model, self.num_heads, self.dropout_rate = dff, d_model, num_heads, dropout_rate

    def get_model(self):
        inputs = tf.keras.layers.Input(
            shape=(self.n_window, self.n_timesteps, self.n_features,))
        window_repr, _ = MultiWindowEncoder(self.modality_indices, self.n_window, self.n_timesteps,
                                  self.n_features, self.d_model, self.num_heads, self.dff, dropout_rate=0.1)(inputs)
        x = ModalityEncoderBlock(n_timesteps=self.n_window, d_model=self.d_model, num_heads=self.num_heads,
                                 dff=self.dff, num_sa_blocks=2, dropout_rate=self.dropout_rate)(window_repr)
        session_repr, _ = CombinedSensorSelfAttention(
            self.d_model, 1, self.dff, self.dropout_rate, concat=False)(x)
        session_repeated = tf.keras.layers.Reshape((self.n_window, self.d_model)) (tf.repeat(session_repr, self.n_window, axis=0))
        window_session_combined = tf.keras.layers.Concatenate(axis=-1) ([window_repr, session_repeated])
        window_prediction = tf.keras.layers.Dense(
            self.n_outputs_window, activation='softmax', name='window_pred')(window_session_combined)
        session_prediction = tf.keras.layers.Dense(
            self.n_outputs, activation='softmax', name='session_pred')(session_repr)
        model = tf.keras.Model(inputs=inputs, outputs=[window_prediction, session_prediction])
        return model

    def get_compiled_model(self, lr=0.001):
        model = self.get_model()
        losses = {'session_pred': tf.keras.losses.CategoricalCrossentropy() , 'window_pred':tf.keras.losses.CategoricalCrossentropy() }
        model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
        return model


class HSA_VAE():
    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim

        self.hsa_model = tf.keras.Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(
            'combined_sensor_self_attention_1').output, name='hierarchical_encoder')
        self.hsa_model.trainable = False

    def get_model(self):
        hsa_vae = VariationalAutoEncoder(
            base_model=self.hsa_model, original_dim=self.feature_dim)

        return hsa_vae
