import tensorflow as tf
import numpy as np

model_hsa = None # trained hierarchical-attention-model

window_model = tf.keras.Model(inputs=model_hsa.input, outputs=model_hsa.get_layer("multi_window_encoder").output)
session_model = tf.keras.Model(inputs=model_hsa.input, outputs=model_hsa.get_layer("combined_sensor_self_attention_1").output)

# Obtain weights on data
X_test = None
_, w_out = window_model.predict(X_test)
_, s_out = session_model.predict(X_test)

# Save attention weights
np.save('dataset_benm_window.npy', w_out)
np.save('dataset_benm_session.npy', s_out)