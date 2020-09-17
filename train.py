from model.hierarchical_self_attention_model import HSA_model, HSA_VAE

import tensorflow as tf
import sys
import yaml
import warnings

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('INFO')
# dataset = str(sys.argv[1])

hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)


def train_model(dataset, train_data, train_vae=False):
    X_train, y_train = train_data

    n_window, n_timesteps, n_features, n_outputs = X_train.shape[
        1], X_train.shape[2], X_train.shape[3], y_train.shape[1]

    print('\nHIERARCHICAL SELF-ATTENTION MODEL:')

    hsa_model = HSA_model(hyperparameters['HSA_model']['modality_indices'][dataset], n_window, n_timesteps, n_features, n_outputs, d_model=hyperparameters['HSA_model']['d_model'],
                          num_heads=hyperparameters['HSA_model']['num_heads'], dff=hyperparameters['HSA_model']['dff'], dropout_rate=hyperparameters['HSA_model']['dropout']).get_model()

    hsa_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(
        lr=hyperparameters['train']['learning_rate']), metrics='accuracy')

    hsa_model.fit(X_train, y_train, epochs=hyperparameters['train']['epochs'], batch_size=hyperparameters['train']
                  ['batch_size'], verbose=2, validation_split=hyperparameters['train']['val_split'])

    if not train_vae:
        return hsa_model

    print('\nVARIATIONAL AUTOENCODER ON TOP OF HIERARCHICAL SELF-ATTENTION MODEL:')

    hsa_vae = HSA_VAE(base_model=hsa_model,
                      feature_dim=hyperparameters['HSA_model']['d_model']).get_model()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hyperparameters['train']['learning_rate'])
    hsa_vae.compile(optimizer)

    hsa_vae.fit(X_train, epochs=hyperparameters['train']['epochs'],
                batch_size=hyperparameters['train']['batch_size'])

    print('---TRAINING COMPLETE---')

    return hsa_model, hsa_vae
