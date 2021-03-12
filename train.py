import sys
import warnings

import tensorflow as tf
import yaml

from model.hierarchical_self_attention_model import HSA_VAE, HSA_model_session_guided_window

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('INFO')
# dataset = str(sys.argv[1])

hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)


def train_model(dataset, train_data, val_data =None, train_vae=False, train_hsa=True):
    X_train, y_train, y_train_mid = train_data
    if val_data != None:
        X_val, y_val, y_val_mid = val_data

    print('\n[HIERARCHICAL SELF-ATTENTION MODEL]')

    hparams = hyperparameters['HSA_model'][dataset]

    hparams['n_window'], hparams['n_timesteps'], hparams['n_features'], hparams['n_outputs'] = X_train.shape[1], X_train.shape[2], X_train.shape[3], y_train.shape[1]
    hparams['n_outputs_window']=  y_train.shape[1]
    epochs = hparams.pop('epochs', None)
    batch_size = hparams.pop('batch_size', None)
    val_split = hparams.pop('val_split', None)

    hsa_model = HSA_model_session_guided_window(**hparams).get_compiled_model()
    # hsa_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(
    #     lr=hyperparameters['train']['learning_rate']), metrics='accuracy')
    
    if not train_hsa:
        return hsa_model

    # hsa_model.fit(X_train, y_train, epochs=hyperparameters['train']['epochs'], batch_size=hyperparameters['train']
    #               ['batch_size'], verbose=2, validation_split=hyperparameters['train']['val_split'])
    
    if val_data != None:
        hsa_model.fit(X_train, [y_train_mid, y_train], batch_size= batch_size, epochs=epochs, validation_data=(X_val, [y_val_mid, y_val]), use_multiprocessing=True)
    else:
        hsa_model.fit(X_train, [y_train_mid, y_train], batch_size= batch_size, epochs=epochs, validation_split=val_split, use_multiprocessing=True)

    if not train_vae:
        return hsa_model

    print('\n[VARIATIONAL AUTOENCODER ON TOP OF HIERARCHICAL SELF-ATTENTION MODEL]')

    hsa_vae = HSA_VAE(base_model=hsa_model,
                      feature_dim=hyperparameters['HSA_model']['d_model']).get_model()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hyperparameters['train']['learning_rate'])
    hsa_vae.compile(optimizer)

    hsa_vae.fit(X_train, epochs=hyperparameters['train']['epochs'], verbose=2,batch_size=hyperparameters['train']['batch_size'])

    print('---TRAINING COMPLETE---')

    return hsa_model, hsa_vae
