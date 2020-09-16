from preprocessing.utils import get_train_test_data
from model.hierarchical_self_attention_model import HSA_model
import tensorflow as tf
import sys
import yaml 
import warnings

warnings.filterwarnings("ignore")                                                                                        

dataset = str(sys.argv[1])

hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)

def train_model(dataset):
    (X_train, y_train), (X_test, y_test) = get_train_test_data(dataset=dataset)

    n_window, n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], X_train.shape[3], y_train.shape[1]

    hsa_model = HSA_model(hyperparameters['HSA_model']['modality_indices'][dataset], n_window, n_timesteps, n_features, n_outputs, d_model=hyperparameters['HSA_model']['d_model'], num_heads = hyperparameters['HSA_model']['num_heads'], dff=hyperparameters['HSA_model']['dff'], dropout_rate=hyperparameters['HSA_model']['dropout'])

    model = hsa_model.get_model()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=hyperparameters['train']['learning_rate']), metrics='accuracy')
    print(model.summary())

    model.fit(X_train, y_train, epochs=hyperparameters['train']['epochs'], batch_size=hyperparameters['train']['batch_size'], verbose=1, validation_split=hyperparameters['train']['val_split'])

if __name__ == "__main__":
    train_model(dataset)
