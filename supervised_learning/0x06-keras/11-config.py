#!/usr/bin/env python3
'''Save and Load Weights'''


import tensorflow.keras as K


def save_config(network, filename):
    '''saves a model’s configuration in JSON format:
    Args:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None
    '''
    network = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(network)
    return None


def load_config(filename):
    '''loads a model with a specific configuration
    Args:
        filename is the path of the file containing the model’s configuration
        in JSON format
    Returns: the loaded model
    '''
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model
