import numpy as np
from keras.callbacks import Callback
from model_structure import create_model

# RELATIVE THRESHOLD PRUNING

class Threshold(Callback):
    def __init__(self):
        super(Threshold, self).__init__()

        self.__lower_delta_percentiles = []
        self.__upper_delta_percentiles = []

    def on_train_end(self, logs={}):
        index = 0 
        for layer in self.model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                weights = layer.get_weights()
                self.get_delta_percentiles(layer.name, weights)
                self.prune_weights(index, weights)
            else:
                self.__lower_delta_percentiles.append(0)
                self.__upper_delta_percentiles.append(0)
            index += 1

    def get_delta_percentiles(self, layer_type, weights):
        if "conv" in layer_type:
            lower_delta = 45
            upper_delta = 55
        else:
            lower_delta = 15
            upper_delta = 85

        lower_percentile_value = np.percentile(weights[0], lower_delta)
        upper_percentile_value = np.percentile(weights[0], upper_delta)
        self.__lower_delta_percentiles.append(lower_percentile_value)
        self.__upper_delta_percentiles.append(upper_percentile_value)

    def prune_weights(self, index, weights):
        original_shape = np.shape(weights[0])
        flat_weights = np.array(weights[0]).flatten()

        weight_index = 0 
        for weight in flat_weights:
            if weight > self.__lower_delta_percentiles[index] and weight < self.__upper_delta_percentiles[index]:
                    flat_weights[weight_index] = 0
            weight_index += 1
        weights[0] = np.reshape(flat_weights, original_shape)

        self.model.layers[index].set_weights(weights)
        