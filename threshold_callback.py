import numpy as np
from keras.callbacks import Callback
from model_structure import create_model

# RELATIVE THRESHOLD PRUNING

class Threshold(Callback):
    def __init__(self):
        super(Threshold, self).__init__()

        self.PRUNING_FREQUENCY = 1

        self.delta_percentiles = []

    def on_train_end(self, logs={}):
        index = 0 
        for layer in self.model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                weights = layer.get_weights()
                self.get_delta_percentiles(layer.name, weights)
                self.prune_weights(index, weights)
            else:
                self.delta_percentiles.append(0)
            index += 1

    def get_delta_percentiles(self, layer_type, weights):
        if "conv" in layer_type:
            delta = 50
        else:
            delta = 70

        percentile_value = np.percentile(weights[0], delta)
        self.delta_percentiles.append(percentile_value)

    def prune_weights(self, index, weights):
        original_shape = np.shape(weights[0])
        flat_weights = np.array(weights[0]).flatten()

        weight_index = 0 
        for weight in flat_weights:
            if weight < 0 and self.delta_percentiles[index] < 0:
                if weight > self.delta_percentiles[index]:
                    flat_weights[weight_index] = 0
            if weight > 0 and self.delta_percentiles[index] > 0:
                if weight < self.delta_percentiles[index]:
                    flat_weights[weight_index] = 0
            weight_index += 1
        weights[0] = np.reshape(flat_weights, original_shape)

        self.model.layers[index].set_weights(weights)
        