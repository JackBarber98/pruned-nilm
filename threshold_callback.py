import numpy as np
from keras.callbacks import Callback
from model_structure import create_model

# STRUCTURED PROBABILISTIC PRUNING PROTOTYPE

# Currently prunes weights based on the probability that they are 
# useless until a sparsity ratio is achieved.

# Things to improve:
#  - Compute probability for each column, not 
#    just each filter (for Conv layers) and layers (for Dense layers). 
#    The current implementation heavily prunes the lighter layers first.
#
#  - Implement Monte Carlo sampling for masking. It's currently just 
#    uniform random distribution.

class Threshold(Callback):
    def __init__(self):
        super(Threshold, self).__init__()

        self.PRUNING_FREQUENCY = 5

        self.delta_percentiles = []

    def on_train_end(self, logs={}):
        index = 0 
        for layer in self.model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                weights = layer.get_weights()
                self.get_delta_percentiles(weights)
                self.prune_weights(index, weights)
            else:
                self.delta_percentiles.append(0)
            index += 1

    def get_delta_percentiles(self, weights):
        delta = 0.5

        percentile_value = np.percentile(weights[0], delta)
        self.delta_percentiles.append(percentile_value)

    def set_weights_to_prune(self, index, weights):
        original_shape = np.shape(weights[0])
        flat_weights = np.array(weights[0]).flatten()

        weight_index = 0 
        for weight in flat_weights:
            if weight < self.delta_percentiles[index]:
                flat_weights[weight_index] = 0
            weight_index += 1

        weights[0] = np.reshape(flat_weights, original_shape)

        self.model.layers[index].set_weights(weights)
        