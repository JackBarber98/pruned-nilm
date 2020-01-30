import numpy as np
from numpy.linalg import norm
import tensorflow as tf 

class SPP2(tf.keras.callbacks.Callback):
    def __init__(self):
        super(SPP2, self).__init__()

        self.PRUNING_FREQUENCY = 5

        self.R = 0.5
        self.A = 0.05
        self.u = 0.25

        self.layer_distances = []
        self.network_probabilities = []
        self.network_deltas = []

        self.pruning_iteration = 0

    def on_epoch_end(self, epoch, logs={}):
        self.print_layers()
        self.pruning_iteration += 1

    def manhattan_distance(self, weight_group):
        return(norm(weight_group, ord=1))

    def rank_weight_groups(self, distances):
        return np.array(distances).argsort().argsort()

    def calculate_delta_ranks(self, filter_ranks):
        alpha = self.alpha(len(filter_ranks))
        N = self.N(alpha)

        delta_ranks = []
        for rank in filter_ranks:
            if rank <= N:
                delta = self.A * np.power(np.exp(1), - alpha * rank)
            else:
                delta = (2 * self.u * self.A) - (self.A * np.power(np.exp(1), - alpha * ((2 * N) - rank)))
            delta_ranks.append(delta)
        return delta_ranks


    def alpha(self, num_filters):
        return (np.log10(2) - np.log10(self.u)) / (self.R * num_filters)

    def N(self, alpha):
        return - np.log10(self.u) / alpha

    # [ [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4] ]

    def update_probabilities(self, layer_index, delta_ranks):
        filter_probabilities = []

        old = self.network_probabilities

        filter_index = 0
        for delta_rank in delta_ranks:
            if self.pruning_iteration == 0:
                filter_probabilities.append(np.maximum(np.minimum(delta_rank, 1), 0))
                self.network_probabilities.append(filter_probabilities)
            else:
                print("OLD: ", self.network_probabilities[layer_index])
                filter_probabilities.append(np.maximum(np.minimum(self.network_probabilities[layer_index][filter_index] + delta_rank, 1), 0))
                print("NEW: ", filter_probabilities)
                self.network_probabilities[layer_index] = filter_probabilities

        if old == self.network_probabilities:
            print("THEY ARE THE SAME ://")
            print()


    def print_layers(self):
        layer_index = 0
        for layer in self.model.layers:

            weights = layer.get_weights()
            if np.shape(weights)[0] != 0:
                if np.ndim(weights[0]) == 4:

                    filter_distances = []
                    filter_ranks = []
                    for kernel in weights[0]:
                        filter_distances.append(self.manhattan_distance(np.array(kernel).flatten()))
                    
                    filter_ranks = self.rank_weight_groups(filter_distances)
                    delta_ranks = self.calculate_delta_ranks(filter_ranks)

                    self.update_probabilities(layer_index, delta_ranks)

                    layer_index += 1