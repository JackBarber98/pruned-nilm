import numpy as np
from numpy.linalg import norm
import math
from keras.callbacks import Callback

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

class SPP(Callback):
    def __init__(self, model):
        super(SPP, self).__init__()

        self.PRUNING_FREQUENCY = 5

        # The ratio of weights to prune.
        self.R = 1

        # The greater the value, the more aggressive the pruning.
        self.A = 0.2

        # Changes the flatness of the function (higher value results in a more flat function).
        self.u = 0.25

        self.num_of_groups = self.get_num_weight_groups(model)

        self.layers = []
        self.distances = np.array(self.num_of_groups)
        self.rankings = np.array(self.num_of_groups)
        self.pruning_probabilities = np.zeros(self.num_of_groups)
        self.zero_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.PRUNING_FREQUENCY == 0:
            if np.sum(self.pruning_probabilities == 1) / np.size(self.pruning_probabilities) < self.R:
                distances = self.calculate_distances(self.model.layers)
                rankings = distances.argsort().argsort()
                self.pruning_probabilities = self.calculate_pruning_probabilities(rankings)

                self.prune_weights(self.pruning_probabilities)

                print(self.zero_count, " weights have been pruned.")
            else:
                print("Pruning complete.")
                return

        return

    def get_num_weight_groups(self, model):
        group_count = 0

        for layer in model.layers:
            weights = layer.get_weights()
            if np.shape(weights)[0] != 0:
                if len(np.shape(weights[0])) == 4:
                    for _ in weights[0]:
                        group_count += 1
                else:
                    group_count += 1

        return group_count

    def calculate_kernel_distances(self, weights):
        kernel_distances = []
        for kernel in weights[0]:
            distance = norm(kernel[0], ord=1)
            kernel_distances.append(distance)

        kernel_distances = np.asarray(kernel_distances)    
        return kernel_distances

    def calculate_dense_distance(self, weights):
        return norm(weights[0], ord=1)

    def calculate_distances(self, layers):
        distances = []

        for layer in layers:
            weights = layer.get_weights()

            if np.shape(weights)[0] != 0:

                if len(np.shape(weights[0])) == 4:

                    kernel_distances = self.calculate_kernel_distances(weights)
                    distances.append(kernel_distances)
                else:

                    distance = self.calculate_dense_distance(weights)
                    distances.append(distance)

        distances = np.hstack(np.asarray(distances))
        return distances

    def calculate_pruning_probabilities(self, rankings):
        alpha = (math.log(2) - math.log(self.u)) / (self.R * np.size(rankings))

        index = 0
        for rank in rankings:
            N = - math.log(self.u) / alpha
            if rank <= N:
                delta = self.A * math.exp(-alpha * rank)
            else:
                delta = (2 * self.u * self.A) - (self.A * math.exp(-alpha * ((2 * N) - rank)))

            self.pruning_probabilities[index] = np.maximum(np.minimum(self.pruning_probabilities[index] + delta, 1), 0)

            index += 1
        return self.pruning_probabilities

    def calculate_zero_indicies(self, num_desired_zeros, mask):
        flattened_mask = mask.flatten()

        indices = [num_desired_zeros]
        mask_length = np.size(flattened_mask) - 1
        arange_array = np.arange(mask_length)
        for _ in range(0, num_desired_zeros):
            index = np.random.choice(arange_array)
            indices.append(index)

        flattened_mask.put(indices, 0)
        return flattened_mask

    def prune_weights(self, pruning_probabilities):
        self.zero_count = 0

        probability_count = 0
        layer_count = 0
        for layer in self.model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                mask = []
                weights = layer.get_weights()

                if len(np.shape(weights[0])) == 4:
                    for kernel in weights[0]:
                        pruning_probability = self.pruning_probabilities[probability_count]

                        num_weights = np.size(kernel) - 1
                        num_desired_zeros = int(round(pruning_probability * num_weights))

                        mask = np.ones(np.shape(kernel), dtype="float32")
                        flattened_mask = self.calculate_zero_indicies(num_desired_zeros, mask)
                        mask = flattened_mask.reshape(np.shape(kernel))

                        new_kernel = np.multiply(kernel, mask)
                        np.putmask(weights[0], weights[0] == kernel, new_kernel)

                        probability_count += 1

                else:
                    pruning_probability = self.pruning_probabilities[probability_count]

                    num_weights = np.size(weights[0]) - 1
                    num_desired_zeros = int(round(pruning_probability * num_weights))

                    mask = np.ones(np.shape(weights[0]), dtype="float32")
                    flattened_mask = self.calculate_zero_indicies(num_desired_zeros, mask)
                    mask = flattened_mask.reshape(np.shape(mask))

                    new_weights = np.multiply(weights[0], mask)
                    weights[0] = new_weights

                    probability_count += 1
                self.zero_count += np.count_nonzero(weights[0]==0)
                self.model.layers[layer_count].set_weights(weights)
            layer_count += 1    