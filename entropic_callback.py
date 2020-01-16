import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ROUGH VALUES (probably will be deleted).
# prob_values = []
# info_values = []
# for prob_dens in freq:
#     prob = np.trapz([prob_dens, prob_dens + 0.1], dx=0.01)
#     info = - np.log(prob)
#     prob_values.append(prob)
#     info_values.append(info)

class Entropic(tf.keras.callbacks.Callback):

    """Performs Hur et al.'s entropy-based pruning algorithm. Uses weight information and entropy to 
    determine which weights to prune.

    Parameters:
    PRUNING_FREQUENCY (int): The intervals between pruning is performed (in epochs).
    means (list): The mean of the weights of each layer.
    stds (list): The standards deviation of the weights of each layer.
    layer_probabilities (list): A 2D list of the probability of a weight existing.
    layer_entropies (list): A 1D list of the entropy of each layer of the network.
    layer_information (list): A 2D list of the information held by each weight. 

    """

    def __init__(self):
        super(Entropic, self).__init__()

        self.PRUNING_FREQUENCY = 5

        self.means = []
        self.stds = []

        self.layer_probabilities = []
        self.layer_entropies = []
        self.layer_information = []

    def on_epoch_end(self, epoch, logs={}):

        self.means = []
        self.stds = []
        self.layer_probabilities = []
        self.layer_information = []
        self.layer_entropies = []

        if epoch % self.PRUNING_FREQUENCY == 0:
            index = 0 
            for layer in self.model.layers:
                weights = layer.get_weights()
                self.generate_pruning_stats(weights)
                self.calculate_probability_and_information_values(weights, index)
                self.calculate_weight_entropy(weights, index)
                self.prune_weights(layer, weights, index)

                index += 1

    def get_probability_distribution(self, weights, index):

        """Generates a probability density for each weight in a layer.

        Parameters:
        weights (numpy.array): The layer's weights.
        index (int): The index of the layer in the model's layers array.

        Returns:
        probability_densities (numpy.array): A probability densitity for each weight.

        """

        distribution_values = np.random.normal(self.means[index], self.stds[index], np.size(weights))
        _, bins, _ = plt.hist(distribution_values, int(np.size(weights) - 1), density=True)

        probability_densities = 1 / (self.stds[index] * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.means[index]) ** 2 / (2 * self.stds[index] ** 2))
        return probability_densities

    def generate_pruning_stats(self, weights):
        if np.shape(weights)[0] != 0:

            self.means.append(np.mean(weights[0]))
            self.stds.append(np.std(weights[0]))

        else:
            self.means.append(0)
            self.stds.append(0)

    def calculate_probability_and_information_values(self, weights, index):

        """Calculates the probability of a weight existing over a short interval and the 
        information that weight holds.

        Parameters:
        weights (numpy.array): The layer's weights.
        index (int): The index of the layer in the model's layers array.

        """

        if np.shape(weights)[0] != 0:
            probability_distribution = self.get_probability_distribution(weights[0], index)
            
            probability_values = []
            information_values = []
            for density in probability_distribution:
                probability = np.trapz([density, density + 0.1], dx=0.01)
                probability_values.append(probability)
                information_value = - np.log10(probability)
                information_values.append(information_value)

            self.layer_probabilities.append(probability_values)
            self.layer_information.append(information_values)
        else:
            self.layer_probabilities.append(0)
            self.layer_information.append(0)

    def calculate_weight_entropy(self, weights, index):

        """Calculates the entropy of each layer of the network.

        Parameters:
        weights (numpy.array): The layer's weights.
        index (int): The index of the layer in the model's layers array.

        """

        if np.shape(weights)[0] != 0:
            entropies = np.log(self.stds[index] * np.sqrt(2 * np.pi * np.exp(1)))
        else:
            entropies = 0
        self.layer_entropies.append(entropies)

    def prune_weights(self, layer, weights, index):

        """Sets the weights deemed to be redundant to zero.

        Parameters:
        weights (numpy.array): The layer's weights.
        index (int): The index of the layer in the model's layers array.

        """

        if np.shape(weights)[0] != 0:
            original_shape = np.shape(weights[0])
            flattened_weights = weights[0].flatten()

            weight_index = 0
            for _ in flattened_weights:
                if self.layer_information[index][weight_index] < - self.layer_entropies[index] * 1.1:
                    flattened_weights[weight_index] = 0
                weight_index += 1

            weights[0] = np.reshape(flattened_weights, original_shape)
            layer.set_weights(weights) 

        index += 1        