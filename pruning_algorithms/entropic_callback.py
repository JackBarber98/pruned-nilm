import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Entropic(tf.keras.callbacks.Callback):

    """ Performs Hur et al.'s entropy-based pruning algorithm. Uses weight information and entropy to 
    determine which weights to prune.

    Parameters:
    pruning_frequency (int): The intervals between pruning is performed (in epochs).
    previous_loss (float): The loss experienced during the last training epoch.
    means (list): The mean of the weights of each layer.
    stds (list): The standards deviation of the weights of each layer.
    layer_probabilities (list): A 2D list of the probability of a weight existing.
    layer_entropies (list): A 1D list of the entropy of each layer of the network.
    layer_information (list): A 2D list of the information held by each weight. 

    """

    def __init__(self):
        super(Entropic, self).__init__()

        self.__pruning_frequency = 2001
        self.__previous_loss = 0

        self.__means = []
        self.__stds = []

        self.__layer_probabilities = []
        self.__layer_entropies = []
        self.__layer_information = []

        self.__batch_count = 0

    def on_batch_end(self, epoch, logs={}):

        """ Determines when pruning should occur and triggers each step of the pruning process.

        Parameters:
        epoch (int): The current training epoch.

        """

        self.__means = []
        self.__stds = []
        self.__layer_probabilities = []
        self.__layer_information = []
        self.__layer_entropies = []

        if self.model_is_stable(logs["loss"], epoch) and self.__batch_count % self.__pruning_frequency == 0:
            index = 0

            for layer in self.model.layers:
                weights = layer.get_weights()
                self.generate_pruning_stats(weights)
                self.calculate_probability_and_information_values(weights, index)
                self.calculate_weight_entropy(weights, index)
                self.prune_weights(layer, weights, index)

                index += 1
                self.__batch_count += 1

        self.__previous_loss = logs["loss"]

    def model_is_stable(self, current_loss, epoch):

        """ Determines whether the change in loss is greater than or less than 5 * 10^-3. This has been found to 
        be the point at which a seq2point network becomes stable.

        Parameters:
        current_loss (float): The loss resulting from the current training epoch.
        epoch (int): The current training epoch count.

        """

        if epoch == 0:
            delta = current_loss
        else:
            delta = current_loss - self.__previous_loss

        if np.absolute(delta) <= 0.07:
            return True

    def get_probability_distribution(self, weights, index):

        """ Generates a probability density for each weight in a layer.

        Parameters:
        weights (numpy.array): The layer's weights.
        index (int): The index of the layer in the model's layers array.

        Returns:
        probability_densities (numpy.array): A probability densitity for each weight.

        """

        distribution_values = np.random.normal(self.__means[index], self.__stds[index], np.size(weights))
        _, bins = np.histogram(distribution_values, bins=int(np.size(weights) - 1), density=True)

        probability_densities = 1 / (self.__stds[index] * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.__means[index]) ** 2 / (2 * self.__stds[index] ** 2))
        return probability_densities

    def generate_pruning_stats(self, weights):
        """ Calculates the mean and standard deviation of weights in a layer

        Parameters:
        weights (numpy.ndarray): An array of all non-bias weights in a layer.

        """

        if np.shape(weights)[0] != 0:
            self.__means.append(np.mean(weights[0]))
            self.__stds.append(np.std(weights[0]))

        else:
            self.__means.append(0)
            self.__stds.append(0)

    def calculate_probability_and_information_values(self, weights, index):

        """ Calculates the probability of a weight existing over a short interval and the 
        information that weight holds.

        Parameters:
        weights (numpy.ndarray): An array of all non-bias weights in a layer.
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

            self.__layer_probabilities.append(probability_values)
            self.__layer_information.append(information_values)
        else:
            self.__layer_probabilities.append(0)
            self.__layer_information.append(0)

    def calculate_weight_entropy(self, weights, index):

        """ Calculates the entropy of each layer of the network.

        Parameters:
        weights (numpy.ndarray): An array of all non-bias weights in a layer.
        index (int): The index of the layer in the model's layers array.

        """

        if np.shape(weights)[0] != 0:
            entropies = np.log(self.__stds[index] * np.sqrt(2 * np.pi * np.exp(1)))
        else:
            entropies = 0
        self.__layer_entropies.append(entropies)

    def prune_weights(self, layer, weights, index):

        """ Sets the weights deemed to be redundant to zero.

        Parameters:
        weights (numpy.ndarray): An array of all non-bias weights in a layer.
        index (int): The index of the layer in the model's layers array.

        """

        if np.shape(weights)[0] != 0:
            original_shape = np.shape(weights[0])
            flattened_weights = weights[0].flatten()

            weight_index = 0
            for _ in flattened_weights:
                if self.__layer_information[index][weight_index] < - self.__layer_entropies[index] * 1.1:
                    flattened_weights[weight_index] = 0
                weight_index += 1
            weights[0] = np.reshape(flattened_weights, original_shape)
            layer.set_weights(weights) 

        index += 1        