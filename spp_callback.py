import numpy as np
from numpy.linalg import norm
import tensorflow as tf 

class SPP(tf.keras.callbacks.Callback):

    """ Performs Wang et al.'s structured probabilistic pruning algorithm. Uses 
    probability theory to determine whether a weight is useful or not.

    Parameters:
    PRUNING_FREQUENCY (int): The intervals between pruning is performed (in epochs).
    R (float): The ratio of weights to prune from the convolutional layers.
    A (float): A constant to control the magnitude of delta r (see Wang et al.'s paper).
    u (float): A constant defining the flatness of the probability function.

    l1_norms (list): A list of the Manhattan distances of each filter in each convolutional layer.
    ranks (list): A list of ranks of each filter in each convolutional layer.
    layer_delta_ranks (list): A list of the outputs of delta(r) = f(r) as per Wang et al.'s research.
    layer_probabilities (list): A list of the probability of the weights in each filter being pruned.

    batch_iteration (int): A count of the number of batches that have been processed during training. 

    """

    def __init__(self):
        super(SPP, self).__init__()

        self.PRUNING_FREQUENCY = 1

        self.R = 0.5
        self.A = 0.05
        self.u = 0.25

        self.l1_norms = []
        self.ranks = []
        self.layer_delta_ranks = []
        self.layer_probabilities = []

        self.batch_iteration = 0

    def on_batch_end(self, epoch, logs={}):

        """ Checks whether the pruning conditions have been met and performs pruning only if they have been.

        Parameters:
        epoch (int): The current training epoch. 

        """

        if self.batch_iteration == 0:
            self.spp_pruning()
        try:
            if not self.ratio_is_greater_than_r() and not self.all_probabilities_integers():
                self.spp_pruning()
            else:
                return
        except:
            return

    def on_epoch_end(self, epoch, logs={}):
        print()
        print("Proportion of Resolved Filters: ", (np.count_nonzero(np.hstack(self.layer_probabilities) == 1) + np.count_nonzero(np.hstack(self.layer_probabilities) == 0)) / np.size(np.hstack(self.layer_probabilities)))
        print()

    def ratio_is_greater_than_r(self):

        """ Determines whether R filters have been pruned from the network. """

        flat_probs = np.hstack(self.layer_probabilities)
        if np.count_nonzero(flat_probs == 1) / np.size(flat_probs) >= self.R:
            return True

    def all_probabilities_integers(self):

        """ Checks whether the algorithm has determined exactly which filters to keep and which to prune. """

        flat_probs = np.hstack(self.layer_probabilities)
        ones = np.count_nonzero(flat_probs == 1)
        zeros = np.count_nonzero(flat_probs == 0)

        if ones + zeros == np.size(flat_probs):
            return True

    def spp_pruning(self):
        
        """ Performs all the steps required to update the pruning probabilities of each convolutional layer. """

        model = self.model

        layer_l1_norms = []
        layer_ranks = []
        layer_delta_ranks = []
        for layer in model.layers:
            if "conv" in layer.name:
                filterwise_weights = np.transpose(layer.get_weights()[0])

                filter_count = 0
                layer_norms = []
                for conv_filter in filterwise_weights:
                    norm = self.calc_layer_l1_norm(conv_filter)
                    layer_norms.append(norm)

                    filter_count += 1
                layer_l1_norms.append(layer_norms)
                layer_ranks = self.rank_filters(layer_norms)

                delta_ranks = self.calculate_delta_ranks(layer_ranks)

                layer_delta_ranks.append(delta_ranks)

        self.l1_norms = layer_l1_norms
        self.layer_delta_ranks = layer_delta_ranks

        self.update_probabilities(delta_ranks)

        if self.batch_iteration > 0:
            self.prune_weights()

        self.batch_iteration += 1

    def calc_layer_l1_norm(self, conv_filter):

        """ Calculates and returns the level one norms of each filter in a layer.

        Parameters:
        conv_filter (numpy.ndarray): An array of all the weights in a convolutional layer, 
        organised by filter.

        """

        return np.linalg.norm(np.array(conv_filter).flatten(), ord=1)

    def rank_filters(self, layer_norms):

        """ Returns the ranks of filters in a layer.

        Parameters:
        layer_norms (numpy.ndarray): The level one norms of the filters in a convolutional layer.
        """

        return np.array(layer_norms).argsort().argsort()

    def alpha(self, num_of_filters):

        """ Calculates and returns the value of alpha according to alpha = (log(2) - log(u)) / (R * N).

        Parameters:
        num_of_filters (int): The number of filters in the convolutional layer.

        """

        return (np.log10(2) - np.log10(self.u)) / (self.R * num_of_filters)

    def N(self, alpha):

        """ Calculates and returns the value of N such that N = -log(u) / alpha.

        Parameters: 
        alpha (float): The value calculated by the function self.alpha
        """

        return - np.log10(self.u) / alpha

    def calculate_delta_ranks(self, ranks):

        """ Calculates delta(r) such that delta(r) is a function of r.

        Parameters:
        ranks (numpy.ndarray): The ranks of the level one norms of the filters in each convolutional layer.

        """

        alpha = self.alpha(len(ranks))
        N = self.N(alpha)

        delta_ranks = []
        for rank in ranks:
            if rank <= N:
                delta = self.A * np.power(np.exp(1), - alpha * rank)
            else:
                delta = (2 * self.u * self.A) - (self.A * np.power(np.exp(1), - alpha * ((2 * N) - rank)))
            delta_ranks.append(delta)
        return delta_ranks

    def update_probabilities(self, delta_ranks):

        """ Updates the probability of each filter being pruned.

        Parameters:
        delta_ranks (numpy.ndarray): The outputs of the function self.delta_ranks.
        
        """

        if self.batch_iteration == 0:
            self.layer_probabilities = self.layer_delta_ranks

        layer_index = 0
        for layer in self.layer_probabilities:
            updated_layer_probs = np.maximum(np.minimum(np.array(layer) + self.layer_delta_ranks[layer_index], 1), 0)
            self.layer_probabilities[layer_index] = np.array(updated_layer_probs)
            layer_index += 1

    def calculate_zero_indicies(self, num_desired_zeros, mask):

        """ Sets weights that should not be pruned this pruning iteration to zero.

        Parameters:
        num_desired_zeros (int): The number of weights to remove from the layer.
        mask (numpy.ndarray): A weight mask. Where mask[x, y] = 0, the weight should be removed from the layer.

        """

        flattened_mask = mask.flatten()

        indices = [num_desired_zeros]
        mask_length = np.size(flattened_mask) - 1
        arange_array = np.arange(mask_length)
        for _ in range(0, num_desired_zeros):
            index = np.random.choice(arange_array)
            indices.append(index)

        flattened_mask.put(indices, 0)

        return np.reshape(flattened_mask, np.shape(mask))

    def prune_weights(self):

        """ Multiplies the mask by the weights in each filter of a convolutional layer to remove unrequired weights. """

        layer_index = 0
        conv_layer_index = 0
        for layer in self.model.layers:
            if "conv" in layer.name:
                weights = layer.get_weights()
                filterwise_weights = np.transpose(weights[0])

                filter_index = 0
                for conv_filter in filterwise_weights:
                    probability = self.layer_probabilities[conv_layer_index][filter_index]
                    num_desired_zeros = int(round(probability * np.size(conv_filter)))

                    pruned_weights = self.calculate_zero_indicies(num_desired_zeros, conv_filter)
                    filterwise_weights[filter_index] = pruned_weights

                    filter_index += 1

                weights[0] = np.reshape(filterwise_weights, np.shape(weights[0]))

                self.model.layers[layer_index].set_weights(weights)
            
                conv_layer_index += 1
            layer_index += 1