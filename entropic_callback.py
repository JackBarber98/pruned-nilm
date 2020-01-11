import numpy as np
from numpy.linalg import norm
import scipy
import scipy.stats as stats
import math
import tensorflow as tf

class Entropic(tf.keras.callbacks.Callback):
    def __init__(self):
        super(Entropic, self).__init__()

        self.means = []
        self.stds = []

        self.layer_probabilities = []
        self.layer_entropies = []
        self.layer_information = []

    def on_epoch_end(self, epoch, logs={}):
        self.generate_pruning_stats()
        self.probability_of_weights()
        self.calculate_information()

    def generate_pruning_stats(self):
        for layer in self.model.layers:
            weights = layer.get_weights()
            if np.shape(weights)[0] != 0:

                self.means.append(np.nanmean(weights[0]))
                self.stds.append(np.nanstd(weights[0]))

            else:
                self.means.append(0)
                self.stds.append(0)

    def probability(self, input_value, index):
        return np.power((math.exp(1) / (np.sqrt(2 * math.pi * np.power(self.stds[index], 2)))), (-np.power(input_value - self.means[index], 2) / (2 * np.power(self.stds[index], 2))))

    def probability_of_weights(self):
        index = 0 
        for layer in self.model.layers:

            weights = layer.get_weights()
            if np.shape(weights)[0] != 0:
                probability = self.probability(weights[0], index)
                self.layer_probabilities.append(probability)
            else:
                self.layer_probabilities.append(-1)
            index += 1

            print(self.layer_probabilities)

    def calculate_weight_entropy(self):
        index = 0
        for layer in self.model.layers:
            weights = layer.get_weights()

            if np.shape(weights)[0] != 0:
                entropies = np.log(self.stds[index] * np.sqrt(2 * math.pi * math.exp(1)))
                self.layer_entropies.append(entropies)
            else:
                self.layer_entropies.append(0)
            index += 1

    def calculate_information(self):
        index = 0
        for layer in self.model.layers:
            weights = layer.get_weights()

            # if np.shape(weights)[0] != 0:
            #     # print(np.shape(weights[0]))
            #     # print(self.layer_probabilities[0])
            #         # for weight in weights[0][0]:
            #         #     # y_1 = self.probability(weight)
            #         #     # y_2 = self.probability(weight + 0.00001)

            # else:
            #     self.layer_entropies.append(0)
            # index += 1        