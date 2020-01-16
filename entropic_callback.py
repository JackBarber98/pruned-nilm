import numpy as np
from numpy.linalg import norm
import scipy
import scipy.stats as stats
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

class Entropic(tf.keras.callbacks.Callback):
    def __init__(self):
        super(Entropic, self).__init__()

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

        self.generate_pruning_stats()

        self.calculate_probability_and_information_values()
        self.calculate_weight_entropy()
            
        # ROUGH VALUES.
        # prob_values = []
        # info_values = []
        # for prob_dens in freq:
        #     prob = np.trapz([prob_dens, prob_dens + 0.1], dx=0.01)
        #     info = - np.log(prob)
        #     prob_values.append(prob)
        #     info_values.append(info)

        self.prune_weights()

    def get_probability_distribution(self, weights, index):
        distribution_values = np.random.normal(self.means[index], self.stds[index], np.size(weights))
        _, bins, _ = plt.hist(distribution_values, int(np.size(weights) - 1), density=True)
        return 1 / (self.stds[index] * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.means[index]) ** 2 / (2 * self.stds[index] ** 2))

    def generate_pruning_stats(self):
        for layer in self.model.layers:
            weights = layer.get_weights()
            if np.shape(weights)[0] != 0:

                self.means.append(np.mean(weights[0]))
                self.stds.append(np.std(weights[0]))

            else:
                self.means.append(0)
                self.stds.append(0)

    def calculate_probability_and_information_values(self):
        index = 0
        for layer in self.model.layers[:6]:
            weights = layer.get_weights()
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
            index += 1


    def calculate_weight_entropy(self):
        index = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            if np.shape(weights)[0] != 0:
                entropies = np.log(self.stds[index] * np.sqrt(2 * math.pi * np.exp(1)))
            else:
                entropies = 0
            self.layer_entropies.append(entropies)

            index += 1

    def prune_weights(self):
        index = 0
        for layer in self.model.layers[:6]:
            weights = layer.get_weights()
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