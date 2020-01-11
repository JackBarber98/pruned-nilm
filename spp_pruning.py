import math
import random
import numpy as np
from numpy.linalg import norm
from keras.models import Model
from model_structure import create_model, load_model, save_model

# A Structured Probability Pruning implementation based on https://arxiv.org/pdf/1709.06994.pdf.
# Swaps out Monte Carlo sampling for uniform random distribution.

def calculate_kernel_distances(weights):
    # The distance of each filter.
    kernel_distances = []
    for kernel in weights[0]:
        distance = norm(kernel[0], ord=1)
        kernel_distances.append(distance)

    kernel_distances = np.asarray(kernel_distances)    
    return kernel_distances

def calculate_dense_distance(weights):
    return norm(weights[0], ord=1)

def calculate_distances(layers):
    distances = []

    for layer in layers:
        weights = layer.get_weights()

        # Ignores flatten, input, etc.
        if np.shape(weights)[0] != 0:

            # If it's a conv layer look at each filter.
            if len(np.shape(weights[0])) == 4:

                kernel_distances = calculate_kernel_distances(weights)
                distances.append(kernel_distances)
            else:

                # The distance of each non-conv layer.
                distance = calculate_dense_distance(weights)
                distances.append(distance)

    distances = np.hstack(np.asarray(distances))
    return distances

def calculate_pruning_probabilities(pruning_probabilities, rankings):
    alpha = (math.log(2) - math.log(u)) / (R * len(rankings))

    index = 0
    for rank in rankings:
        N = - math.log(u) / alpha
        if rank <= N:
            delta = A * math.exp(-alpha * rank)
        else:
            delta = (2 * u * A) - (A * math.exp(-alpha * ((2 * N) - rank)))

        pruning_probabilities[index] = np.maximum(np.minimum(pruning_probabilities[index] + delta, 1), 0)

        index += 1
    return pruning_probabilities

def calculate_zero_indicies(num_desired_zeros, mask):
    flattened_mask = mask.flatten()

    indices = [num_desired_zeros]
    mask_length = np.size(flattened_mask) - 1
    arange_array = np.arange(mask_length)
    for _ in range(0, num_desired_zeros):
        index = np.random.choice(arange_array)
        indices.append(index)

    flattened_mask.put(indices, 0)
    return flattened_mask

def prune_weights(pruning_probabilities):
    probability_count = 0
    layer_count = 0
    for layer in layers:
        if np.shape(layer.get_weights())[0] != 0:
            mask = []
            weights = layer.get_weights()

            # Conv layers
            if len(np.shape(weights[0])) == 4:
                for kernel in weights[0]:
                    pruning_probability = pruning_probabilities[probability_count]

                    num_weights = np.size(kernel) - 1
                    num_desired_zeros = int(round(pruning_probability * num_weights))

                    mask = np.ones(np.shape(kernel), dtype="float32")
                    flattened_mask = calculate_zero_indicies(num_desired_zeros, mask)
                    mask = flattened_mask.reshape(np.shape(kernel))

                    new_kernel = np.multiply(kernel, mask)
                    np.putmask(weights[0], weights[0] == kernel, new_kernel)

                    probability_count += 1

            # Dense layers
            else:
                pruning_probability = pruning_probabilities[probability_count]

                num_weights = np.size(kernel) - 1
                num_desired_zeros = int(round(pruning_probability * num_weights))

                mask = np.ones(np.shape(weights[0]), dtype="float32")
                flattened_mask = calculate_zero_indicies(num_desired_zeros, mask)
                mask = flattened_mask.reshape(np.shape(weights[0]))

                new_weights = np.multiply(weights[0], mask)
                weights[0] = new_weights

                probability_count += 1
            model.layers[layer_count].set_weights(weights)
        layer_count += 1

def get_num_weight_groups(model):
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

R = 1
A = 0.05
u = 0.25
t = 180

model = create_model()
model = load_model(model, "asdf")

rankings = np.array(get_num_weight_groups(model))
pruning_probabilities = np.ones(get_num_weight_groups(model))
for i in range(0, 14):

    layers = model.layers
    distances = calculate_distances(model.layers)
    rankings = distances.argsort().argsort()
    pruning_probabilities = calculate_pruning_probabilities(pruning_probabilities, rankings)
    prune_weights(pruning_probabilities)

save_model(model, "./kettle/saved_model/kettle_model")