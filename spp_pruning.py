import numpy as np
from numpy.linalg import norm
import tensorflow as tf 

class SPP(tf.keras.callbacks.Callback):
    def __init__(self):
        super(SPP, self).__init__()

        self.PRUNING_FREQUENCY = 1

        self.R = 0.5
        self.A = 0.05
        self.u = 0.25

        self.layer_distances = []
        self.network_probabilities = []
        self.network_deltas = []

        self.pruning_iteration = 0

    def on_epoch_end(self, epoch, logs={}):
        model = self.model

        for layer in model.layers:
            if "conv" in layer.type:
                print("DOING IT")