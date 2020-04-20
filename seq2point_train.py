import os 
import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization import sparsity

from data_feeder import TrainSlidingWindowGenerator
from model_structure import create_model, create_reduced_model, create_dropout_model, create_reduced_dropout_model, save_model
from pruning_algorithms.spp_callback import SPP
from pruning_algorithms.entropic_callback import Entropic
from pruning_algorithms.threshold_callback import Threshold

import smtplib

class Trainer():
    def __init__(self, appliance, pruning_algorithm, batch_size, crop, network_type):
        self.__appliance = appliance
        self.__pruning_algorithm = pruning_algorithm
        self.__batch_size = batch_size
        self.__crop = crop
        self.__network_type = network_type

        self.__sequence_length = 601
        self.__window_offset = int((0.5 * self.__sequence_length) - 1)
        self.__max_chunk_size = 5 * 10 ** 2

        # Directories of the training and validation files. Always has the structure 
        # ./{appliance_name}/{appliance_name}_train_.csv for training or 
        # ./{appliance_name}/{appliance_name}_validation_.csv
        self.__training_directory = "./" + self.__appliance + "/" + self.__appliance + "_training_.csv"
        self.__validation_directory = "./" + self.__appliance + "/" + self.__appliance + "_validation_.csv"

        self.__training_chunker = TrainSlidingWindowGenerator(file_name=self.__training_directory, 
                                        chunk_size=self.__max_chunk_size, 
                                        batch_size=self.__batch_size, 
                                        crop=self.__crop, shuffle=True,
                                        skip_rows=10000000, 
                                        offset=self.__window_offset, 
                                        ram_threshold=5*10**5)
        self.__validation_chunker = TrainSlidingWindowGenerator(file_name=self.__validation_directory, 
                                            chunk_size=self.__max_chunk_size, 
                                            batch_size=self.__batch_size, 
                                            crop=self.__crop, 
                                            shuffle=True,
                                            skip_rows=0, 
                                            offset=self.__window_offset, 
                                            ram_threshold=5*10**5)

        self.__validation_frequency = 1

    def train_model(self):

        """ Trains an energy disaggregation model using a user-selected pruning algorithm (default is no pruning). 
        Plots and saves the resulting model. """

        # Calculate the optimum steps per epoch.
        self.__training_chunker.check_if_chunking()
        steps_per_training_epoch = np.round(int(self.__training_chunker.total_size / self.__batch_size), decimals=0)

        if self.__network_type == "reduced":
            model = create_reduced_model()
        elif self.__network_type == "dropout":
            model = create_dropout_model()
        elif self.__network_type == "reduced_dropout":
            model = create_reduced_dropout_model()
        else:
            model = create_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "msle", "mae"]) 
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")

        if self.__pruning_algorithm == "tfmot":
            training_history = self.tfmot_pruning(model, early_stopping, steps_per_training_epoch)
        if self.__pruning_algorithm == "spp":
            training_history = self.spp_pruning(model, early_stopping, steps_per_training_epoch)
        if self.__pruning_algorithm == "entropic":
            training_history = self.entropic_pruning(model, early_stopping, steps_per_training_epoch)
        if self.__pruning_algorithm == "threshold":
            training_history = self.threshold_pruning(model, early_stopping, steps_per_training_epoch)
        if self.__pruning_algorithm == "default":
            training_history = self.default_train(model, early_stopping, steps_per_training_epoch)

        training_history.history["val_loss"] = np.repeat(training_history.history["val_loss"], self.__validation_frequency)

        model.summary()
        save_model(model, self.__network_type, self.__pruning_algorithm, self.__appliance)

        self.plot_training_results(training_history)

    def default_train(self, model, early_stopping, steps_per_training_epoch):

        """The default training method the neural network will use. No pruning occurs.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        training_history = model.fit_generator(self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=2,
            verbose=1,
            validation_data = self.__validation_chunker.load_dataset(),
            validation_steps=1,
            validation_freq=1,
            callbacks=[early_stopping])

        return training_history

    def tfmot_pruning(self, model, early_stopping, steps_per_training_epoch):

        """Trains the model with TensorFlow Optimisation Tookit's prune_low_magnitude method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        pruning_params = {
            'pruning_schedule': sparsity.keras.ConstantSparsity(0.70, 0),
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }

        model = sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "mae"])

        training_history = model.fit_generator(self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=50,
            verbose=1,
            validation_data = self.__validation_chunker.load_dataset(),
            validation_steps=10,
            validation_freq=self.__validation_frequency,
            callbacks=[early_stopping, sparsity.keras.UpdatePruningStep()])

        model = sparsity.keras.strip_pruning(model)

        return training_history

    def spp_pruning(self, model, early_stopping, steps_per_training_epoch):


        """Trains the model with Wang et al.'s structured probabilistic pruning method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        spp = SPP()

        training_history = model.fit_generator(self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=50,
            verbose=1,
            validation_data = self.__validation_chunker.load_dataset(),
            validation_steps=10,
            validation_freq=self.__validation_frequency,
            callbacks=[early_stopping, spp])
        return training_history

    def entropic_pruning(self, model, early_stopping, steps_per_training_epoch):


        """Trains the model with the entropic pruning method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        entropic = Entropic()

        training_history = model.fit_generator(self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=50,
            verbose=1,
            validation_data = self.__validation_chunker.load_dataset(),
            validation_steps=10,
            validation_freq=self.__validation_frequency,
            callbacks=[early_stopping, entropic])
        return training_history

    def threshold_pruning(self, model, early_stopping, steps_per_training_epoch):


        """Trains the model with Ashouri et al.'s threshold-based pruning method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        threshold = Threshold()

        training_history = model.fit_generator(self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=50,
            verbose=1,
            validation_data = self.__validation_chunker.load_dataset(),
            validation_steps=10,
            validation_freq=self.__validation_frequency,
            callbacks=[early_stopping, threshold])
        return training_history

    def plot_training_results(self, training_history):

        """ Plots and saves a graph of training loss against epoch.

        Parameters:
        training_history (numpy.ndarray): A timeseries of loss against epoch count.

        """

        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title('Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        file_name = "./" + self.__appliance + "/saved_models/" + self.__appliance + "_" + self.__pruning_algorithm + "_training_results.png"
        plt.savefig(fname=file_name)