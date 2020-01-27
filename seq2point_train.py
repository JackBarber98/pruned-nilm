import os 
import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization import sparsity

from data_feeder import InputChunkSlider
from model_structure import create_dropout_model, create_model, save_model, load_model
from spp_callback_2 import SPP2
from entropic_callback import Entropic
from threshold_callback import Threshold

def train_model(APPLIANCE, PRUNING_ALGORITHM, BATCH_SIZE, CROP):

    """Trains an energy disaggregation model using a user-selected pruning algorithm (default is no pruning). 
    Plots and saves the resulting model.

    Parameters:
    appliance (string): The appliance that the neural network will be able to infer energy readings for.
    pruning_algorithm (string): The pruning algorithm that will be applied when training the network.
    crop (int): The number of rows of the training dataset to train the network with.
    batch_size (int): The portion of the cropped dataset to be processed by the network at once.

    """

    # Static maximum chunk.
    SIZE_OF_CHUNK = 5 * 10 ** 2

    # Directories of the training and validation files. Always has the structure 
    # ./{appliance_name}/{appliance_name}_train_.csv for training or 
    # ./{appliance_name}/{appliance_name}_validation_.csv
    TRAINING_DIRECTORY = "./" + APPLIANCE + "/" + APPLIANCE + "_test_.csv"
    VALIDATION_DIRECTORY = "./" + APPLIANCE + "/" + APPLIANCE + "_test_.csv"

    WINDOW_OFFSET = int((0.5 * 601 ) - 1)

    # Generator function to produce batches of training data.
    TRAINING_CHUNKER = InputChunkSlider(file_name=TRAINING_DIRECTORY, 
                                        chunk_size=SIZE_OF_CHUNK, 
                                        batch_size=BATCH_SIZE, 
                                        crop=CROP, shuffle=True, 
                                        offset=WINDOW_OFFSET, 
                                        ram_threshold=5*10**5)

    # Generator function to produce batches of validation data.
    VALIDATION_CHUNKER = InputChunkSlider(file_name=VALIDATION_DIRECTORY, 
                                            chunk_size=SIZE_OF_CHUNK, 
                                            batch_size=BATCH_SIZE, 
                                            crop=CROP, shuffle=True, 
                                            offset=WINDOW_OFFSET, 
                                            ram_threshold=5*10**5)

    def default_train(model, early_stopping, steps_per_training_epoch):

        """The default training method the neural network will use. No pruning occurs.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.array): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=1,
            verbose=1,
            # validation_data = VALIDATION_CHUNKER.load_dataset(),
            # validation_steps=0,
            # validation_freq=0,
            callbacks=[early_stopping])
        return training_history

    def tfmot_pruning(model, early_stopping, steps_per_training_epoch):

        """Trains the model with TensorFlow Optimisation Tookit's prune_low_magnitude method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.array): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        pruning_params = {
            'pruning_schedule': sparsity.keras.ConstantSparsity(0.5, 0),
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }

        model = sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "mae"])

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=15,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=10,
            validation_freq=2,
            callbacks=[early_stopping, sparsity.keras.UpdatePruningStep()])

        model = sparsity.keras.strip_pruning(model)
        return training_history

    def spp_pruning(model, early_stopping, steps_per_training_epoch):


        """Trains the model with Wang et al.'s structured probabilistic pruning method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.array): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        spp = SPP2()

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=15,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=100,
            validation_freq=2,
            callbacks=[early_stopping, spp])
        return training_history

    def entropic_pruning(model, early_stopping, steps_per_training_epoch):


        """Trains the model with the entropic pruning method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.array): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        entropic = Entropic()

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=15,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=100,
            validation_freq=2,
            callbacks=[early_stopping, entropic])
        return training_history

    def threshold_pruning(model, early_stopping, steps_per_training_epoch):


        """Trains the model with Ashouri et al.'s threshold-based pruning method.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.array): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """

        threshold = Threshold()

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=15,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=100,
            validation_freq=2,
            callbacks=[early_stopping, threshold])
        return training_history


    # Calculate the optimum steps per epoch.
    TRAINING_CHUNKER.check_if_chunking()
    steps_per_training_epoch = np.round(int(TRAINING_CHUNKER.total_size / BATCH_SIZE), decimals=0)

    # Compile the model with an Adam optimiser. Initialise early stopping callback that stops only 
    # if there's no improvement 2 epochs later.
    model = create_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "msle", "mae"]) 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=0, verbose=1, mode="auto")

    if PRUNING_ALGORITHM == "default":
        training_history = default_train(model, early_stopping, steps_per_training_epoch)
    if PRUNING_ALGORITHM == "tfmot":
        training_history = tfmot_pruning(model, early_stopping, steps_per_training_epoch)
    if PRUNING_ALGORITHM == "spp":
        training_history = spp_pruning(model, early_stopping, steps_per_training_epoch)
    if PRUNING_ALGORITHM == "entropic":
        training_history = entropic_pruning(model, early_stopping, steps_per_training_epoch)
    if PRUNING_ALGORITHM == "threshold":
        training_history = threshold_pruning(model, early_stopping, steps_per_training_epoch)

    model.summary()

    # Plot the monitored metrics and save.
    plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
    #plt.plot(training_history.history["val_loss"], label="Val Loss")
    plt.title('Training Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(fname="training_results.png")
    plt.show()

    save_model(model, PRUNING_ALGORITHM, "./" + APPLIANCE + "/saved_model/" + APPLIANCE + "_model_")