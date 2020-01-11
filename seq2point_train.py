import os 
import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization import sparsity

from data_feeder import InputChunkSlider
from model_structure import create_model, save_model, load_model
from spp_callback import SPP
from entropic_callback import Entropic

def train_model(APPLIANCE, PRUNING_ALGORITHM, BATCH_SIZE, CROP):

    # Static maximum chunk.
    SIZE_OF_CHUNK = 5 * 10 ** 2

    TRAINING_DIRECTORY = "./" + APPLIANCE + "/" + APPLIANCE + "_test_.csv"
    VALIDATION_DIRECTORY = "./" + APPLIANCE + "/" + APPLIANCE + "_test_.csv"

    WINDOW_OFFSET = int((0.5 * 601 ) - 1)

    # Generator function to produce batches.
    TRAINING_CHUNKER = InputChunkSlider(file_name=TRAINING_DIRECTORY, 
                                        chunk_size=SIZE_OF_CHUNK, 
                                        batch_size=BATCH_SIZE, 
                                        crop=CROP, shuffle=True, 
                                        offset=WINDOW_OFFSET, 
                                        header=0, 
                                        ram_threshold=5*10**5)

    VALIDATION_CHUNKER = InputChunkSlider(file_name=VALIDATION_DIRECTORY, 
                                            chunk_size=SIZE_OF_CHUNK, 
                                            batch_size=BATCH_SIZE, 
                                            crop=CROP, shuffle=True, 
                                            offset=WINDOW_OFFSET, 
                                            header=0, 
                                            ram_threshold=5*10**5)

    def default_train(model, early_stopping, steps_per_training_epoch):
        entropic = Entropic()

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=10,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=10,
            validation_freq=2,
            callbacks=[early_stopping, entropic])
        return training_history

    def tfmot_pruning(model, early_stopping, steps_per_training_epoch):
        pruning_params = {
            'pruning_schedule': sparsity.keras.ConstantSparsity(0.8, 0),
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }

        model = sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "mae", "accuracy"])

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=10,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=10,
            validation_freq=2,
            callbacks=[early_stopping, sparsity.keras.UpdatePruningStep()])

        model = sparsity.keras.strip_pruning(model)
        return training_history

    def spp_pruning(model, early_stopping, steps_per_training_epoch):
        spp = SPP(model=model)

        training_history = model.fit_generator(TRAINING_CHUNKER.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=10,
            verbose=1,
            validation_data = VALIDATION_CHUNKER.load_dataset(),
            validation_steps=100,
            validation_freq=2,
            callbacks=[early_stopping, spp])
        return training_history

    # Calculate the optimum steps per epoch.
    TRAINING_CHUNKER.check_if_chunking()
    steps_per_training_epoch = np.round(int(TRAINING_CHUNKER.total_size / BATCH_SIZE), decimals=0)

    # Compile the model with an Adam optimiser. Initialise early stopping callback that stops only 
    # if there's no improvement 2 epochs later.
    model = create_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "msle", "mae", "accuracy"]) 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=0, verbose=1, mode="auto")

    if PRUNING_ALGORITHM == "default":
        training_history = default_train(model, early_stopping, steps_per_training_epoch)
    if PRUNING_ALGORITHM == "tfmot_pruning":
        training_history = tfmot_pruning(model, early_stopping, steps_per_training_epoch)
    if PRUNING_ALGORITHM == "spp_pruning":
        training_history = spp_pruning(model, early_stopping, steps_per_training_epoch)

    model.summary()

    # Plot the monitored metrics and save.
    plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
    plt.plot(training_history.history["val_loss"], label="Val Loss")
    plt.title('Training Metrics')
    plt.ylabel('y')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(fname="training_results.png")
    plt.show()

    save_model(model, "./kettle/saved_model/kettle_model")