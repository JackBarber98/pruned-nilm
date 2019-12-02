import os
from data_feeder import InputChunkSlider
from model_structure import create_model, save_model
import keras.backend as K 
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def train_model():
    # Static maximum chunk size and batch size.
    SIZE_OF_CHUNK = 5 * 10 ** 2
    DEFAULT_BATCH_SIZE = 1000

    training_directory = "./kettle/kettle_validation_.csv"
    validation_directory = "./kettle/kettle_test_.csv"
    window_offset = int((0.5 * 601 ) - 1)

    # Generator function to produce batches.
    training_chunker = InputChunkSlider(file_name=training_directory, chunk_size=SIZE_OF_CHUNK, batch_size=DEFAULT_BATCH_SIZE, crop=100000, shuffle=True, offset=window_offset, header=0, ram_threshold=5*10**5)

    validation_chunker = InputChunkSlider(file_name=validation_directory, chunk_size=SIZE_OF_CHUNK, batch_size=DEFAULT_BATCH_SIZE, crop=100000, shuffle=False, offset=window_offset, header=0, ram_threshold=5*10**5)

    # Calculate the optimum steps per epoch.
    training_chunker.check_if_chunking()
    steps_per_training_epoch = np.round(int(training_chunker.total_size / DEFAULT_BATCH_SIZE), decimals=0)

    # Compile the model with an Adam optimiser. Initialise early stopping callback that stops only 
    # if there's no improvement 2 epochs later.
    model = create_model()
    model.compile(Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mse", metrics=["mse", "msle", "mae"]) 
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=1, verbose=2, mode="auto")

    # Train the model with the generator function and early stopping callback.
    training_history = model.fit_generator(training_chunker.load_dataset(),
        steps_per_epoch=steps_per_training_epoch,
        epochs=50,
        verbose=2,
        validation_data = validation_chunker.load_dataset(),
        validation_steps=1000,
        validation_freq=2,
        callbacks=[early_stopping])

    # Plot the monitored metrics and save.
    plt.plot(training_history.history['mse'], label="MAE")
    plt.plot(training_history.history['msle'], label="MSE (Loss)")
    plt.title('Training Metrics')
    plt.ylabel('y')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(fname="training_results.png")
    plt.show()

    save_model(model, "./kettle/saved_model/kettle_model")

train_model()