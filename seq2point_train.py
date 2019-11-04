import os
from data_feeder import InputChunkSlider
from model_structure import create_model, save_model
import keras.backend as K 
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np 
import pandas as pd 

SIZE_OF_CHUNK = 5 * 10 ** 6

training_directory = "./kettle/kettle_training_.csv"
validation_directory = "./kettle/kettle_validation_.csv"
window_offset = int((0.5 * 599 ) - 1)

training_chunker = InputChunkSlider(file_name=training_directory, batch_size=1000, chunk_size=SIZE_OF_CHUNK, crop=None, shuffle=True, offset=window_offset, header=0, ram_threshold=5*10**5)
model = create_model(100)
model.compile(Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mean_squared_error", metrics=["mae"])
model.fit_generator(training_chunker.load_dataset(), steps_per_epoch=500, epochs=10, verbose=2, callbacks=None, initial_epoch=1)

save_model(model, "./kettle/kettle_model")