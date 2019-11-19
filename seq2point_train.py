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

SIZE_OF_CHUNK = 5 * 10 ** 2

training_directory = "./kettle/kettle_training_.csv"
validation_directory = "./kettle/kettle_validation_.csv"
window_offset = int((0.5 * 599 ) - 1)

training_chunker = InputChunkSlider(file_name=training_directory, chunk_size=SIZE_OF_CHUNK, batch_size=1000, crop=10000,shuffle=True, offset=window_offset, header=0, ram_threshold=5*10**5)
model = create_model()
model.compile(Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss="mean_squared_error", metrics=["mae"])
training_history = model.fit_generator(training_chunker.load_dataset(), steps_per_epoch=1, epochs=4, verbose=2, callbacks=None)



plt.plot(training_history.history['loss'], label="Training")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(fname="training_results.png")
plt.show()

save_model(model, "./kettle/kettle_model")