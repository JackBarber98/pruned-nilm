import os 
import numpy as np 
import keras
import pandas as pd
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider

def load_dataset(file_name, crop):
    data_frame = pd.read_csv(file_name, nrows = crop, header=0, na_filter = False)
    test_input = np.round(np.array(data_frame.iloc[:, 0], float), 5)
    test_target = np.round(np.array(data_frame.iloc[:, 1], float), 5)
    del data_frame
    return test_input, test_target

for file_name in os.listdir("./kettle/"):
    if ("TEST" in file_name):
        test_file_name = file_name

offset = int(0.5 * 599 - 1)

crop = 1000
test_input, test_target = load_dataset(test_file_name, crop)

model = create_model(batch_size=1000)

test_generator = TestingChunkSlider(number_of_windows=100, offset=offset)

model.predict_generator(test_generator.load_data(test_input))
