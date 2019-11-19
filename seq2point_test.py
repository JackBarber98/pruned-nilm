import os 
import numpy as np 
import keras
import pandas as pd
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider
import matplotlib.pyplot as plt

def load_dataset(file_name, crop):
    data_frame = pd.read_csv(file_name, nrows=crop, header=0)
    test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
    test_target = np.round(np.array(data_frame.iloc[:, 1], float), 6)
    del data_frame
    return test_input, test_target

for file_name in os.listdir("./kettle/"):
    if ("TEST" in file_name):
        test_file_name = file_name
    else:
        print("No testing file was found.")
        break

TEST_FILE_NAME = "./kettle/kettle_test_.csv"

offset = int(0.5 * 599 - 1)

crop = 10000
test_input, test_target = load_dataset(TEST_FILE_NAME, crop)

model = create_model()

test_generator = TestingChunkSlider(number_of_windows=100, offset=offset)

testing_history = model.predict_generator(test_generator.load_data(test_input), steps=1)

test_target[test_target < 0] = 0
testing_history[testing_history < 0] = 0

plt.plot(testing_history, label="Testing")
plt.plot(test_target[0 : testing_history.size], label="Truth")
plt.title('Testing Results')
plt.ylabel('Prediction')
plt.xlabel('Testing Iteration')
plt.ylim(-100, 100)
plt.legend()
plt.savefig(fname="testing_results.png")
plt.show()