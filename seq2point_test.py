import os 
import numpy as np 
import keras
import pandas as pd
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider
import matplotlib.pyplot as plt

# Split the test dataset into features and targets.
def load_dataset(file_name, crop):
    data_frame = pd.read_csv(file_name, nrows=crop, header=0)
    test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
    test_target = np.round(np.array(data_frame.iloc[:, 1], float), 6)
    del data_frame
    return test_input, test_target

# Identify the test file.
for file_name in os.listdir("./kettle/"):
    if ("TEST" in file_name):
        test_file_name = file_name
    else:
        print("No testing file was found.")
        break

offset = int(0.5 * 599 - 1)

# Get the (cropped) testing dataset.
CROP = None
test_input, test_target = load_dataset(test_file_name, CROP)

# Initialise the model and testing generator.
model = create_model()
<<<<<<< HEAD
test_generator = TestingChunkSlider(number_of_windows=100, offset=offset)
=======

test_generator = TestingChunkSlider(number_of_windows=1000, offset=offset)
>>>>>>> 7e15f265340991a6697e4964b7415c99bde496a3

# Test the model.
testing_history = model.predict_generator(test_generator.load_data(test_input), steps=1)

# Can't have negative energy readings - set any results below 0 to 0.
test_target[test_target < 0] = 0
testing_history[testing_history < 0] = 0

# Plot testing outcomes against ground truth.
plt.plot(testing_history, label="Testing")
plt.plot(test_target[0 : testing_history.size], label="Ground Truth")
plt.title('Testing Results')
plt.ylabel('Prediction')
plt.xlabel('Testing Iteration')
<<<<<<< HEAD
=======
# plt.ylim(-100, 100)
>>>>>>> 7e15f265340991a6697e4964b7415c99bde496a3
plt.legend()
plt.savefig(fname="testing_results.png")
plt.show()