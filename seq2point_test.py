import os 
import numpy as np 
import keras
import pandas as pd
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider
import matplotlib.pyplot as plt

def test_model():

    # Split the test dataset into features and targets.
    def load_dataset(file_name, crop):
        data_frame = pd.read_csv(file_name, nrows=crop, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_target = np.round(np.array(data_frame.iloc[:, 1], float), 6)
        del data_frame
        return test_input, test_target

    # Identify the test file.
    # for file_name in os.listdir("./kettle/"):
    #     print("file name: ", file_name)
    #     if ("test" in file_name):
    #         test_file_name = file_name
    #     else:
    #         print("No testing file was found.")
    #         break

    test_file_name = "./kettle/kettle_test_.csv"

    offset = int(0.5 * 599 - 1)

    # Get the (cropped) testing dataset.
    CROP = None
    DEFAULT_BATCH_SIZE = 1000
    test_input, test_target = load_dataset(test_file_name, CROP)

    # Initialise the model and testing generator.
    model = create_model()
    model = load_model(model, "./kettle/saved_model/kettle_model")
    test_generator = TestingChunkSlider(number_of_windows=100, inputs=test_input, offset=offset)

    # Calculate the optimum steps per epoch.
    steps_per_test_epoch = np.round(int(test_generator.total_size / DEFAULT_BATCH_SIZE), decimals=0)

    # Test the model.
    testing_history = model.predict_generator(test_generator.load_data(test_input), steps=steps_per_test_epoch)

    # Can't have negative energy readings - set any results below 0 to 0.
    test_target[test_target < 0] = 0
    testing_history[testing_history < 0] = 0

    # Plot testing outcomes against ground truth.
    plt.plot(testing_history, label="Testing")
    plt.plot(test_target[0 : testing_history.size], label="Ground Truth")
    plt.title('Testing Results')
    plt.ylabel('Prediction')
    plt.xlabel('Testing Iteration')
    plt.legend()
    plt.savefig(fname="testing_results.png")
    plt.show()