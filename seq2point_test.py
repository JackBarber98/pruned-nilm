import os 
import numpy as np 
import keras
import pandas as pd
import time
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider
from appliance_data import appliance_data
import matplotlib.pyplot as plt

def test_model(appliance, test_domain, crop, batch_size):

    # Split the test dataset into features and targets.
    def load_dataset(file_name, crop):
        data_frame = pd.read_csv(file_name, nrows=crop, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_target = np.round(np.array(data_frame.iloc[offset:-offset, 1], float), 6)
        del data_frame
        return test_input, test_target

    model_directory = "./" + appliance + "/saved_model/"
    test_file = "./" + test_domain + "/" + test_domain + "_test_.csv"

    offset = int(0.5 * 601 - 1)

    test_input, test_target = load_dataset(test_file, crop)

    # Initialise the model and testing generator.
    model = create_model()

    model = load_model(model, model_directory)

    test_generator = TestingChunkSlider(number_of_windows=1000, inputs=test_input, offset=offset)

    # Calculate the optimum steps per epoch.
    steps_per_test_epoch = np.round(int(test_generator.total_size / batch_size), decimals=0)

    # Test the model.
    start_time = time.time()
    testing_history = model.predict_generator(test_generator.load_data(), steps=steps_per_test_epoch)
    end_time = time.time()
    test_time = end_time - start_time
    print("Test Time: ", test_time)
    print("Num of Zeros: ", np.count_nonzero(model.layers[2].get_weights()[0]==0))

    testing_history = ((testing_history * appliance_data[test_domain]["std"]) + appliance_data[test_domain]["mean"])
    test_target = ((test_target * appliance_data[test_domain]["std"]) + appliance_data[test_domain]["mean"])
    test_agg = (test_input.flatten() * 814) + 522
    test_agg = test_agg[:testing_history.size]

    # Can't have negative energy readings - set any results below 0 to 0.
    # test_target[test_target < 0] = 0
    # testing_history[testing_history < 0] = 0
    # test_input[test_input < 0] = 0

    # Plot testing outcomes against ground truth.
    plt.figure(1)
    plt.plot(test_agg[offset: -offset], label="Aggregate")
    plt.plot(test_target[0 : testing_history.size], label="Ground Truth")
    plt.plot(testing_history, label="Testing")
    plt.title('Kettle Preliminary Test Results')
    plt.ylabel('Normalised Prediction')
    plt.xlabel('Testing Window')
    plt.legend()
    plt.savefig(fname="testing_results.png")
    
    plt.show()