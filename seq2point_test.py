import os
import logging
import numpy as np 
import keras
import pandas as pd
import time
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider
from appliance_data import appliance_data
import matplotlib.pyplot as plt

def count_pruned_weights(model):
    num_zeros = 0
    num_weights = 0
    for layer in model.layers:
        if np.shape(layer.get_weights())[0] != 0:
            layer_weights = layer.get_weights()[0].flatten()
            num_zeros += np.count_nonzero(layer_weights==0)
            num_weights += np.size(layer_weights)
    pruned_log = "Pruned Weights: " + str(num_zeros)
    logging.info(pruned_log)

    compression_log = "Compression Factor: " + str((num_weights - num_zeros) / num_weights)
    logging.info(compression_log)

def test_model(appliance, algorithm, test_domain, crop, batch_size):

    """Tests a pre-trained seq2point model in either the domain it was trained in or using transfer 
    learning Plots and saves the results.

    Parameters:
    appliance (string): The appliance that the neural network was trained to make inferences for.
    algorithm (string): The pruning algorithm of the model to test.
    test_domain (string): The appliance that the neural network will be tested with.
    crop (int): The number of rows of the testing dataset to train the network with.
    batch_size (int): The portion of the cropped dataset to be processed by the network at once.

    """

    # Split the test dataset into features and targets.
    def load_dataset(file_name, crop):

        """Loads the testing dataset from the location specified by file_name.

        Parameters:
        file_name (string): The path and file name of the test CSV file.
        crop (int): The number of rows of the test dataset to use.

        Returns:
        test_input (numpy.array): The first n (crop) features of the test dataset.
        test_target (numpy.array): The first n (crop) targets of the test dataset.

        """

        data_frame = pd.read_csv(file_name, nrows=crop, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_target = np.round(np.array(data_frame.iloc[offset: -offset, 1], float), 6)
        
        del data_frame
        return test_input, test_target

    model_directory = "./" + appliance + "/saved_model/" + appliance + "_model_"
    test_file = "./" + appliance + "/" + appliance + "_test_.csv"

    offset = int(0.5 * 601 - 1)

    test_input, test_target = load_dataset(test_file, crop)

    # Initialise the model and testing generator.
    model = create_model()

    model = load_model(model, algorithm, model_directory)

    test_generator = TestingChunkSlider(number_of_windows=100, inputs=test_input, targets=test_target, offset=offset)

    # Calculate the optimum steps per epoch.
    steps_per_test_epoch = np.round(int(test_generator.total_size / batch_size), decimals=0)

    # Test the model.
    start_time = time.time()
    testing_history = model.predict(x=test_generator.load_data() ,steps=steps_per_test_epoch)
    end_time = time.time()
    test_time = end_time - start_time

    log_file = "./" + appliance + "/saved_model/" + appliance + "_" + algorithm + "_test_log.log"
    logging.basicConfig(filename=log_file,level=logging.INFO)

    inference_log = "Inference Time: " + str(test_time)
    logging.info(inference_log)

    count_pruned_weights(model)

    evaluation_metrics = model.evaluate(x=test_generator.load_data(), steps=steps_per_test_epoch)

    metric_string = "MSE: ", str(evaluation_metrics[0]), " MAE: ", str(evaluation_metrics[3])
    logging.info(metric_string)

    testing_history = ((testing_history * appliance_data[appliance]["std"]) + appliance_data[appliance]["mean"])
    test_target = ((test_target * appliance_data[appliance]["std"]) + appliance_data[appliance]["mean"])
    test_agg = (test_input.flatten() * 814) + 522
    test_agg = test_agg[:testing_history.size]

    # Can't have negative energy readings - set any results below 0 to 0.
    test_target[test_target < 0] = 0
    testing_history[testing_history < 0] = 0
    test_input[test_input < 0] = 0

    # Plot testing outcomes against ground truth.
    plt.figure(1)
    plt.plot(test_agg[offset: -offset], label="Aggregate")
    plt.plot(test_target[:test_agg.size - (2 * offset)], label="Ground Truth")
    plt.plot(testing_history[:test_agg.size - (2 * offset)], label="Testing")
    plt.title('Kettle Preliminary Test Results')
    plt.ylabel('Normalised Prediction')
    plt.xlabel('Testing Window')
    plt.legend()

    file_path = "./" + appliance + "/saved_model/" + appliance + "_" + algorithm + "_test_figure.png"
    plt.savefig(fname=file_path)