import os
import logging
import numpy as np 
import keras
import pandas as pd
import tensorflow as tf 
import time
from model_structure import create_model, load_model
from data_feeder import TestingChunkSlider
from appliance_data import appliance_data
import matplotlib.pyplot as plt

class Tester():
    def __init__(self, appliance, pruning_algorithm, transfer_domain, crop, batch_size, network_type):
        self.__appliance = appliance
        self.__pruning_algorithm = pruning_algorithm
        self.__transfer_domain = transfer_domain
        self.__network_type = network_type

        self.__crop = crop
        self.__batch_size = batch_size
        self.__window_size = 601
        self.__window_offset = int(0.5 * self.__window_size - 1)

        self.plotting = True

        self.__test_directory = "./" + self.__appliance + "/" + self.__appliance + "_test_.csv"

        log_file = "./" + self.__appliance + "/saved_model/" + self.__appliance + "_" + self.__pruning_algorithm + "_test_log.log"
        logging.basicConfig(filename=log_file,level=logging.INFO)

    def test_model(self):

        """ Tests a fully-trained model using a sliding window generator as an input. Measures inference time, gathers, and 
        plots evaluationg metrics. """

        test_input, test_target = self.load_dataset(self.__test_directory)
        model = create_model()
        model = load_model(model, self.__network_type, self.__pruning_algorithm, self.__appliance)

        test_generator = TestingChunkSlider(number_of_windows=100, inputs=test_input, targets=test_target, offset=self.__window_offset)

        # Calculate the optimum steps per epoch.
        steps_per_test_epoch = np.round(int(test_generator.total_size / self.__batch_size), decimals=0)

        # Test the model.
        start_time = time.time()
        testing_history = model.predict(x=test_generator.load_data(), steps=steps_per_test_epoch, verbose=2)

        print("MAX: ", np.max(testing_history))
        print("MIN: ", np.min(testing_history))
        end_time = time.time()
        test_time = end_time - start_time

        evaluation_metrics = model.evaluate(x=test_generator.load_data(), steps=steps_per_test_epoch)

        self.log_results(model, test_time, evaluation_metrics)
        self.plot_results(testing_history, test_input, test_target)


    def load_dataset(self, directory):
        """Loads the testing dataset from the location specified by file_name.

        Parameters:
        directory (string): The location at which the dataset is stored, concatenated with the file name.

        Returns:
        test_input (numpy.array): The first n (crop) features of the test dataset.
        test_target (numpy.array): The first n (crop) targets of the test dataset.

        """

        data_frame = pd.read_csv(directory, nrows=self.__crop, skiprows=0, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_target = np.round(np.array(data_frame.iloc[self.__window_offset: -self.__window_offset, 1], float), 6)
        
        del data_frame
        return test_input, test_target

    def log_results(self, model, test_time, evaluation_metrics):

        """Logs the inference time, MAE and MSE of an evaluated model.

        Parameters:
        model (tf.keras.Model): The evaluated model.
        test_time (float): The time taken by the model to infer all required values.
        evaluation metrics (list): The MSE, MAE, and various compression ratios of the model.

        """

        inference_log = "Inference Time: " + str(test_time)
        logging.info(inference_log)

        metric_string = "MSE: ", str(evaluation_metrics[0]), " MAE: ", str(evaluation_metrics[3])
        logging.info(metric_string)

        self.count_pruned_weights(model)  

    def count_pruned_weights(self, model):

        """ Counts the total number of weights, pruned weights, and weights in convolutional 
        layers. Calculates the sparsity ratio of different layer types and logs these values.

        Parameters:
        model (tf.keras.Model): The evaluated model.

        """
        num_total_zeros = 0
        num_total_weights = 0
        num_dense_zeros = 0
        num_dense_weights = 0
        num_conv_zeros = 0
        num_conv_weights = 0
        for layer in model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                layer_weights = layer.get_weights()[0].flatten()

                if "conv" in layer.name:
                    num_conv_weights += np.size(layer_weights)
                    num_conv_zeros += np.count_nonzero(layer_weights==0)

                    num_total_weights += num_conv_weights
                    num_total_zeros += num_conv_zeros
                else:
                    num_dense_weights += np.size(layer_weights)
                    num_dense_zeros += np.count_nonzero(layer_weights==0)

                    num_total_weights += num_dense_weights
                    num_total_zeros += num_dense_zeros

        conv_zeros_string = "CONV. ZEROS: " + str(num_conv_zeros)
        conv_weights_string = "CONV. WEIGHTS: " + str(num_conv_weights)
        conv_sparsity_ratio = "CONV. RATIO: " + str(num_conv_zeros / num_conv_weights)

        dense_weights_string = "DENSE WEIGHTS: " + str(num_dense_weights)
        dense_zeros_string = "DENSE ZEROS: " + str(num_dense_zeros)
        dense_sparsity_ratio = "DENSE RATIO: " + str(num_dense_zeros / num_dense_weights)

        total_zeros_string = "TOTAL ZEROS: " + str(num_total_zeros)
        total_weights_string = "TOTAL WEIGHTS: " + str(num_total_weights)
        total_sparsity_ratio = "TOTAL RATIO: " + str(num_total_zeros / num_total_weights)

        print("LOGGING PATH: ", "./" + self.__appliance + "/saved_model/" + self.__appliance + "_" + self.__pruning_algorithm + "_test_log.log")

        logging.info(conv_zeros_string)
        logging.info(conv_weights_string)
        logging.info(conv_sparsity_ratio)
        logging.info("")
        logging.info(dense_zeros_string)
        logging.info(dense_weights_string)
        logging.info(dense_sparsity_ratio)
        logging.info("")
        logging.info(total_zeros_string)
        logging.info(total_weights_string)
        logging.info(total_sparsity_ratio)

    def plot_results(self, testing_history, test_input, test_target):

        """ Generates and saves a plot of the testing history of the model against the (actual) 
        aggregate energy values and the true appliance values.

        Parameters:
        testing_history (numpy.ndarray): The series of values inferred by the model.
        test_input (numpy.ndarray): The aggregate energy data.
        test_target (numpy.ndarray): The true energy values of the appliance.

        """

        testing_history = ((testing_history * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance]["mean"])
        test_target = ((test_target * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance]["mean"])
        test_agg = (test_input.flatten() * 814) + 522
        test_agg = test_agg[:testing_history.size]

        # Can't have negative energy readings - set any results below 0 to 0.
        test_target[test_target < 0] = 0
        testing_history[testing_history < 0] = 0
        test_input[test_input < 0] = 0

        # Plot testing outcomes against ground truth.
        plt.figure(1)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Testing")
        plt.title('Kettle Preliminary Test Results')
        plt.ylabel('Normalised Prediction')
        plt.xlabel('Testing Window')
        plt.legend()

        file_path = "./" + self.__appliance + "/saved_model/" + self.__appliance + "_" + self.__pruning_algorithm + "_test_figure.png"
        plt.savefig(fname=file_path)

        plt.show()