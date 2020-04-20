import os
import time
import re
import pandas as pd
import argparse
from remove_space import remove_space
from appliance_data import appliance_data

class DatasetGenerator():
    def __init__(self):
        args = self.get_arguments()

        self.__appliance = args.appliance_name
        self.__directory = "./refit_dataset/"
        self.__agg_mean = 522
        self.__agg_std = 814

        self.__training_set_length = 0

    def get_arguments(self):
        """ Lets the user specify the target appliance from the terminal. """

        parser = argparse.ArgumentParser(description="Generate the train, test, and validation datasets requried for an appliance. ")
        parser.add_argument("--appliance_name", 
                            type=remove_space, 
                            default="kettle", 
                            help="The appliance to generate datasets for. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
        return parser.parse_args()

    def digits_in_file_name(self, file_name):

        """ Returns grouped digits in a file name. (e.g. if the file name is "CLEAN_HOUSE_12", the 
        value 12 will be returned). 
        
        Parameters:
        file_name (string): The name of the file being processed.

        Returns: 
        digits (int): The grouped digits found in the file name.
        
        """

        digits = int(re.search(r"\d+", file_name).group())
        return digits

    # Loads data about a specified appliance and returns the data as a DataFrame.
    def load_file(self, house, channel):

        """ Loads and returns the file required to generate a specific dataset file as a pandas DataFrame. 
        
        Parameters:
        house (int): The house number of the data to load.
        channel (int): The column from which data should be extracted.

        Returns: 
        file_contents (pandas.DataFrame): The data from the portion of the file required.
        
        """

        file_name = self.__directory + "CLEAN_House" + str(house) + ".csv"
        file_contents = pd.read_csv(file_name,
                                    names=["aggregate", self.__appliance],
                                    usecols=[2, channel + 2],
                                    header=0,
                                    infer_datetime_format=True,
                                    parse_dates=True,
                                    memory_map=True)
        return file_contents

    def generate_test_house(self):

        """ Normalises the aggregate and appliance data for a specified house to be part of the testing set. 
        Writes this data to the testing file. """

        print("Formatting " + self.__appliance + " test data...")
        
        # Load the test data.
        test_data = self.load_file(appliance_data[self.__appliance]["test_house"], 
                                    appliance_data[self.__appliance]['channels'][appliance_data[self.__appliance]['houses']
                                    .index(appliance_data[self.__appliance]['test_house'])])
    
        # Normalise the appliance's test data.
        test_data["aggregate"] = (test_data["aggregate"] - self.__agg_mean) / self.__agg_std
        test_data[self.__appliance] = (test_data[self.__appliance] - appliance_data[self.__appliance]["mean"]) / appliance_data[self.__appliance]["std"]

        # Save the test data.
        test_data.to_csv("./" + self.__appliance + "/" + self.__appliance + "_test_.csv", index=False)

        # Delete test data from memory.
        del test_data

    def generate_validation_house(self):

        """ Normalises the aggregate and appliance data for a specified house to be part of the validation set. Writes 
        this data to the validation file. """

        print("Formatting " + self.__appliance + " validation data...")
        
        # Load the validation data.
        validation_data = self.load_file(appliance_data[self.__appliance]["validation_house"],
                                    appliance_data[self.__appliance]['channels'][appliance_data[self.__appliance]['houses']
                                    .index(appliance_data[self.__appliance]['validation_house'])])
        
        # Normalise the validation data.
        validation_data["aggregate"] = (validation_data["aggregate"] - self.__agg_mean) / self.__agg_std
        validation_data[self.__appliance] = (validation_data[self.__appliance] - appliance_data[self.__appliance]["mean"]) / appliance_data[self.__appliance]["std"]

        # Save validation data.
        validation_data.to_csv("./" + self.__appliance + "/" + self.__appliance + "_validation_.csv", index=False)

        # Delete validation data from memory.
        del validation_data

    def generate_train_house(self, file_name):

        """ Normalises the aggregate and appliance data for a specified house to be part of the training set. Writes 
        this data to the training file. 
        
        Parameters:
        file_name (string): The name of the file to be processed.
        
        """

        try:
            training_data = self.load_file(self.digits_in_file_name(file_name),
                                            appliance_data[self.__appliance]["channels"][appliance_data[self.__appliance]
                                            ["houses"].index(self.digits_in_file_name(file_name))])

            # Normalise the training data.
            training_data["aggregate"] = (training_data["aggregate"] - self.__agg_mean) / self.__agg_std
            training_data[self.__appliance] = (training_data[self.__appliance] - appliance_data[self.__appliance]["mean"]) / appliance_data[self.__appliance]["std"]
            rows, _ = training_data.shape

            self.__training_set_length += rows

            training_data.to_csv("./" + self.__appliance + "/" + self.__appliance + "_training_.csv", 
                                mode="a", 
                                index=False, 
                                header=False)
            
            # Delete training data from memory.
            del training_data
        except:
            print("House", self.digits_in_file_name(file_name), " not found. ")
            pass

    def generate(self):

        """ Generates normalised training, validation, and testing datasets from the cleaned REFIT dataset. """

        initial_time = time.time()

        print("Selected Appliance: ", self.__appliance)
        print("Directory of Dataset: ", self.__directory)


        if not os.path.exists(self.__appliance):
            os.makedirs(self.__appliance)
        
        # Loops through files and folders found in the directory
        for _, file_name in enumerate(os.listdir(self.__directory)):

            # Format the appliance's test data.
            if file_name == "CLEAN_House" + str(appliance_data[self.__appliance]["test_house"]) + ".csv":
                self.generate_test_house()

            # Format the appliance's validation data.
            elif file_name == "CLEAN_House" + str(appliance_data[self.__appliance]["validation_house"]) + ".csv":
                self.generate_validation_house()

            # Format training data.
            elif self.digits_in_file_name(file_name) in appliance_data[self.__appliance]["houses"]:
                self.generate_train_house(file_name)

        print("The training dataset contains " + str(self.__training_set_length) + " rows of data.")
        print("Datasets took " + str(time.time() - initial_time) + "s to generate")

dsg = DatasetGenerator()
dsg.generate()