import os
import time
import re
import pandas as pd
from appliance_data import appliance_data

class DatasetGenerator():
    def __init__(self, appliance):
        self.__appliance = appliance
        self.__directory = "./refit_dataset"
        self.__agg_mean = 522
        self.__agg_std = 814

    def digits_in_file_name(self, file_name):
        return int(re.search(r"\d+", file_name).group())

    # Loads data about a specified appliance and returns the data as a DataFrame.
    def load_file(self, house, channel):
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
        print("Formatting " + self.__appliance + " validation data...")
        
        # Load the validation data.
        validation_data = self.load_file(appliance_data[self.__appliance]["validation_house"],
                                    appliance_data[self.__appliance]['channels'][appliance_data[self.__appliance]['houses']
                                    .index(appliance_data[self.__appliance]['validation_house'])])
        
        # Normalise the validation data.
        validation_data["aggregate"] = (validation_data["aggregate"] - self.__agg_mean) / self.__agg_std
        validation_data[self.__appliance] = (validation_data[self.__appliance] - appliance_data[self.__appliance]["mean"]) / appliance_data[self.__appliance]["std"]

        # Save validation data.
        validation_data.to_csv("./" + self.__appliance + "./" + self.__appliance + "_validation_.csv", index=False)

        # Delete validation data from memory.
        del validation_data

    def generate_train_house(self):
        try:
            training_data = self.load_file(self.digits_in_file_name(file_name),
                                            appliance_data[self.__appliance]["channels"][appliance_data[self.__appliance]["houses"]
                                            .index(self.digits_in_file_name(file_name))])

            # Normalise the training data.
            training_data["aggregate"] = (training_data["aggregate"] - self.__agg_mean) / self.__agg_std
            training_data[self.__appliance] = (training_data[self.__appliance] - appliance_data[self.__appliance]["mean"]) / appliance_data[self.__appliance]["std"]
            rows, _ = training_data.shape
            
            length += rows

            training_data.to_csv("./" + self.__appliance + "/" + self.__appliance + "_training_.csv", mode="a", index=False, header=False)
            
            # Delete training data from memory.
            del training_data
        except:
            print("PASS")
            pass

    def generate(self):
        initial_time = time.time()

        print("Selected Appliance: ", self.__appliance)
        print("Directory of Dataset: ", self.__directory)

        length = 0

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
                self.generate_train_house()

        print("The training dataset contains " + str(length) + " rows of data.")
        print("Datasets took " + str(time.time() - initial_time) + "s to generate")