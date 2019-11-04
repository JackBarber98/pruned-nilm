# JACK BARBER (2019) / 17633953 #
#
# Overview of Operation:
#  - One of the REFIT houses is used for testing, and another for validation
#  - The rest are used for training
#  - The training, testing, and validation datasets are extracted from the REFIT csv files
#  - Training data is merged from the different csv files to form a larger, single training file
#  - The training set should be 60,00,000 rows in length, or 3.6GB in size
#  - The datasets are saved to the ./[appliance] directory

import os
import time
import re
import pandas as pd

# PLACE THIS IN A SEPERATE FILE #
kettle_params = {
    "training": [3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
    "channels": [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
    "testing": 2,
    "testing-channel": 8,
    "validation": 5,
    "validation-channel": 8,
    "mean": 700,
    "std": 1000,
}

# Loads data about a specified appliance and returns the data as a DataFrame.
def load_file(directory, house, appliance, channel):
    file_name = directory + "CLEAN_House" + str(house) + ".csv"
    file_contents = pd.read_csv(file_name,
                                names=["aggregate", appliance],
                                usecols=[2, channel + 2],
                                header=0,
                                infer_datetime_format=True,
                                parse_dates=True,
                                memory_map=True)
    return file_contents

# Returns the digits present in the file name.
def digits_in_file_name(file_name):
    return int(re.search(r"\d+", file_name).group())

if __name__ == "__main__":
    initial_time = time.time()

    appliance = "kettle"
    directory = "./refit_dataset/"
    agg_mean = 522
    agg_std = 814
    length = 0
    print("Selected appliance: ", appliance)
    print("Dataset directory: ", directory)

    if not os.path.exists(appliance):
        os.makedirs(appliance)
    
    # Loops through files and folders found in the directory
    for index, file_name in enumerate(os.listdir(directory)):

        # Format the appliance's test data.
        if file_name == "CLEAN_House" + str(kettle_params["testing"]) + ".csv":
            print("Formatting " + appliance + " test data...")
            
            # Load the test data.
            test_data = load_file(directory,
                                kettle_params["testing"],
                                appliance,
                                kettle_params["testing-channel"]
                                )
        
            # Normalise the appliance's test data.
            test_data["aggregate"] = (test_data["aggregate"] - agg_mean) / agg_std
            test_data[appliance] = test_data[appliance] - kettle_params["mean"] / kettle_params["std"]

            # Save the test data.
            test_data.to_csv("./" + appliance + "/" + appliance + "_test_.csv", index=False)

            # Delete test data from memory.
            del test_data

        # Format the appliance's validation data.
        elif file_name == "CLEAN_House" + str(kettle_params["validation"]) + ".csv":
            print("Formatting " + appliance + " validation data...")
            
            # Load the validation data.
            validation_data = load_file(directory,
                                        kettle_params["validation"],
                                        appliance,
                                        kettle_params["validation-channel"])
            
            # Normalise the validation data.
            validation_data["aggregate"] = (validation_data["aggregate"] - agg_mean) / agg_std
            validation_data[appliance] = validation_data[appliance] - kettle_params["mean"] / kettle_params["std"]

            # Save validation data.
            validation_data.to_csv("./" + appliance + "./" + appliance + "_validation_.csv", index=False)

            # Delete validation data from memory.
            del validation_data

        # Format training data.
        elif digits_in_file_name(file_name) in kettle_params["training"]:
            try:
                print("Adding house " + str(digits_in_file_name(file_name)) + " to training dataset.")
                training_data = load_file(directory,
                                digits_in_file_name(file_name),
                                appliance,
                                kettle_params["channels"][kettle_params["training"].index(digits_in_file_name(file_name))]
                                )

                # Normalise the training data.
                training_data["aggregate"] = (training_data["aggregate"] - agg_mean) / agg_std
                training_data[appliance] = (training_data[appliance] - kettle_params["mean"]) / kettle_params["std"]
                rows, columns = training_data.shape
                
                length += rows

                training_data.to_csv("./" + appliance + "/" + appliance + "_training_.csv", mode="a", index=False, header=False)
                
                # Delete training data from memory.
                del training_data
            except:
                pass

    print("The training dataset contains " + str(length) + " rows of data.")
    print("Datasets took " + str(time.time() - initial_time) + "s to generate")