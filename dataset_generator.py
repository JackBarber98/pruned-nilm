import os
import time
import re
import pandas as pd
from appliance_data import appliance_data

APPLIANCE = "microwave"
DIRECTORY = "./refit_dataset/"
AGG_MEAN = 522
AGG_STD = 814

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

    print("Selected Appliance: ", APPLIANCE)
    print("Directory of Dataset: ", DIRECTORY)

    length = 0

    if not os.path.exists(APPLIANCE):
        os.makedirs(APPLIANCE)
    
    # Loops through files and folders found in the directory
    for index, file_name in enumerate(os.listdir(DIRECTORY)):

        # Format the appliance's test data.
        if file_name == "CLEAN_House" + str(appliance_data[APPLIANCE]["test_house"]) + ".csv":
            print("Formatting " + APPLIANCE + " test data...")
            
            # Load the test data.
            test_data = load_file(DIRECTORY,
                                appliance_data[APPLIANCE]["test_house"],
                                APPLIANCE,
                                appliance_data[APPLIANCE]['channels'][appliance_data[APPLIANCE]['houses']
                                    .index(appliance_data[APPLIANCE]['test_house'])]
                                )
        
            # Normalise the appliance's test data.
            test_data["aggregate"] = (test_data["aggregate"] - AGG_MEAN) / AGG_STD
            test_data[APPLIANCE] = (test_data[APPLIANCE] - appliance_data[APPLIANCE]["mean"]) / appliance_data[APPLIANCE]["std"]

            # Save the test data.
            test_data.to_csv("./" + APPLIANCE + "/" + APPLIANCE + "_test_.csv", index=False)

            # Delete test data from memory.
            del test_data

        # Format the appliance's validation data.
        elif file_name == "CLEAN_House" + str(appliance_data[APPLIANCE]["validation_house"]) + ".csv":
            print("Formatting " + APPLIANCE + " validation data...")
            
            # Load the validation data.
            validation_data = load_file(DIRECTORY,
                                        appliance_data[APPLIANCE]["validation_house"],
                                        APPLIANCE,
                                        appliance_data[APPLIANCE]['channels'][appliance_data[APPLIANCE]['houses']
                                            .index(appliance_data[APPLIANCE]['validation_house'])]
                                        )
            
            # Normalise the validation data.
            validation_data["aggregate"] = (validation_data["aggregate"] - AGG_MEAN) / AGG_STD
            validation_data[APPLIANCE] = (validation_data[APPLIANCE] - appliance_data[APPLIANCE]["mean"]) / appliance_data[APPLIANCE]["std"]

            # Save validation data.
            validation_data.to_csv("./" + APPLIANCE + "./" + APPLIANCE + "_validation_.csv", index=False)

            # Delete validation data from memory.
            del validation_data

        # Format training data.
        elif digits_in_file_name(file_name) in appliance_data[APPLIANCE]["houses"]:

            try:
                training_data = load_file(DIRECTORY,
                                digits_in_file_name(file_name),
                                APPLIANCE,
                                appliance_data[APPLIANCE]["channels"][appliance_data[APPLIANCE]["houses"]
                                    .index(digits_in_file_name(file_name))]
                                )

                # Normalise the training data.
                training_data["aggregate"] = (training_data["aggregate"] - AGG_MEAN) / AGG_STD
                training_data[APPLIANCE] = (training_data[APPLIANCE] - appliance_data[APPLIANCE]["mean"]) / appliance_data[APPLIANCE]["std"]
                rows, columns = training_data.shape
                
                length += rows

                training_data.to_csv("./" + APPLIANCE + "/" + APPLIANCE + "_training_.csv", mode="a", index=False, header=False)
                
                # Delete training data from memory.
                del training_data
            except:
                print("PASS")
                pass

    print("The training dataset contains " + str(length) + " rows of data.")
    print("Datasets took " + str(time.time() - initial_time) + "s to generate")