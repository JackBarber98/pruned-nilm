def batch_size(crop):
    while True:
        try:
            size = int(input("{0}".format("Enter the batch size: ")))
        except ValueError:
            print("Batch size must be an integer.")
            continue
        if size > crop:
            print("Batch size cannot have more rows than the training dataset")
            continue
        else:
            print()
            return size

def crop():
    while True:
        try:
            crop_size = int(input("{0}".format("Enter the number of rows of training data to use: ")))
        except ValueError:
            print("Crop size must be an integer.")
            continue
        else:
            print()
            return crop_size


def appliance():
    appliances = ["kettle", "microwave", "fridge", "dishwasher", "washing_machine"]
    while True:
        try:
            appliance = str(input("{0}".format("Enter an appliance (kettle, microwave, fridge, dishwasher, washing_machine): ")))
        except ValueError:
            print("Appliance must be a string.")
            continue
        if appliance not in appliances:
            print("That is not a valid appliance.")
            continue
        else:
            print()
            return appliance

def parameter_menu():
    app = appliance()
    crop_size = crop()
    batch = batch_size(crop_size)
    print("Appliance: {0}, Crop: {1}, Batch Size: {2}".format(app, crop_size, batch))

def model_menu():
    while True:
        try:
            transfer = str(input("{0}".format("Use transfer learning?")))
            if (transfer == "y" or transfer == "yes"):
                directory = str(input("{0}".format("Enter model directory: ")))
        except ValueError:
            print("This value must be a string.")
            continue
        else:
            print()
            transfer_appliance = appliance()
            return directory, transfer_appliance

def save_directory():
    while True:
        try:
            directory = str(input("{0}".format("Enter save directory: ")))
        except ValueError:
            print("This value must be a string")
            continue
        else:
            print()
            return directory