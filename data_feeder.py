import numpy as np 
import pandas as pd 

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.

class InputChunkSlider():

    """Yields features and targets for training a ConvNet.

    Parameters:
    file_name (string): The path where the training dataset is located.
    chunk_size (int): The size of each chunk of data to be processed.
    shuffle (bool): Whether the dataset should be shuffled before being returned.
    offset (int):
    batch_size (int): The size of each batch in a chunk.
    crop (int): The number of rows of the dataset to return.
    ram_threshold (int): The maximum amount of RAM to utilise at a time.

    """

    def __init__(self, file_name, chunk_size, shuffle, offset, batch_size=1000, crop=100000, ram_threshold=5 * 10 ** 5):
        self.file_name = file_name
        self.batch_size = batch_size
        self.chunk_size = 10 ** 8
        self.shuffle = shuffle
        self.offset = offset
        self.crop = crop
        self.ram_threshold = ram_threshold
        self.total_size = 0

    def check_if_chunking(self):

        """Count the number of rows in the dataset and determine whether this is larger than the chunking threshold or not.

        """

        # Loads the file and counts the number of rows it contains.
        print("Importing training file...")
        chunks = pd.read_csv(self.file_name, header=0, nrows=self.crop)
        print("Counting number of rows...")
        self.total_size = len(chunks)
        del chunks
        print("Done.")

        print("The dataset contains ", self.total_size, " rows")

        # Display a warning if there are too many rows to fit in the designated amount RAM.
        if (self.total_size > self.ram_threshold):
            print("There is too much data to load into memory, so it will be loaded in chunks. Please note that this may result in decreased training times.")


    def load_dataset(self):

        """Yields pairs of features and targets that will be used directly by a neural network for training.

        Yields:
        input_data (numpy.array): A 1D array of size batch_size containing features of a single input. 
        output_data (numpy.array): A 1D array of size batch_size containing the target values corresponding to each feature set.

        """

        if self.total_size == 0:
            self.check_if_chunking()

        # If the data can be loaded in one go, don't skip any rows.
        if (self.total_size <= self.ram_threshold):
            # Returns an array of the content from the CSV file.
            data_array = np.array(pd.read_csv(self.file_name, nrows=self.crop, header=0))
            inputs = data_array[:, 0]
            outputs = data_array[:, 1]

            maximum_batch_size = inputs.size - 2 * self.offset
            if self.batch_size < 0:
                self.batch_size = maximum_batch_size

            indicies = np.arange(maximum_batch_size)
            if self.shuffle:
                np.random.shuffle(indicies)

            while True:
                for start_index in range(0, maximum_batch_size, self.batch_size):
                    splice = indicies[start_index : start_index + self.batch_size]
                    input_data = np.array([inputs[index : index + 2 * self.offset + 1] for index in splice])
                    output_data = outputs[splice + self.offset].reshape(-1, 1)

                    yield input_data, output_data
        # Skip rows where needed to allow data to be loaded properly when there is not enough memory.
        if (self.total_size >= self.ram_threshold):
            indicies_to_skip = np.arange(self.total_size / self.chunk_size)
            if self.shuffle:
                np.random.shuffle(indicies_to_skip)

            # Yield the data in sections.
            for index in indicies_to_skip:
                data_array = np.array(pd.read_csv(self.file_name, skiprows=int(index) * self.chunk_size, header=self.header, nrows=self.crop))                   
                inputs = data_array[:, 0]
                outputs = data_array[:, 1]

                maximum_batch_size = inputs.size - 2 * self.offset
                if self.batch_size < 0:
                    self.batch_size = maximum_batch_size

                indicies = np.arange(maximum_batch_size)
                if self.shuffle:
                    np.random.shuffle(indicies)
            while True:
                for start_index in range(0, maximum_batch_size, self.batch_size):
                    splice = indicies[start_index : start_index + self.batch_size]
                    input_data = np.array([inputs[index : index + 2 * self.offset + 1] for index in splice])
                    output_data = outputs[splice + self.offset].reshape(-1, 1)

                    yield input_data, output_data

class TestingChunkSlider(object):

    """Yields features and targets for testing and validating a ConvNet.

    Parameters:
    number_of_windows (int): The number of sliding windows to produce.
    inputs (numpy.array): The available testing / validation features.
    offset (int):

    """

    def __init__(self, number_of_windows, inputs, targets, offset):
        self.number_of_windows = number_of_windows
        self.offset = offset
        self.inputs = inputs
        self.targets = targets
        self.total_size = len(inputs)

    def load_data(self):

        """Yields features and targets for testing and validating a ConvNet.

        Yields:
        input_data (numpy.array): An array of features to test / validate the network with.

        """

        self.inputs = self.inputs.flatten()
        max_number_of_windows = self.inputs.size - 2 * self.offset

        if self.number_of_windows < 0:
            self.number_of_windows = max_number_of_windows

        indicies = np.arange(max_number_of_windows, dtype=int)
        for start_index in range(0, max_number_of_windows, self.number_of_windows):
            splice = indicies[start_index : start_index + self.number_of_windows]
            input_data = np.array([self.inputs[index : index + 2 * self.offset + 1] for index in splice])
            target_data = self.targets[splice + self.offset].reshape(-1, 1)
            yield input_data, target_data


