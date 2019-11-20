import numpy as np 
import pandas as pd 

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.

class InputChunkSlider():
    def __init__(self, file_name, chunk_size, shuffle, offset, batch_size=1000, crop=100000, header=0, ram_threshold=5 * 10 ** 5):
        self.file_name = file_name
        self.batch_size = 1000
        self.chunk_size = 10000
        self.shuffle = shuffle
        self.offset = offset
        self.crop = crop
        self.header = header
        self.ram_threshold = ram_threshold
        self.total_size = 0

    def check_if_chunking(self):
        # Loads the file and counts the number of rows it contains.
        chunks = pd.read_csv(self.file_name, header=self.header, nrows=self.crop)
        self.total_size = len(chunks)
        print("The dataset contains ", self.total_size, " rows")

        # Display a warning if there are too many rows to fit in the designated amount RAM.
        if (self.total_size > self.ram_threshold):
            print("There is too much data to load into memory, so it will be loaded in chunks. Please note that this may result in decreased training times.")


    def load_dataset(self):
        if self.total_size == 0:
            self.check_if_chunking()

        # If the data can be loaded in one go, don't skip any rows.
        if (self.total_size <= self.ram_threshold):
            # Returns an array of the content from the CSV file.
            data_array = np.array(pd.read_csv(self.file_name, nrows=self.crop, header=self.header))
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
    def __init__(self, number_of_windows, offset):
        self.number_of_windows = number_of_windows
        self.offset = offset

    def load_data(self, inputs):
        inputs = inputs.flatten()
        max_number_of_windows = inputs.size - 2 * self.offset

        if self.number_of_windows < 0:
            self.number_of_windows = max_number_of_windows

        indicies = np.arange(max_number_of_windows, dtype=int)
        for start_index in range(0, max_number_of_windows, self.number_of_windows):
            splice = indicies[start_index : start_index + self.number_of_windows]
            input_data = np.array([inputs[index : index + 2 * self.offset + 1] for index in splice])
            yield input_data


