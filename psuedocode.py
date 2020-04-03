import numpy as np

def SlidingWindowGenerator(file_name, crop, max_batch_size, batch_size):

    if lines_in_file <= maximum_ram_threshold:
        loaded_data = load_data_as_array(file=file_name, crop=crop)
        features = loaded_data[:, 0]
        outputs = loaded_data[:, 1]
        row_indicies = shuffle(arrange(max_batch_size))
        while True:
            for window_start in range (0, max_batch_size, batch_size):
                required_indicies = row_indicies[window_start : window_start + batch_size]
                window = features[index : index + window_length] (for index in required_indicies)
                targets = outputs[required_indicies + (0.5 * window_length - 1)]
                yield window,targets
