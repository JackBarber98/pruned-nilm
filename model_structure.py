import tensorflow as tf 
import os

def create_model():

    """Specifies the structure of a seq2point model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = tf.keras.layers.Input(shape=(599,))
    reshape_layer = tf.keras.layers.Reshape((1, 599, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_dropout_model():

    """ Specifies the structure of a seq2point with dropout model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model with dropout layers.

    """

    input_layer = tf.keras.layers.Input(shape=(599,))
    reshape_layer = tf.keras.layers.Reshape((1, 599, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    dropout_layer_1 = tf.keras.layers.Dropout(0.5)(conv_layer_1)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_1)
    dropout_layer_2 = tf.keras.layers.Dropout(0.5)(conv_layer_2)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_2)
    dropout_layer_3 = tf.keras.layers.Dropout(0.5)(conv_layer_3)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_3)
    dropout_layer_4 = tf.keras.layers.Dropout(0.5)(conv_layer_4)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_4)
    dropout_layer_5 = tf.keras.layers.Dropout(0.5)(conv_layer_5)
    flatten_layer = tf.keras.layers.Flatten()(dropout_layer_5)
    label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
    dropout_layer_6 = tf.keras.layers.Dropout(0.5)(label_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(dropout_layer_6)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_reduced_model():

    """ Create a seq2point model with 10 fewer filters per convolutional layer and 2^9 instead 
    of 2^10 hidden layer neurons. 
    
    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model with fewer filters and hidden neurons.
    
    """

    input_layer = tf.keras.layers.Input(shape=(599,))
    reshape_layer = tf.keras.layers.Reshape((1, 599, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=20, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=20, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(4, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(4, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(512, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_reduced_dropout_model():

    """ Applies dropout to the reduced seq2point architecture. 
    
    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model with dropout layers, 
    fewer filters, and fewer hidden neurons.
    
    """

    input_layer = tf.keras.layers.Input(shape=(599,))
    reshape_layer = tf.keras.layers.Reshape((1, 599, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=20, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    dropout_layer_1 = tf.keras.layers.Dropout(0.5)(conv_layer_1)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=20, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_1)
    dropout_layer_2 = tf.keras.layers.Dropout(0.5)(conv_layer_2)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_2)
    dropout_layer_3 = tf.keras.layers.Dropout(0.5)(conv_layer_3)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(4, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_3)
    dropout_layer_4 = tf.keras.layers.Dropout(0.5)(conv_layer_4)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(4, 1), strides=(1, 1), padding="same", activation="relu")(dropout_layer_4)
    dropout_layer_5 = tf.keras.layers.Dropout(0.5)(conv_layer_5)
    flatten_layer = tf.keras.layers.Flatten()(dropout_layer_5)
    label_layer = tf.keras.layers.Dense(512, activation="relu")(flatten_layer)
    dropout_layer_6 = tf.keras.layers.Dropout(0.5)(label_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(dropout_layer_6)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model    

def save_model(model, network_type, algorithm, appliance):

    """ Saves a model to a specified location. Models are named using a combination of their 
    target appliance, architecture, and pruning algorithm.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """
    
    model_path = "./" + appliance + "/saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"

    if not os.path.exists (model_path):
        open((model_path), 'a').close()

    model.save(model_path)

def load_model(model, network_type, algorithm, appliance):

    """ Loads a model from a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """

    model_name = "./" + appliance + "/saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    print("PATH NAME: ", model_name)

    model = tf.keras.models.load_model(model_name)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model