import keras
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

    """Specifies the structure of a seq2point model using Keras' functional API.

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

def create_lite_model(path):
    converter = tf.lite.TFLiteConverter.from_saved_model(path + "_weights.h5")
    lite_model = converter.convert()
    
    if not os.path.exists(path + "_lite.tflite"):
        open((path + "_lite.tflite"), 'a').close()
    lite_model.save(path + "_lite.tflite")

def save_model(model, algorithm, path):

    """Saves a model to a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    path (string): The path to which the model will be saved.

    """

    if not os.path.exists (path + "_" + algorithm + "_weights.h5"):
        open((path + algorithm + "_" + "_weights.h5"), 'a').close()
    if not os.path.exists (path + "_" + algorithm + ".h5"):
        open((path + "_" + algorithm + ".h5"), 'a').close()
    model.save(path + "_" + algorithm + ".h5")
    model.save_weights(path + "_" + algorithm + "_weights.h5")

def load_model(model, path):

    """Loads a model from a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    path (string): The path from which the model will be load from.

    """

    model = tf.keras.models.load_model("./kettle/saved_model/kettle_model.h5")
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model