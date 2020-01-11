import keras
import tensorflow as tf 
import os

def create_model():
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

def save_model(model, path):
    if not os.path.exists (path + "_weights.h5"):
        open((path + "_weights.h5"), 'a').close()
    if not os.path.exists (path + ".h5"):
        open((path + ".h5"), 'a').close()
    model.save(path + ".h5")
    model.save_weights(path + "_weights.h5")

def load_model(model, path):
    model = tf.keras.models.load_model("./kettle/saved_model/kettle_model.h5")
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model