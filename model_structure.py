from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, Reshape
import os

def create_model():
    input_layer = Input(shape=(597,))
    reshape_layer = Reshape((-1, 597, 1))(input_layer)
    conv_layer_1 = Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    conv_layer_2 = Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
    conv_layer_3 = Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
    conv_layer_4 = Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
    conv_layer_5 = Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
    flatten_layer = Flatten()(conv_layer_5)
    label_layer = Dense(1024, activation="relu")(flatten_layer)
    output_layer = Dense(1, activation="linear")(label_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def save_model(model, path):
    if not os.path.exists (path + "_weights.h5"):
        open((path + "_weights.h5"), 'a').close()
    if not os.path.exists (path + ".h5"):
        open((path + ".h5"), 'a').close()
    model.save(path + ".h5")
    model.save_weights(path + "_weights.h5")

def load_model(model, path):
    try:
        print("PATH: ", path)
        model.load_weights(path + "_weights.h5")
        return model
    except:
        print("No saved model was found. (Takes the form of appliancename_weights.h5)")
        return None