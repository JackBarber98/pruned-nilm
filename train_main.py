import argparse
from remove_space import remove_space
from seq2point_train import train_model

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to train the network with. ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. ")
parser.add_argument("--crop", type=int, default="10000", help="The number of rows of the dataset to take training data from. ")
parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. ")

arguments = parser.parse_args()

train_model(arguments.appliance_name, arguments.pruning_algorithm, arguments.batch_size, arguments.crop)

