import argparse
from remove_space import remove_space
from seq2point_train import Trainer

# Allows a model to be trained from the terminal.

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default="5000000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. Default is none. Available are: spp, entropic, threshold. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")

arguments = parser.parse_args()

trainer = Trainer(arguments.appliance_name, arguments.pruning_algorithm, arguments.batch_size, arguments.crop, arguments.network_type)
trainer.train_model()