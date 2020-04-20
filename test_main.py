import argparse
from remove_space import remove_space
from seq2point_test import Tester

# Allows a model to be tested from the terminal.

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave. ")
parser.add_argument("--transfer_domain", type=remove_space, default="kettle", help="The appliance used to train the model that you would like to test (i.e. transfer learning). ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default="1000000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm of the model to test. Default is none. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")

arguments = parser.parse_args()

tester = Tester(arguments.appliance_name, arguments.pruning_algorithm, arguments.transfer_domain, arguments.crop, arguments.batch_size, arguments.network_type)
tester.test_model()