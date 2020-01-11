import argparse
from remove_space import remove_space
from seq2point_test import test_model

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to perform disaggregation with. ")
parser.add_argument("--transfer_domain", type=remove_space, default="kettle", help="The appliance used to train the model that you would like to test (i.e. transfer learning). ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. ")
parser.add_argument("--crop", type=int, default="10000", help="The number of rows of the dataset to take training data from. ")

arguments = parser.parse_args()

test_model(arguments.appliance_name, arguments.pruning_algorithm, arguments.batch_size, arguments.crop)