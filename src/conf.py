import sys
import importlib
from types import SimpleNamespace
import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

print("Using config file", parser_args.config)

args = importlib.import_module(parser_args.config).args

args["experiment_name"] = parser_args.config
print("experiment_name: {}".format(parser_args.config))
# args['DEBUG']=False
args = SimpleNamespace(**args)
