import argparse
from distutils.util import strtobool

str2bool = lambda x: bool(strtobool(x))

def deployment_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='input yaml file for experiment', type=str)
    parser.add_argument('n_workers', help='number of dask workers to deploy', type=int)
    parser.add_argument('--cloud', help='store results to cloud', type=str2bool, default=False)
    return vars(parser.parse_args())

def experiment_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='input yaml file for experiment', type=str)
    parser.add_argument('dask_addr', help='dak scheduler address', type=str)
    parser.add_argument('--cloud', help='store results to cloud', type=str2bool, default=False)
    return vars(parser.parse_args())