import argparse
import os
import sys
import yaml


def get_arg_parser():
    """Convenience function to return an argument parser with common arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y", type=str, help="Input YAML file")
    return parser


def load_yaml(args):
    with open(args.yaml, "r") as f:
        data = yaml.safe_load(f)
    return data


def check_dirs(dirs, input_names=(), output_names=()):
    """Ensure that the specified input and output directories exist.

    Missing input directories abort with error, missing output directories are
    created.

    """
    for name in input_names:
        path = dirs[name]
        if not os.path.isdir(path):
            sys.exit(f"ERROR: Input directory {path} does not exist.")

    for name in output_names:
        path = dirs[name]
        if not os.path.isdir(path):
            os.mkdir(path)
