# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

"""Implement an ArgParser for main.py ."""

import argparse

def options():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser(description='Argparser for main configuration settings')
    # File paths
    parser.add_argument('--file', type=str, default=None,help='AES file path')
    parser.add_argument('--lib', type=str, default=None,help='Library Path')
    parser.add_argument('--dump', type=str, default=None,help="Root dump location")
    parser.add_argument('--params', type=str, default='params.yml',help="Parameters yaml filepath")

    return parser