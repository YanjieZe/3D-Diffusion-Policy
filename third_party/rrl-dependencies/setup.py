# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur

import os
import sys
from setuptools import setup, find_packages

print("Installing RRL. \n Package intended for use with provided conda env. See setup instructions here:")

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='RRL',
    version='1.0.0',
    packages=find_packages(),
    description='RL algorithm for environments in MuJoCo',
    author='Rutav Shah',
)
