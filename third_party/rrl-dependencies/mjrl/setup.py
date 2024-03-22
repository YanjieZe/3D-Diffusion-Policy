import os
import sys
from setuptools import setup, find_packages

print("Installing mjrl. \n Package intended for use with provided conda env. See setup instructions here: https://github.com/aravindr93/mjrl/tree/master/setup")

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mjrl',
    version='1.0.0',
    packages=find_packages(),
    description='RL algorithms for environments in MuJoCo',
    long_description=read('README.md'),
    url='https://github.com/aravindr93/mjrl.git',
    author='Aravind Rajeswaran',
)
