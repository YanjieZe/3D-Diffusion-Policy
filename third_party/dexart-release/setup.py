import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(
    name='dexart',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/Kami-code/dexart-release',
    license='',
    author="Xiaolong Wang's Lab",
    install_requires=[
        'transforms3d', 'sapien==2.2.1', 'numpy', 'open3d>=0.15.1'
    ],
)
