# Installation

A short guide to install this package is below. The package relies on `mujoco-py` which might be the trickiest part of the installation. See `known issues` below and also instructions from the mujoco-py [page](https://github.com/openai/mujoco-py) if you are stuck with mujoco-py installation.

The package can handle both `MuJoCo v1.5` as well as `MuJoCo v2.0`, but the former is not supported for future updates. We encourage you to use v2.0.

## Linux

- Download MuJoCo v2.0 binaries from the official [website](http://www.mujoco.org/) and also obtain the license key.
- Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`. Note that unzip of the MuJoCo binaries will generate `mujoco200_linux`. You need to rename the directory and place it at `~/.mujoco/mujoco200`.
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mujoco200/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
- Install this package using
```
$ conda update conda
$ cd <path/to/mjrl>
$ conda env create -f setup/env.yml
$ source activate mjrl-env
$ pip install -e .
```
- *NOTE 1:* If there are issues with install of pytorch, please follow instructions from the [pytorch website](https://pytorch.org/) to install it properly based on the specific version of CUDA (or CPU-only) you have.

- *NOTE 2:* If you encounter a patchelf error in mujoco_py install, you can fix this with the following command when inside the anaconda env: `conda install -c anaconda patchelf`. See this [page](https://github.com/openai/mujoco-py/issues/147) for additional info.

## Mac OS

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and also obtain the license key.
- Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200` (rename unzipped directory to this), and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`.
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mujoco200/bin:$LD_LIBRARY_PATH"
```
- Install this package using
```
$ conda update conda
$ cd path/to/mjrl
$ conda env create -f setup/env.yml
$ source activate mjrl-env
$ pip install -e .
```

- *NOTE 1:* If there are issues with install of pytorch, please follow instructions from the [pytorch website](https://pytorch.org/) to install it properly.

- *NOTE 2:* If you encounter a patchelf error in mujoco_py install, you can fix this with the following command when inside the anaconda env: `conda install -c anaconda patchelf`. See this [page](https://github.com/openai/mujoco-py/issues/147) for additional info.


## Known Issues

- Visualization in linux: If the linux system has a GPU, then mujoco-py does not automatically preload the correct drivers. We added an alias `MJPL` in bashrc (see instructions) which stands for mujoco pre-load. When runing any python script that requires rendering, prepend the execution with MJPL.
```
$ MJPL python script.py
```

- Errors related to osmesa during installation. This is a `mujoco-py` build error and would likely go away if the following command is used before creating the conda environment. If the problem still persists, please contact the developers of mujoco-py
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```

- If conda environment creation gets interrupted for some reason, you can resume it with the following:
```
$ conda env update -n mjrl-env -f setup/env.yml
```

- GCC error in Mac OS: If you get a GCC error from mujoco-py, you can get the correct version mujoco-py expects with `brew install gcc --without-multilib`. This may require uninstalling other versions of GCC that may have been previously installed with `brew remove gcc@6` for example. You can see which brew packages were already installed with `brew list`.

