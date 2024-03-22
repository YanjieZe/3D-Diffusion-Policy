# RL for MuJoCo

This package  contains implementations of various RL algorithms for continuous control tasks simulated with [MuJoCo.](http://www.mujoco.org/)

# Installation
The main package dependencies are `MuJoCo`, `python=3.7`, `gym>=0.13`, `mujoco-py>=2.0`, and `pytorch>=1.0`. See `setup/README.md` ([link](https://github.com/aravindr93/mjrl/tree/master/setup#installation)) for detailed install instructions.

# Bibliography
If you find the package useful, please cite the following papers.
```
@INPROCEEDINGS{Rajeswaran-NIPS-17,
    AUTHOR    = {Aravind Rajeswaran and Kendall Lowrey and Emanuel Todorov and Sham Kakade},
    TITLE     = "{Towards Generalization and Simplicity in Continuous Control}",
    BOOKTITLE = {NIPS},
    YEAR      = {2017},
}

@INPROCEEDINGS{Rajeswaran-RSS-18,
    AUTHOR    = {Aravind Rajeswaran AND Vikash Kumar AND Abhishek Gupta AND
                 Giulia Vezzani AND John Schulman AND Emanuel Todorov AND Sergey Levine},
    TITLE     = "{Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations}",
    BOOKTITLE = {Proceedings of Robotics: Science and Systems (RSS)},
    YEAR      = {2018},
}
```

# Credits
This package is maintained by [Aravind Rajeswaran](http://homes.cs.washington.edu/~aravraj/) and other members of the [Movement Control Lab,](http://homes.cs.washington.edu/~todorov/) University of Washington Seattle.
