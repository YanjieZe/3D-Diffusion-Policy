# Examples

Here we provide a job script to illustrate policy optimization with incrimental learning methods like NPG and PPO. To run the experiments, use the commands below. The experiments are run through the job script provided which tasks two arguments:
- `output`: path to directory where all the results will be saved
- `config`: a config `.txt` file with all the experiment parameters (examples are provided)
The script has to be run from this directory, i.e. `mjrl/examples` 

1. To train an NPG agent on a task shipped with `mjrl` (e.g. swimmer)
```
$ python policy_opt_job_script.py --output swimmer_npg_exp --config example_configs/swimmer_npg.txt
```

2. To train an NPG agent on an OpenAI gym benchmark task (e.g. hopper)
```
$ python policy_opt_job_script.py --output hopper_npg_exp --config example_configs/hopper_npg.txt
```
Note that since the Hopper env has termination conditions, we pick the sampling mode in the config to be `samples` rather than trajectories, so that per update we have 10K samples.

3. To train a PPO agent on the swimmer task
```
$ python policy_opt_job_script.py --output swimmer_ppo_exp --config example_configs/swimmer_ppo.txt
```