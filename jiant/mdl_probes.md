# Description Length and Random Models

<img src="../resources/random_model_with_head-min.png" width="150" align="right">

Our code is a modified version of [the original code](https://github.com/jsalt18-sentence-repl/jiant) for the [edge probing paper](https://arxiv.org/abs/1905.06316). 


## First steps

First, follow general instructions in the [main repo](.): setup the environment, load data for the edge probing tasks and learn how to launch experiments (e.g., their example experiment).

## Experiments

### Configs

Folder [./mdl_configs](./mdl_configs) contains examples of config files for variational and online codings. 

The fields you have to change are marked with the `CHANGE THIS!!!` comment. 

These fields are: task name, ELMO layer number, paths to the repo and the data.

### Run experiment

For pretrained model:

```python path_to_this_repo/jiant/main.py --config_file pos_online_layer1.conf --overrides "exp_name = pos_online_layer1, run_name = run1, project_dir = ."```

For randomly initalized model, download our model weights [`elmo_random.pth`](https://drive.google.com/open?id=1Izq2LKrAVIaLUetfgZcPR5IKjJkKas0i) and run:

```python path_to_this_repo/jiant/main.py --config_file pos_online_layer1.conf --overrides "exp_name = pos_online_layer1, run_name = run1, project_dir = ., elmo_from_pth = path_to_your_random_elmo/elmo_random.pth"```

## Getting results and Evaluating MDL

Folder [./mdl_eval_notebooks](./mdl_eval_notebooks) contains notebooks shoing how to evaluate online and variational codelength from the logs, and how to get pruned probe architecture for the variational probe.
