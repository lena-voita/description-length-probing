# Description Length and Control Tasks

<img src="resources/random_labels_with_head-min.png" width="150" align="right">

Our code is a modified version of [the original code](https://worksheets.codalab.org/worksheets/0xb0c351d6f1ac4c51b54f1023786bf6b2) for the [control tasks paper](https://www.aclweb.org/anthology/D19-1275.pdf). 

__Warning__! We used code from [the codalab worksheets](https://worksheets.codalab.org/worksheets/0xb0c351d6f1ac4c51b54f1023786bf6b2) and NOT from [the repo](https://github.com/john-hewitt/control-tasks), because we used the code referred to in the paper. We do not know if the code is different from the one in the repo.

## Loading data

Download data and ELMo vectors using [our link](https://drive.google.com/open?id=12e2Q0hllKWr73-aQNB1WdQCe0bJIN21G) or from [the original codalab worksheets](https://worksheets.codalab.org/worksheets/0xb0c351d6f1ac4c51b54f1023786bf6b2): the data is the same.

## Experiments

### Change config

Folder [./mdl_configs](./mdl_configs) contains examples of config files for variational and online codings. 

You have to change the lines marked by the `# CHANGE THIS: ...` comment:

* path to the ptb data,

* path to the elmo vectors,

* layer number,

* output directory with checkpoints and logs.

If you want, you can also change the lines marked by the `# CHANGE THIS (IF YOU REALLY WANT)` comment:

* number of probe layers

* size of the probe hidden layer.

### Run experiment

How to run an experiment:

```python /path_to_this_repo/control_tasks/run_experiment.py /path_to_this_repo/mdl_configs/bayes_l0.yml```


## Getting results and Evaluating MDL

Folder [./mdl_eval_notebooks](./mdl_eval_notebooks) contains notebooks shoing how to evaluate online and variational codelength from the logs, and how to get pruned probe architecture for the variational probe.
