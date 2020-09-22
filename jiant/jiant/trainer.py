""" Trainer """
import copy
import glob
import itertools
import logging as log
import math
import os
import pickle
import random
import re
import time

import numpy as np
import torch
from allennlp.common import Params  # pylint: disable=import-error
from allennlp.common.checks import ConfigurationError  # pylint: disable=import-error
from allennlp.data.iterators import BasicIterator, BucketIterator  # pylint: disable=import-error
from allennlp.nn.util import device_mapping, move_to_device
from allennlp.training.learning_rate_schedulers import (  # pylint: disable=import-error
    LearningRateScheduler,
)
from allennlp.training.optimizers import Optimizer  # pylint: disable=import-error
from tensorboardX import SummaryWriter  # pylint: disable=import-error
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from jiant.tasks.seq2seq import Seq2SeqTask
from jiant.utils import config
from jiant.utils.utils import (
    assert_for_log,
    check_for_previous_checkpoints,
)  # pylint: disable=import-error

from jiant import BayesianLayers


def build_trainer_params(args, task_names, phase="pretrain"):
    """ Helper function which extracts trainer parameters from args.
    In particular, we want to search args for task specific training parameters.
    """

    def _get_attr(attr_name, default=None):
        if phase == "pretrain":
            # We ignore task-specific trainer attributes during pretraining.
            if default is not None:
                return default
            return args[attr_name]
        else:
            # During target task training, we get task-specific attributes if available.
            assert len(task_names) == 1
            return config.get_task_attr(args, task_names[0], attr_name, default)

    params = {}
    train_opts = [
        "optimizer",
        "lr",
        "batch_size",
        "lr_decay_factor",
        "lr_patience",
        "patience",
        "scheduler_threshold",
        "sent_enc",
        "d_hid",
        "max_grad_norm",
        "min_lr",
        "batch_size",
        "cuda",
        "keep_all_checkpoints",
        "val_data_limit",
        "max_epochs",
        "min_epochs",
        "dec_val_scale",

    ]
    if args.train_type == "SamplingMultiTaskTrainerOnlineCoding":
        train_opts += ["online_coding_load_checkpoint", "online_coding_portion_split"]
        # V-- handle shuffling params with default values for backward compatibility
        params["online_coding_pre_shuffle"] = int(args.get("online_coding_pre_shuffle", "0")) == 1
        params["online_coding_pre_shuffle_seed"] = int(args.get("online_coding_pre_shuffle_seed", 1337))
    for attr in train_opts:
        params[attr] = _get_attr(attr)

    # Special case: If no task-specific arg is found for these, we'll need to fall back to a
    # phase-specific default.
    params["max_vals"] = _get_attr(
        "max_vals", default=args.target_train_max_vals if phase == "target_train" else None
    )
    params["val_interval"] = _get_attr(
        "val_interval", default=args.target_train_val_interval if phase == "target_train" else None
    )
    params["training_data_fraction"] = _get_attr(
        "training_data_fraction",
        default=args.target_train_data_fraction
        if phase == "target_train"
        else args.pretrain_data_fraction,
    )

    return Params(params)


def build_trainer(
    args,
    task_names,
    model,
    run_dir,
    metric_should_decrease=True,
    train_type="SamplingMultiTaskTrainer",
    phase="pretrain",
):
    """Build a trainer from params.

    Parameters
    ----------
    params: Trainer parameters as built by build_trainer_params.
    model: A module with trainable parameters.
    run_dir: The directory where we save the models.

    Returns
    -------
    A trainer object, a trainer config object, an optimizer config object,
        and a scheduler config object.
    """
    params = build_trainer_params(args, task_names, phase)

    opt_params = {"type": params["optimizer"], "lr": params["lr"]}
    if params["optimizer"] == "adam":
        # AMSGrad is a flag variant of Adam, not its own object.
        opt_params["amsgrad"] = True
    elif params["optimizer"] == "bert_adam":
        # Transformer scheduler uses number of opt steps, if known in advance, to set the LR.
        # We leave it as -1 here (unknown) and set it if known later.
        opt_params["t_total"] = -1
        opt_params["warmup"] = 0.1
    opt_params = Params(opt_params)

    schd_params = Params(
        {
            "type": "reduce_on_plateau",
            "mode": "min" if metric_should_decrease else "max",
            "factor": params["lr_decay_factor"],
            "patience": params["lr_patience"],
            "threshold": params["scheduler_threshold"],
            "threshold_mode": "abs",
            "verbose": True,
        }
    )

    train_params = Params(
        {
            "cuda_device": params["cuda"],
            "patience": params["patience"],
            "grad_norm": params["max_grad_norm"],
            "val_interval": params["val_interval"],
            "max_vals": params["max_vals"],
            "lr_decay": 0.99,
            "min_lr": params["min_lr"],
            "keep_all_checkpoints": params["keep_all_checkpoints"],
            "val_data_limit": params["val_data_limit"],
            "max_epochs": params["max_epochs"],
            "min_epochs": params.get("min_epochs", 0),
            "dec_val_scale": params["dec_val_scale"],
            "training_data_fraction": params["training_data_fraction"],
        }
    )

    if train_type == "SamplingMultiTaskTrainer":
        trainer = SamplingMultiTaskTrainer.from_params(model, run_dir, copy.deepcopy(train_params))
    elif train_type == "SamplingMultiTaskTrainerBayes":
        trainer = SamplingMultiTaskTrainerBayes.from_params(model, run_dir, copy.deepcopy(train_params))
    elif train_type == "SamplingMultiTaskTrainerOnlineCoding":
        train_params["online_coding_load_checkpoint"] = params["online_coding_load_checkpoint"]
        train_params["online_coding_portion_split"] = params["online_coding_portion_split"]
        train_params["online_coding_pre_shuffle"] = params["online_coding_pre_shuffle"]
        train_params["online_coding_pre_shuffle_seed"] = params["online_coding_pre_shuffle_seed"]

        trainer = SamplingMultiTaskTrainerOnlineCoding.from_params(model, run_dir, copy.deepcopy(train_params))
    else:
        raise NotImplementedError("Not supported trainer: ", train_type)
    return trainer, train_params, opt_params, schd_params



class SamplingMultiTaskTrainer:
    def __init__(
        self,
        model,
        patience=2,
        val_interval=100,
        max_vals=50,
        serialization_dir=None,
        cuda_device=-1,
        grad_norm=None,
        grad_clipping=None,
        lr_decay=None,
        min_lr=None,
        keep_all_checkpoints=False,
        val_data_limit=5000,
        max_epochs=-1,
        min_epochs=0,
        dec_val_scale=100,
        training_data_fraction=1.0,
    ):
        """
        The training coordinator. Unusually complicated to handle MTL with tasks of
        diverse sizes.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        patience , optional (default=2)
            Number of validations to be patient before early stopping.
        val_metric , optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model after each validation. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        serialization_dir , optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device , optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        keep_all_checkpoints : If set, keep checkpoints from every validation. Otherwise, keep only
            best and (if different) most recent.
        val_data_limit: During training, use only the first N examples from the validation set.
            Set to -1 to use all.
        training_data_fraction: If set to a float between 0 and 1, load only the specified
            percentage of examples. Hashing is used to ensure that the same examples are loaded
            each epoch.
        """
        self._model = model

        self._patience = patience
        self._max_vals = max_vals
        self._val_interval = val_interval
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._lr_decay = lr_decay
        self._min_lr = min_lr
        self._keep_all_checkpoints = keep_all_checkpoints
        self._val_data_limit = val_data_limit
        self._max_epochs = max_epochs
        self._min_epochs = min_epochs
        self._dec_val_scale = dec_val_scale
        self._training_data_fraction = training_data_fraction
        self._task_infos = None
        self._metric_infos = None

        self._log_interval = 10  # seconds
        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

        self._TB_dir = None
        if self._serialization_dir is not None:
            self._TB_dir = os.path.join(self._serialization_dir, "tensorboard")
            self._TB_train_log = SummaryWriter(os.path.join(self._TB_dir, "train"))
            self._TB_validation_log = SummaryWriter(os.path.join(self._TB_dir, "val"))

    def _check_history(self, metric_history, cur_score, should_decrease=False):
        """
        Given a the history of the performance on a metric
        and the current score, check if current score is
        best so far and if out of patience.
        """
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        if should_decrease:
            index_of_last_improvement = metric_history.index(min(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (patience + 1)
        else:
            index_of_last_improvement = metric_history.index(max(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (patience + 1)

        return best_so_far, out_of_patience

    def _setup_training(
        self, tasks, batch_size, train_params, optimizer_params, scheduler_params, phase
    ):
        """ Set up the trainer by initializing task_infos and metric_infos, which
        track necessary information about the training status of each task and metric respectively.

        Returns:
            - task_infos (Dict[str:Dict[str:???]]): dictionary containing where each task_info
              contains:
                - iterator: a task specific (because it uses that task's fields to dynamically
                    batch) batcher
                - n_tr_batches: the number of training batches
                - tr_generator: generator object that returns the batches, set to repeat
                    indefinitely
                - loss: the accumulated loss (during training or validation)
                - n_batches_since_val: number of batches trained on since the last validation
                - total_batches_trained: number of batches trained over all validation checks
                - optimizer: a task specific optimizer, not used if the global optimizer is not
                    None
                - scheduler: a task specific scheduler, not used if the global optimizer is not
                    None
                - stopped: a bool indicating if that task is stopped or not (if it ran out of
                    patience or hit min lr)
                - last_log: the time we last logged progress for the task

            - metric_infos (Dict[str:Dict[str:???]]): dictionary containing metric information.
                Each metric should be the validation metric of a task, except {micro/macro}_avg,
                which are privileged to get an aggregate multi-task score. Each dict contains:
                - hist (List[float]): previous values of the metric
                - stopped (Bool): whether or not that metric is stopped or not
                - best (Tuple(Int, Dict)): information on the best value of that metric and when
                    it happened
        """
        task_infos = {task.name: {} for task in tasks}
        for task in tasks:
            task_info = task_infos[task.name]
            if (
                os.path.exists(os.path.join(self._serialization_dir, task.name)) is False
                and phase == "target_train"
            ):
                os.mkdir(os.path.join(self._serialization_dir, task.name))

            # Adding task-specific smart iterator to speed up training
            instance = [i for i in itertools.islice(task.train_data, 1)][0]
            pad_dict = instance.get_padding_lengths()
            sorting_keys = []
            for field in pad_dict:
                for pad_field in pad_dict[field]:
                    sorting_keys.append((field, pad_field))
            iterator = BucketIterator(
                sorting_keys=sorting_keys,
                max_instances_in_memory=10000,
                batch_size=batch_size,
                biggest_batch_first=True,
            )
            tr_generator = iterator(task.train_data, num_epochs=None)
            tr_generator = move_to_device(tr_generator, self._cuda_device)
            task_info["iterator"] = iterator

            if phase == "pretrain":
                # Warning: This won't be precise when training_data_fraction is set, since each
                #  example is included or excluded independently using a hashing function.
                # Fortunately, it doesn't need to be.
                task_info["n_tr_batches"] = math.ceil(
                    task.n_train_examples * self._training_data_fraction / batch_size
                )
            else:
                task_info["n_tr_batches"] = math.ceil(task.n_train_examples / batch_size)

            task_info["tr_generator"] = tr_generator
            task_info["loss"] = 0.0
            task_info["total_batches_trained"] = 0
            task_info["n_batches_since_val"] = 0
            # deepcopy b/c using Params pops values and we may want to reuse
            # the Params object later
            opt_params = copy.deepcopy(optimizer_params)
            if "t_total" in optimizer_params:
                # If we know in advance how many opt steps there will be, set it so the LR scheduler
                # can use that information. This should be the next validation after we hit the
                # epoch limit.
                if self._max_epochs > 0:
                    max_epochs_in_vals = math.ceil(
                        (task_info["n_tr_batches"] * self._max_epochs) / self._val_interval
                    )
                    val_limit = min(max_epochs_in_vals, self._max_vals)
                else:
                    val_limit = self._max_vals
                opt_params["t_total"] = val_limit * self._val_interval
            task_info["optimizer"] = Optimizer.from_params(train_params, opt_params)
            task_info["scheduler"] = LearningRateScheduler.from_params(
                task_info["optimizer"], copy.deepcopy(scheduler_params)
            )
            task_info["stopped"] = False
            task_info["last_log"] = time.time()

        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks] + ["micro_avg", "macro_avg"]
        metric_infos = {
            metric: {"hist": [], "stopped": False, "best": (-1, {})} for metric in all_metrics
        }
        self.task_to_metric_mapping = {task.val_metric: task.name for task in tasks}
        self._task_infos = task_infos
        self._metric_infos = metric_infos
        return task_infos, metric_infos

    def get_scaling_weights(self, scaling_method, num_tasks, task_names, task_n_train_examples):
        """

        Parameters
        ----------------
        scaling_method : str, scaling method
        num_tasks: int
        task_names: list of str
        task_n_train_examples: list of ints of number of examples per task

        Returns
        ----------------
        scaling weights: list of ints, to scale loss
        """
        if scaling_method == "uniform":
            scaling_weights = [1.0] * num_tasks
        elif scaling_method == "max_proportional":
            scaling_weights = task_n_train_examples.astype(float)
        elif scaling_method == "max_proportional_log":
            scaling_weights = np.log(task_n_train_examples)
        elif "max_power_" in scaling_method:
            scaling_power = float(scaling_method.strip("max_power_"))
            scaling_weights = task_n_train_examples ** scaling_power
        elif scaling_method == "max_inverse_log":
            scaling_weights = 1 / np.log(task_n_train_examples)
        elif scaling_method == "max_inverse":
            scaling_weights = 1 / task_n_train_examples
        # Weighting losses based on best validation step for each task from a previous uniform run,
        # normalized by the maximum validation step
        # eg. 'max_epoch_9_18_1_11_18_2_14_16_1'
        elif "max_epoch_" in scaling_method:
            epochs = scaling_method.strip("max_epoch_").split("_")
            assert len(epochs) == num_tasks, "Loss Scaling Error: epoch number not match."
            scaling_weights = np.array(list(map(int, epochs)))

        # normalized by max weight
        if "max" in scaling_method:
            scaling_weights = scaling_weights / np.max(scaling_weights)

        scaling_weights = dict(zip(task_names, scaling_weights))
        return scaling_weights

    def get_sampling_weights(
        self, weighting_method, num_tasks, task_n_train_examples, task_n_train_batches
    ):
        """
        Parameters
        ----------------
        weighting_method: str, weighting method
        num_tasks: int
        task_n_train_examples: list of ints of number of examples per task
        task_n_train_batches: list of ints of number of batches per task

        Returns
        ----------------
        sampling weights: list of ints, to sample tasks to train on
        """
        if weighting_method == "uniform":
            sample_weights = [1.0] * num_tasks
        elif weighting_method == "proportional":
            sample_weights = task_n_train_examples.astype(float)
        elif weighting_method == "proportional_log_batch":
            sample_weights = np.log(task_n_train_batches)
        elif weighting_method == "proportional_log_example":
            sample_weights = np.log(task_n_train_examples)
        elif weighting_method == "inverse":
            sample_weights = 1 / task_n_train_examples
        elif weighting_method == "inverse_log_example":
            sample_weights = 1 / np.log(task_n_train_examples)
        elif weighting_method == "inverse_log_batch":
            sample_weights = 1 / np.log(task_n_train_batches)
        elif "power_" in weighting_method:
            weighting_power = float(weighting_method.strip("power_"))
            sample_weights = task_n_train_examples ** weighting_power
        elif "softmax_" in weighting_method:  # exp(x/temp)
            weighting_temp = float(weighting_method.strip("softmax_"))
            sample_weights = np.exp(task_n_train_examples / weighting_temp)
        else:
            raise KeyError(f"Unknown weighting method: {weighting_method}")
        return sample_weights

    def train(
        self,
        tasks,
        stop_metric,
        batch_size,
        weighting_method,
        scaling_method,
        train_params,
        optimizer_params,
        scheduler_params,
        load_model=1,
        phase="pretrain",
    ):
        """
        The main training loop.
        Training will stop if we run out of patience or hit the minimum learning rate.

        Parameters
        ----------
        tasks: a list of task objects to train on
        stop_metric: str, metric to use for early stopping
        batch_size: int, batch size to use for the tasks
        weighting_method: str, how to sample which task to use
        scaling_method:  str, how to scale gradients
        train_params: trainer config object
        optimizer_params: optimizer config object
        scheduler_params: scheduler config object
        load_model: bool, whether to restore and continue training if a checkpoint is found
        phase: str, usually 'pretrain' or 'target_train'

        Returns
        ----------
        Validation results
        """
        validation_interval = self._val_interval
        task_infos, metric_infos = self._setup_training(
            tasks, batch_size, train_params, optimizer_params, scheduler_params, phase
        )

        optimizer_params = copy.deepcopy(optimizer_params)
        if "t_total" in optimizer_params:
            # If we know in advance how many opt steps there will be, set it so the LR scheduler
            # can use that information. This should be the next validation after we hit the epoch
            # limit.
            if self._max_epochs > 0:
                n_epoch_steps = sum([info["n_tr_batches"] for info in task_infos.values()])
                max_epochs_in_vals = math.ceil(
                    (n_epoch_steps * self._max_epochs) / self._val_interval
                )
                val_limit = min(max_epochs_in_vals, self._max_vals)
            else:
                val_limit = self._max_vals
            optimizer_params["t_total"] = val_limit * self._val_interval

        optimizer = Optimizer.from_params(train_params, optimizer_params)
        scheduler = LearningRateScheduler.from_params(optimizer, copy.deepcopy(scheduler_params))
        self._optimizer = optimizer
        self._scheduler = scheduler

        # define these here b/c they might get overridden on load

        n_step, should_stop = 0, False
        if self._serialization_dir is not None:
            # Resume from serialization path
            if load_model:
                ckpt_directory, _, _ = check_for_previous_checkpoints(
                    self._serialization_dir, tasks, phase, load_model
                )
                if ckpt_directory is None:
                    log.warning(
                        "load_model=1 but there is not checkpoint. \
                        Starting training without restoring from a checkpoint."
                    )
                else:
                    n_step, should_stop = self._restore_checkpoint(phase, tasks)
                    log.info("Loaded model from checkpoint. Starting at step %d.", n_step)
            else:
                log.info("Starting training without restoring from a checkpoint.")
                check_for_previous_checkpoints(self._serialization_dir, tasks, phase, load_model)
        if self._grad_clipping is not None:  # pylint: disable=invalid-unary-operand-type

            def clip_function(grad):
                return grad.clamp(-self._grad_clipping, self._grad_clipping)

            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        # Calculate per task sampling weights
        assert_for_log(len(tasks) > 0, "Error: Expected to sample from 0 tasks.")

        task_names = [task.name for task in tasks]
        task_n_train_examples = np.array([task.n_train_examples for task in tasks])
        task_n_train_batches = np.array([task_infos[task.name]["n_tr_batches"] for task in tasks])
        log.info(
            "Training examples per task, before any subsampling: "
            + str(dict(zip(task_names, task_n_train_examples)))
        )
        if len(tasks) > 1:
            sample_weights = self.get_sampling_weights(
                weighting_method, len(tasks), task_n_train_examples, task_n_train_batches
            )

            normalized_sample_weights = np.array(sample_weights) / sum(sample_weights)
            log.info(
                "Using weighting method: %s, with normalized sample weights %s ",
                weighting_method,
                np.array_str(normalized_sample_weights, precision=4),
            )
            scaling_weights = self.get_scaling_weights(
                scaling_method, len(tasks), task_names, task_n_train_examples
            )
        else:
            sample_weights = normalized_sample_weights = [1.0]
            scaling_weights = {task_names[0]: 1.0}

        # Sample the tasks to train on. Do it all at once (val_interval) for
        # MAX EFFICIENCY.
        samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

        offset = 0
        all_tr_metrics = {}
        log.info("Beginning training with stopping criteria based on metric: %s", stop_metric)
        while not should_stop:
            self._model.train()
            task = samples[(n_step + offset) % validation_interval]  # randomly select a task
            task_info = task_infos[task.name]
            if task_info["stopped"]:
                offset += 1
                continue
            tr_generator = task_info["tr_generator"]
            total_batches_trained = task_info["total_batches_trained"]
            n_batches_since_val = task_info["n_batches_since_val"]
            tr_loss = task_info["loss"]

            for batch in itertools.islice(tr_generator, 1):
                n_batches_since_val += 1
                total_batches_trained += 1
                optimizer.zero_grad()
                output_dict = self._forward(batch, task=task)
                assert_for_log(
                    "loss" in output_dict, "Model must return a dict containing a 'loss' key"
                )
                loss = output_dict["loss"]  # optionally scale loss
                loss *= scaling_weights[task.name]

                loss.backward()
                assert_for_log(not torch.isnan(loss).any(), "NaNs in loss.")
                tr_loss += loss.data.cpu().numpy()

                # Gradient regularization and application
                if self._grad_norm:
                    clip_grad_norm_(self._model.parameters(), self._grad_norm)
                optimizer.step()
                n_step += 1  # update per batch

                # step scheduler if it's not ReduceLROnPlateau
                if not isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                    scheduler.step_batch(n_step)

            # Update training progress on that task
            task_info["n_batches_since_val"] = n_batches_since_val
            task_info["total_batches_trained"] = total_batches_trained
            task_info["loss"] = tr_loss

            # Intermediate log to logger and tensorboard
            if time.time() - task_info["last_log"] > self._log_interval:
                task_metrics = task.get_metrics()

                # log to tensorboard
                if self._TB_dir is not None:
                    task_metrics_to_TB = task_metrics.copy()
                    task_metrics_to_TB["loss"] = float(task_info["loss"] / n_batches_since_val)
                    self._metrics_to_tensorboard_tr(n_step, task_metrics_to_TB, task.name)

                task_metrics["%s_loss" % task.name] = tr_loss / n_batches_since_val
                description = self._description_from_metrics(task_metrics)
                log.info(
                    "Update %d: task %s, batch %d (%d): %s",
                    n_step,
                    task.name,
                    n_batches_since_val,
                    total_batches_trained,
                    description,
                )
                task_info["last_log"] = time.time()

                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric()
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

            # Validation
            if n_step % validation_interval == 0:

                # Dump and log all of our current info
                n_val = int(n_step / validation_interval)
                log.info("***** Step %d / Validation %d *****", n_step, n_val)
                # Get metrics for all training progress so far
                for task in tasks:
                    task_info = task_infos[task.name]
                    n_batches_since_val = task_info["n_batches_since_val"]
                    if n_batches_since_val > 0:
                        task_metrics = task.get_metrics(reset=True)
                        for name, value in task_metrics.items():
                            all_tr_metrics["%s_%s" % (task.name, name)] = value
                        # Updating loss from training
                        all_tr_metrics["%s_loss" % task.name] = float(
                            task_info["loss"] / n_batches_since_val
                        )
                    else:
                        all_tr_metrics["%s_loss" % task.name] = 0.0
                    log.info(
                        "%s: trained on %d batches, %.3f epochs",
                        task.name,
                        n_batches_since_val,
                        n_batches_since_val / task_info["n_tr_batches"],
                    )
                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric(reset=True)
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

                # Validate
                log.info("Validating...")
                all_val_metrics, should_save, new_best = self._validate(n_val, tasks, batch_size)

                # Check stopping conditions
                should_stop = self._check_stop(n_val, stop_metric, tasks)

                # Log results to logger and tensorboard
                for name, value in all_val_metrics.items():
                    log_str = "%s:" % name
                    if name in all_tr_metrics:
                        log_str += " training: %3f" % all_tr_metrics[name]
                    log_str += " validation: %3f" % value
                    log.info(log_str)
                if self._TB_dir is not None:
                    self._metrics_to_tensorboard_val(n_step, all_val_metrics)
                log.info(f"Global learning rate: {self._optimizer.param_groups[0]['lr']}")
                elmo_params = self._model.get_elmo_mixing_weights(tasks)
                if elmo_params:  # log ELMo mixing weights
                    for task_name, task_params in elmo_params.items():
                        log.info("ELMo mixing weights for {}:".format(task_name))
                        log.info(
                            "\t"
                            + ", ".join(
                                [
                                    "{}: {:.6f}".format(layer, float(param))
                                    for layer, param in task_params.items()
                                ]
                            )
                        )

                # Reset training preogress
                all_tr_metrics = {}
                samples = random.choices(
                    tasks, weights=sample_weights, k=validation_interval
                )  # pylint: disable=no-member

                if should_save:
                    self._save_checkpoint(
                        {"step": n_step, "validation_pass": n_val, "should_stop": should_stop},
                        tasks=tasks,
                        phase=phase,
                        new_best=new_best,
                    )

        log.info("Stopped training after %d validation checks", n_step / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)  # , validation_interval)

    def _aggregate_results(self, tasks, task_infos, metric_infos):
        """ Helper function to print results after finishing training """
        results = {}
        for task in tasks:
            task_info = task_infos[task.name]
            log.info(
                "Trained %s for %d batches or %.3f epochs",
                task.name,
                task_info["total_batches_trained"],
                task_info["total_batches_trained"] / task_info["n_tr_batches"],
            )
            # * validation_interval
            results[task.name] = metric_infos[task.val_metric]["best"][0]
        # * validation_interval
        results["micro"] = metric_infos["micro_avg"]["best"][0]
        # * validation_interval
        results["macro"] = metric_infos["macro_avg"]["best"][0]
        log.info("***** VALIDATION RESULTS *****")
        for metric in metric_infos.keys():
            best_val_pass, val_pass_metrics = metric_infos[metric]["best"]
            all_metrics_str = ", ".join(
                ["%s: %.5f" % (metric, score) for metric, score in val_pass_metrics.items()]
            )
            log.info("%s (for best val pass %d): %s", metric, best_val_pass, all_metrics_str)
        return results

    def _update_metric_history(
        self,
        val_pass,
        all_val_metrics,
        metric,
        task_name,
        metric_infos,
        metric_decreases,
        should_save,
        new_best,
    ):
        """
        This function updates metric history with the best validation score so far.

        Parameters
        ---------
        val_pass: int.
        all_val_metrics: dict with performance on current validation pass.
        metric: str, name of metric
        task_name: str, name of task
        metric_infos: dict storing information about the various metrics
        metric_decreases: bool, marker to show if we should increase or
        decrease validation metric.
        should_save: bool, for checkpointing
        new_best: bool, indicator of whether the previous best preformance score was exceeded

        Returns
        ________
        metric_infos: dict storing information about the various metrics
        this_val_metric: dict, metric information for this validation pass, used for optimization
            scheduler
        should_save: bool
        new_best: bool
        """
        this_val_metric = all_val_metrics[metric]
        metric_history = metric_infos[metric]["hist"]
        metric_history.append(this_val_metric)
        is_best_so_far, out_of_patience = self._check_history(
            metric_history, this_val_metric, metric_decreases
        )
        if is_best_so_far:
            log.info("Best result seen so far for %s.", task_name)
            metric_infos[metric]["best"] = (val_pass, all_val_metrics)
            should_save = True
            if task_name == "macro":
                new_best = True
        if out_of_patience:
            metric_infos[metric]["stopped"] = True
            # Commented out the below line as more confusing than helpful. May make sense to
            # restore if we wind up using more complex stopping strategies.
            # log.info("Out of early stopping patience. Stopped tracking %s.", task_name)
        return metric_infos, this_val_metric, should_save, new_best

    def _calculate_validation_performance(
        self,
        task,
        task_infos,
        tasks,
        batch_size,
        all_val_metrics,
        n_examples_overall,
        print_output=True,
    ):
        """
        Builds validation generator, evaluates on each task and produces validation metrics.

        Parameters
        ----------
        task: current task to get validation performance of
        task_infos: Instance of information about the task (see _setup_training for definition)
        tasks: list of task objects to train on
        batch_size: int, batch size to use for the tasks
        all_val_metrics: dictionary. storing the validation performance
        n_examples_overall: int, current number of examples the model is validated on
        print_output: bool, prints one example per validation

        Returns
        -------
        n_examples_overall: int, current number of examples
        task_infos: updated Instance with reset training progress
        all_val_metrics: dictinary updated with micro and macro average validation performance
        """
        n_examples, batch_num = 0, 0
        task_info = task_infos[task.name]
        # to speed up training, we evaluate on a subset of validation data
        if self._val_data_limit >= 0:
            max_data_points = min(task.n_val_examples, self._val_data_limit)
        else:
            max_data_points = task.n_val_examples
        val_generator = BasicIterator(batch_size, instances_per_epoch=max_data_points)(
            task.val_data, num_epochs=1, shuffle=False
        )
        val_generator = move_to_device(val_generator, self._cuda_device)
        n_val_batches = math.ceil(max_data_points / batch_size)
        all_val_metrics["%s_loss" % task.name] = 0.0

        for batch in val_generator:
            batch_num += 1
            with torch.no_grad():
                out = self._forward(batch, task=task)
            loss = out["loss"]
            all_val_metrics["%s_loss" % task.name] += loss.data.cpu().numpy()

            if print_output:
                if isinstance(task, Seq2SeqTask):
                    if batch_num == 1:
                        voc_src = self._model.vocab.get_index_to_token_vocabulary("tokens")
                        voc_trg = self._model.vocab.get_index_to_token_vocabulary(
                            task.name + "_tokens"
                        )
                        inputs = batch["inputs"]["words"][0][1:]
                        gold = batch["targs"]["words"][0][1:]

                        for i in range(out["predictions"].shape[1]):
                            output = out["predictions"][0][i]
                            input_string, gold_string, output_string = task.get_prediction(
                                voc_src, voc_trg, inputs, gold, output
                            )
                            if i == 0:
                                log.info("\tInput:\t%s", input_string)
                                log.info("\tGold:\t%s", gold_string)
                            log.info("\tOutput:\t%s", output_string)

            n_examples += out["n_exs"]

            # log
            if time.time() - task_info["last_log"] > self._log_interval:
                task_metrics = task.get_metrics()
                task_metrics["%s_loss" % task.name] = (
                    all_val_metrics["%s_loss" % task.name] / batch_num
                )
                description = self._description_from_metrics(task_metrics)
                log.info(
                    "Evaluate: task %s, batch %d (%d): %s",
                    task.name,
                    batch_num,
                    n_val_batches,
                    description,
                )
                task_info["last_log"] = time.time()
        assert batch_num == n_val_batches

        # Get task validation metrics and store in all_val_metrics
        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_val_metrics["%s_%s" % (task.name, name)] = value
        all_val_metrics["%s_loss" % task.name] /= batch_num  # n_val_batches
        # compute task contribution to macro and micro averages
        n_examples_overall += n_examples
        if task.val_metric_decreases and len(tasks) > 1:
            all_val_metrics["micro_avg"] += (
                1 - all_val_metrics[task.val_metric] / self._dec_val_scale
            ) * n_examples
            all_val_metrics["macro_avg"] += (
                1 - all_val_metrics[task.val_metric] / self._dec_val_scale
            ) / len(tasks)
        else:
            # triggers for single-task cases and during MTL when task val metric increases
            all_val_metrics["micro_avg"] += all_val_metrics[task.val_metric] * n_examples
            all_val_metrics["macro_avg"] += all_val_metrics[task.val_metric] / len(tasks)

        # Reset training progress
        task_info["n_batches_since_val"] = 0
        task_info["loss"] = 0
        return n_examples_overall, task_infos, all_val_metrics

    def _validate(self, val_pass, tasks, batch_size, periodic_save=True):
        """

        Validate on all tasks and return the results and whether to save this validation
        pass or not.

        Parameters
        ----------
        val_pass: int
        tasks: list of task objects to train on
        batch_size: int, the batch size to use for the tasks.periodic_save
        periodic_save: bool, value of whether or not to save model and progress periodically

        Returns
        __________
        all_val_metrics: dictinary updated with micro and macro average validation performance
        should_save: bool, determines whether to save a checkpoint
        new_best: bool, whether or not the macro performance increased
        """
        task_infos, metric_infos = self._task_infos, self._metric_infos
        scheduler = self._scheduler
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        all_val_metrics["macro_avg"] = 0.0
        all_val_metrics["micro_avg"] = 0.0
        n_examples_overall = 0.0

        # Get validation numbers for each task
        for task in tasks:
            n_examples_overall, task_infos, all_val_metrics = self._calculate_validation_performance(  # noqa
                task, task_infos, tasks, batch_size, all_val_metrics, n_examples_overall
            )
        # scale the micro avg contributions w/ total size of validation set.
        if "micro_avg" in all_val_metrics:
            all_val_metrics["micro_avg"] /= n_examples_overall
        # Track per task patience
        should_save = periodic_save  # whether to save this validation pass or not.
        # Currently we save every validation in the main training runs.
        new_best = False  # whether this validation pass is a new best

        # update metric infos
        for task in tasks + ["micro", "macro"]:
            if task in ["micro", "macro"]:
                metric = "%s_avg" % task
                metric_decreases = tasks[0].val_metric_decreases if len(tasks) == 1 else False
                task_name = task
            else:
                metric = task.val_metric
                metric_decreases = task.val_metric_decreases
                task_name = task.name
            if metric_infos[metric]["stopped"]:
                continue
            metric_infos, this_val_metric, should_save, new_best = self._update_metric_history(
                val_pass,
                all_val_metrics,
                metric,
                task_name,
                metric_infos,
                metric_decreases,
                should_save,
                new_best,
            )

            # Get scheduler, and update using macro score
            # micro has no scheduler updates
            if task_name == "macro" and isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                log.info("Updating LR scheduler:")
                scheduler.step(this_val_metric, val_pass)
                log.info(
                    "\tBest result seen so far for %s: %.3f", metric, scheduler.lr_scheduler.best
                )
                log.info(
                    "\t# validation passes without improvement: %d",
                    scheduler.lr_scheduler.num_bad_epochs,
                )

        return all_val_metrics, should_save, new_best

    def _check_stop(self, val_n, stop_metric, tasks):
        """ Check to see if should stop """
        task_infos, metric_infos = self._task_infos, self._metric_infos
        optimizer = self._optimizer

        for task in tasks:
            task_info = task_infos[task.name]
            n_epochs_trained = task_info["total_batches_trained"] / task_info["n_tr_batches"]
            if n_epochs_trained < self._min_epochs:
                return False

        should_stop = False
        if self._max_epochs > 0:  # check if max # epochs hit
            for task in tasks:
                task_info = task_infos[task.name]
                n_epochs_trained = task_info["total_batches_trained"] / task_info["n_tr_batches"]
                if n_epochs_trained >= self._max_epochs:
                    # Commented out the below line as more confusing than helpful. May make sense
                    # to restore if we wind up using more complex stopping strategies.
                    # log.info("Reached max_epochs limit for %s.", task.name)
                    task_info["stopped"] = True
            stop_epochs = min([info["stopped"] for info in task_infos.values()])
            if stop_epochs:
                log.info("Reached max_epochs limit on all tasks. Stopping training.")
                should_stop = True

        if optimizer.param_groups[0]["lr"] < self._min_lr:
            log.info("Minimum LR reached. Stopping training.")
            should_stop = True

        # check if validation metric is stopped
        stop_metric = metric_infos[stop_metric]["stopped"]
        if stop_metric:
            log.info("Ran out of early stopping patience. Stopping training.")
            should_stop = True

        # check if max number of validations hit
        stop_val = bool(val_n >= self._max_vals)
        if stop_val:
            log.info("Maximum number of validations reached. Stopping training.")
            should_stop = True

        return should_stop

    def _forward(self, batch, task=None):
        tensor_batch = move_to_device(batch, self._cuda_device)
        model_out = self._model.forward(task, tensor_batch)
        return model_out

    def _description_from_metrics(self, metrics):
        # pylint: disable=no-self-use
        """ format some metrics as a string """
        return ", ".join(["%s: %.4f" % (name, value) for name, value in metrics.items()])

    def _unmark_previous_best(self, phase, val_pass, task_dir_name=""):
        marked_best = glob.glob(
            os.path.join(
                self._serialization_dir, task_dir_name, "*_state_{}_val_*.best.th".format(phase)
            )
        )
        for file in marked_best:
            # Skip the just-written checkpoint.
            if "_{}.".format(val_pass) not in file:
                os.rename(file, re.sub("%s$" % (".best.th"), ".th", file))

    def _delete_old_checkpoints(self, phase, val_pass, task_dir_name=""):
        candidates = glob.glob(
            os.path.join(
                self._serialization_dir, task_dir_name, "*_state_{}_val_*.th".format(phase)
            )
        )
        for file in candidates:
            # Skip the best, because we'll need it.
            # Skip the just-written checkpoint.
            if ".best" not in file and "_{}.".format(val_pass) not in file:
                os.remove(file)

    def _save_checkpoint(self, training_state, phase="pretrain", new_best=False, tasks=None):
        """
        Parameters
        ----------
        training_state: An object containing trainer state (step number, etc.), to be saved.
        phase: Usually 'pretrain' or 'target_train'.
        new_best: If true, the saved checkpoint will be marked with .best_macro, and
            potentially used later when switching from pretraining to target task training.
        """
        if not self._serialization_dir:
            raise ConfigurationError(
                "serialization_dir not specified - cannot "
                "restore a model without a directory path."
            )
        log.info("Saving checkpoints to: %s", self._serialization_dir)

        val_pass = training_state["validation_pass"]
        if new_best:
            best_str = ".best"
        else:
            best_str = ""

        task_dir_name = ""

        if phase == "target_train":
            # We only pass in one task at a time during target train phase.
            assert len(tasks) == 1
            task_dir_name = tasks[0].name

        model_path = os.path.join(
            self._serialization_dir,
            task_dir_name,
            "model_state_{}_val_{}{}.th".format(phase, val_pass, best_str),
        )

        model_state = self._model.state_dict()

        # Skip non-trainable params, like the main ELMo params.
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                del model_state[name]

        task_states = {}
        for task_name, task_info in self._task_infos.items():
            task_states[task_name] = {}
            task_states[task_name]["total_batches_trained"] = task_info["total_batches_trained"]
            task_states[task_name]["stopped"] = task_info["stopped"]
        task_states["global"] = {}
        task_states["global"]["optimizer"] = self._optimizer.state_dict()
        # NOTE(Alex): AllenNLP wrapper doesn't expose scheduler state dict methods
        task_states["global"]["scheduler"] = self._scheduler.lr_scheduler.state_dict()

        metric_states = {}
        for metric_name, metric_info in self._metric_infos.items():
            metric_states[metric_name] = {}
            metric_states[metric_name]["hist"] = metric_info["hist"]
            metric_states[metric_name]["stopped"] = metric_info["stopped"]
            metric_states[metric_name]["best"] = metric_info["best"]

        torch.save(
            task_states,
            os.path.join(
                self._serialization_dir,
                task_dir_name,
                "task_state_{}_val_{}{}.th".format(phase, val_pass, best_str),
            ),
        )
        torch.save(
            metric_states,
            os.path.join(
                self._serialization_dir,
                task_dir_name,
                "metric_state_{}_val_{}{}.th".format(phase, val_pass, best_str),
            ),
        )
        torch.save(model_state, model_path)
        model_path_full = os.path.join(
            self._serialization_dir,
            task_dir_name,
            "modelfull_state_{}_val_{}{}.th".format(phase, val_pass, best_str),
        )
        torch.save(self._model, model_path_full)
        torch.save(
            training_state,
            os.path.join(
                self._serialization_dir,
                task_dir_name,
                "training_state_{}_val_{}{}.th".format(phase, val_pass, best_str),
            ),
        )
        if new_best:
            self._unmark_previous_best(phase, val_pass, task_dir_name)

        if not self._keep_all_checkpoints:
            self._delete_old_checkpoints(phase, val_pass, task_dir_name)

    def _restore_checkpoint(self, phase, tasks=None, override_suffix=None):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes a validation pass count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training since
        it will load previous checkpoints.

        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        We restore based on the phase. If phase=target_train, we start from the last
        target task and work backwards, to find the most recent checkpoint in the target
        train phase. If phase=pretrain, we check for checkpoints in the main run
        directory.

        Returns
        -------
        val_pass: the validation pass at which to resume training.
        """

        task_directory, val_pass, suffix = check_for_previous_checkpoints(
            self._serialization_dir, tasks, phase, load_model=True
        )
        if override_suffix is not None:
            print(f"Overriding {suffix}, reading {override_suffix} instead")
            suffix = override_suffix
        
        assert val_pass > -1, "No checkpoint found."
        log.info("Found checkpoint {}. Loading.".format(suffix))
        if task_directory is None:
            task_directory = ""
        model_path = os.path.join(
            self._serialization_dir, task_directory, "_".join(["model", suffix])
        )
        training_state_path = os.path.join(
            self._serialization_dir, task_directory, "_".join(["training", suffix])
        )
        task_state_path = os.path.join(
            self._serialization_dir, task_directory, "_".join(["task", suffix])
        )
        metric_state_path = os.path.join(
            self._serialization_dir, task_directory, "_".join(["metric", suffix])
        )

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))

        for name, param in self._model.named_parameters():
            if param.requires_grad and name not in model_state:
                log.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                log.error("Parameter missing from checkpoint: " + name)
                log.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self._model.load_state_dict(model_state, strict=False)
        task_states = torch.load(task_state_path)
        for task_name, task_state in task_states.items():
            if task_name == "global":
                continue
            self._task_infos[task_name]["total_batches_trained"] = task_state[
                "total_batches_trained"
            ]
            self._task_infos[task_name]["stopped"] = task_state["stopped"]
            generator = self._task_infos[task_name]["tr_generator"]
            for _ in itertools.islice(
                generator,
                task_state["total_batches_trained"] % self._task_infos[task_name]["n_tr_batches"],
            ):
                pass
        self._optimizer.load_state_dict(task_states["global"]["optimizer"])
        # NOTE(Alex): AllenNLP wrapper doesn't expose scheduler state dict methods
        self._scheduler.lr_scheduler.load_state_dict(task_states["global"]["scheduler"])
        metric_states = torch.load(metric_state_path)
        for metric_name, metric_state in metric_states.items():
            self._metric_infos[metric_name]["hist"] = metric_state["hist"]
            self._metric_infos[metric_name]["stopped"] = metric_state["stopped"]
            self._metric_infos[metric_name]["best"] = metric_state["best"]

        training_state = torch.load(training_state_path)
        return training_state["step"], training_state["should_stop"]

    def _metrics_to_tensorboard_tr(self, val_pass, train_metrics, task_name):
        """
        Sends all of the train metrics to tensorboard
        """
        metric_names = train_metrics.keys()

        for name in metric_names:
            if name == "micro_avg" or name == "macro_avg":
                continue
            train_metric = train_metrics.get(name)
            name = os.path.join(task_name, task_name + "_" + name)
            self._TB_train_log.add_scalar(name, train_metric, val_pass)

    def _metrics_to_tensorboard_val(self, val_pass, val_metrics):
        """
        Sends all of the val metrics to tensorboard
        """
        metric_names = val_metrics.keys()

        for name in metric_names:
            if name == "micro_avg" or name == "macro_avg":
                continue
            val_metric = val_metrics.get(name)
            name = os.path.join(name.split("_")[0], name)
            self._TB_validation_log.add_scalar(name, val_metric, val_pass)

    @classmethod
    def from_params(cls, model, serialization_dir, params):
        """ Generate trainer from parameters.  """

        patience = params.pop("patience", 2)
        val_interval = params.pop("val_interval", 100)
        max_vals = params.pop("max_vals", 50)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        keep_all_checkpoints = params.pop("keep_all_checkpoints", False)
        val_data_limit = params.pop("val_data_limit", 5000)
        max_epochs = params.pop("max_epochs", -1)
        min_epochs = params.pop("min_epochs", 0)
        dec_val_scale = params.pop("dec_val_scale", 100)
        training_data_fraction = params.pop("training_data_fraction", 1.0)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            patience=patience,
            val_interval=val_interval,
            max_vals=max_vals,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            lr_decay=lr_decay,
            min_lr=min_lr,
            keep_all_checkpoints=keep_all_checkpoints,
            val_data_limit=val_data_limit,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            dec_val_scale=dec_val_scale,
            training_data_fraction=training_data_fraction,
        )


class SamplingMultiTaskTrainerBayes(SamplingMultiTaskTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_num_examples(
        self,
        batch_size,
        data,
    ):
        """
        Builds validation generator, evaluates on each task and produces validation metrics.

        Parameters
        ----------
        task: current task to get validation performance of
        batch_size: int, batch size to use for the tasks
        data: something like task.val_data

        Returns
        -------
        all_val_metrics: dict with metrics, names with name_suffix
        """
        n_examples, batch_num = 0, 0
        max_data_points = data.size
        val_generator = BasicIterator(batch_size, instances_per_epoch=max_data_points)(
            data, num_epochs=1, shuffle=False
        )
        val_generator = move_to_device(val_generator, self._cuda_device)

        all_num_examples = 0
        for batch in val_generator:
            num_examples = torch.sum((batch['span1s'][..., 0] != -1).long()).data.cpu().numpy()
            all_num_examples += num_examples

        return all_num_examples

    def train(
        self,
        tasks,
        stop_metric,
        batch_size,
        weighting_method,
        scaling_method,
        train_params,
        optimizer_params,
        scheduler_params,
        load_model=1,
        phase="pretrain",
    ):
        """
        The main training loop.
        Training will stop if we run out of patience or hit the minimum learning rate.

        Parameters
        ----------
        tasks: a list of task objects to train on
        stop_metric: str, metric to use for early stopping
        batch_size: int, batch size to use for the tasks
        weighting_method: str, how to sample which task to use
        scaling_method:  str, how to scale gradients
        train_params: trainer config object
        optimizer_params: optimizer config object
        scheduler_params: scheduler config object
        load_model: bool, whether to restore and continue training if a checkpoint is found
        phase: str, usually 'pretrain' or 'target_train'

        Returns
        ----------
        Validation results
        """
        validation_interval = self._val_interval
        task_infos, metric_infos = self._setup_training(
            tasks, batch_size, train_params, optimizer_params, scheduler_params, phase
        )

        assert len(tasks) == 1, "If you are using several tasks, read the code and make sure it is correct.\nBut do not do this, just use one task!"

        num_examples = self._calculate_num_examples(batch_size, tasks[0].train_data)

        optimizer_params = copy.deepcopy(optimizer_params)
        if "t_total" in optimizer_params:
            # If we know in advance how many opt steps there will be, set it so the LR scheduler
            # can use that information. This should be the next validation after we hit the epoch
            # limit.
            if self._max_epochs > 0:
                n_epoch_steps = sum([info["n_tr_batches"] for info in task_infos.values()])
                max_epochs_in_vals = math.ceil(
                    (n_epoch_steps * self._max_epochs) / self._val_interval
                )
                val_limit = min(max_epochs_in_vals, self._max_vals)
            else:
                val_limit = self._max_vals
            optimizer_params["t_total"] = val_limit * self._val_interval

        optimizer = Optimizer.from_params(train_params, optimizer_params)
        scheduler = LearningRateScheduler.from_params(optimizer, copy.deepcopy(scheduler_params))
        self._optimizer = optimizer
        self._scheduler = scheduler

        # define these here b/c they might get overridden on load

        n_step, should_stop = 0, False
        if self._serialization_dir is not None:
            # Resume from serialization path
            if load_model:
                ckpt_directory, _, _ = check_for_previous_checkpoints(
                    self._serialization_dir, tasks, phase, load_model
                )
                if ckpt_directory is None:
                    log.warning(
                        "load_model=1 but there is not checkpoint. \
                        Starting training without restoring from a checkpoint."
                    )
                else:
                    n_step, should_stop = self._restore_checkpoint(phase, tasks)
                    log.info("Loaded model from checkpoint. Starting at step %d.", n_step)
            else:
                log.info("Starting training without restoring from a checkpoint.")
                check_for_previous_checkpoints(self._serialization_dir, tasks, phase, load_model)
        if self._grad_clipping is not None:  # pylint: disable=invalid-unary-operand-type

            def clip_function(grad):
                return grad.clamp(-self._grad_clipping, self._grad_clipping)

            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        # Calculate per task sampling weights
        assert_for_log(len(tasks) > 0, "Error: Expected to sample from 0 tasks.")

        task_names = [task.name for task in tasks]
        task_n_train_examples = np.array([task.n_train_examples for task in tasks])
        task_n_train_batches = np.array([task_infos[task.name]["n_tr_batches"] for task in tasks])
        log.info(
            "Training examples per task, before any subsampling: "
            + str(dict(zip(task_names, task_n_train_examples)))
        )
        if len(tasks) > 1:
            sample_weights = self.get_sampling_weights(
                weighting_method, len(tasks), task_n_train_examples, task_n_train_batches
            )

            normalized_sample_weights = np.array(sample_weights) / sum(sample_weights)
            log.info(
                "Using weighting method: %s, with normalized sample weights %s ",
                weighting_method,
                np.array_str(normalized_sample_weights, precision=4),
            )
            scaling_weights = self.get_scaling_weights(
                scaling_method, len(tasks), task_names, task_n_train_examples
            )
        else:
            sample_weights = normalized_sample_weights = [1.0]
            scaling_weights = {task_names[0]: 1.0}

        # Sample the tasks to train on. Do it all at once (val_interval) for
        # MAX EFFICIENCY.
        samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

        offset = 0
        all_tr_metrics = {}
        log.info("Beginning training with stopping criteria based on metric: %s", stop_metric)
        while not should_stop:
            self._model.train()
            task = samples[(n_step + offset) % validation_interval]  # randomly select a task
            task_info = task_infos[task.name]
            task_info["xent"] = 0
            if task_info["stopped"]:
                offset += 1
                continue
            tr_generator = task_info["tr_generator"]
            total_batches_trained = task_info["total_batches_trained"]
            n_batches_since_val = task_info["n_batches_since_val"]
            tr_loss = task_info["loss"]
            xent_loss = task_info["xent"]

            bayes_modules = list(BayesianLayers.get_kl_modules(self._model))
            for batch in itertools.islice(tr_generator, 1):
                n_batches_since_val += 1
                total_batches_trained += 1
                optimizer.zero_grad()
                output_dict = self._forward(batch, task=task)
                assert_for_log(
                    "loss" in output_dict, "Model must return a dict containing a 'loss' key"
                )
                #------------------------------------------------------------------------
                kl = sum([mod.kl_divergence() for mod in bayes_modules])
                loss = output_dict["loss"] + kl / num_examples
                # ------------------------------------------------------------------------
                loss *= scaling_weights[task.name]
                loss.backward()
                nan_in_batch = False
                
                if torch.isnan(loss).any() or any(torch.isnan(param.grad).any() if param.grad is not None else False for name, param in train_params):
                    nan_in_batch = True
                    print("NaN in loss or grads; dropping batch")
                    print("NaN in loss:", torch.isnan(loss).any())
                    print("NaN grad in grads:", any(torch.isnan(param.grad).any() if param.grad is not None else False for name, param in train_params))
                    print(f"loss={loss}, kl={kl}, output_dict[loss]=f{output_dict['loss']}, output_dict={output_dict}")
                    n_batches_since_val -= 1
                    total_batches_trained -= 1
                    print("Saving dump to nan_dump.pth ...")
                    try:
                        torch.save([batch, self._model], "nan_dump.npz")
                    except: print('failed to save dump')
                    optimizer.zero_grad()
                    break
                
                #assert_for_log(not torch.isnan(loss).any(), "NaNs in loss.")
                xent_loss += loss.data.cpu().numpy()
                tr_loss += loss.data.cpu().numpy()

                # Gradient regularization and application
                if self._grad_norm:
                    clip_grad_norm_(self._model.parameters(), self._grad_norm)
                optimizer.step()
                n_step += 1  # update per batch

                # step scheduler if it's not ReduceLROnPlateau
                if not isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                    scheduler.step_batch(n_step)
            
            if nan_in_batch:
                print('skipped a batch.')  # do not record training metrics with nan in batch
                continue
            
            # Update training progress on that task
            task_info["n_batches_since_val"] = n_batches_since_val
            task_info["total_batches_trained"] = total_batches_trained
            task_info["loss"] = tr_loss
            task_info["xent"] = xent_loss
            task_info["kl"] = sum([mod.kl_divergence() for mod in bayes_modules])

            # Intermediate log to logger and tensorboard
            if time.time() - task_info["last_log"] > self._log_interval:
                task_metrics = task.get_metrics()

                # log to tensorboard
                if self._TB_dir is not None:
                    task_metrics_to_TB = task_metrics.copy()

                    masks = BayesianLayers.get_masks(bayes_modules, [0] * len(bayes_modules))
                    for layer_ind in range(len(masks)):
                        task_metrics_to_TB["neurons_l{}".format(layer_ind)] = sum(masks[layer_ind])

                    task_metrics_to_TB["loss"] = float(task_info["loss"] / n_batches_since_val)
                    task_metrics_to_TB["xent"] = float(task_info["xent"] / n_batches_since_val)
                    self._metrics_to_tensorboard_tr(n_step, task_metrics_to_TB, task.name)

                task_metrics["%s_loss" % task.name] = tr_loss / n_batches_since_val
                description = self._description_from_metrics(task_metrics)
                log.info(
                    "Update %d: task %s, batch %d (%d): %s",
                    n_step,
                    task.name,
                    n_batches_since_val,
                    total_batches_trained,
                    description,
                )
                task_info["last_log"] = time.time()

                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric()
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

            # Validation
            if n_step % validation_interval == 0:

                # Dump and log all of our current info
                n_val = int(n_step / validation_interval)
                log.info("***** Step %d / Validation %d *****", n_step, n_val)
                # Get metrics for all training progress so far
                for task in tasks:
                    task_info = task_infos[task.name]
                    n_batches_since_val = task_info["n_batches_since_val"]
                    if n_batches_since_val > 0:
                        task_metrics = task.get_metrics(reset=True)
                        for name, value in task_metrics.items():
                            all_tr_metrics["%s_%s" % (task.name, name)] = value
                        # Updating loss from training
                        all_tr_metrics["%s_loss" % task.name] = float(
                            task_info["loss"] / n_batches_since_val
                        )
                    else:
                        all_tr_metrics["%s_loss" % task.name] = 0.0
                    log.info(
                        "%s: trained on %d batches, %.3f epochs",
                        task.name,
                        n_batches_since_val,
                        n_batches_since_val / task_info["n_tr_batches"],
                    )
                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric(reset=True)
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

                # Validate
                log.info("Validating...")
                all_val_metrics, should_save, new_best = self._validate(n_val, tasks, batch_size)

                # Check stopping conditions
                should_stop = self._check_stop(n_val, stop_metric, tasks)

                # Log results to logger and tensorboard
                for name, value in all_val_metrics.items():
                    log_str = "%s:" % name
                    if name in all_tr_metrics:
                        log_str += " training: %3f" % all_tr_metrics[name]
                    log_str += " validation: %3f" % value
                    log.info(log_str)
                if self._TB_dir is not None:
                    self._metrics_to_tensorboard_val(n_step, all_val_metrics)
                log.info(f"Global learning rate: {self._optimizer.param_groups[0]['lr']}")
                elmo_params = self._model.get_elmo_mixing_weights(tasks)
                if elmo_params:  # log ELMo mixing weights
                    for task_name, task_params in elmo_params.items():
                        log.info("ELMo mixing weights for {}:".format(task_name))
                        log.info(
                            "\t"
                            + ", ".join(
                                [
                                    "{}: {:.6f}".format(layer, float(param))
                                    for layer, param in task_params.items()
                                ]
                            )
                        )

                # Reset training preogress
                all_tr_metrics = {}
                samples = random.choices(
                    tasks, weights=sample_weights, k=validation_interval
                )  # pylint: disable=no-member

                if should_save:
                    self._save_checkpoint(
                        {"step": n_step, "validation_pass": n_val, "should_stop": should_stop},
                        tasks=tasks,
                        phase=phase,
                        new_best=new_best,
                    )

        log.info("Stopped training after %d validation checks", n_step / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)  # , validation_interval)




class SamplingMultiTaskTrainerOnlineCoding(SamplingMultiTaskTrainer):
    def __init__(self, *args, online_coding_load_checkpoint, online_coding_portion_split,
                 online_coding_pre_shuffle, online_coding_pre_shuffle_seed, **kwargs):
        super().__init__(*args, **kwargs)


        # online coding
        self.online_coding_load_checkpoint = online_coding_load_checkpoint
        self.online_coding_portion_split = online_coding_portion_split

        # shuffle data
        self.online_coding_pre_shuffle = online_coding_pre_shuffle
        self.online_coding_pre_shuffle_seed = online_coding_pre_shuffle_seed


    @classmethod
    def from_params(cls, model, serialization_dir, params):
        """ Generate trainer from parameters.  """

        patience = params.pop("patience", 2)
        val_interval = params.pop("val_interval", 100)
        max_vals = params.pop("max_vals", 50)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        keep_all_checkpoints = params.pop("keep_all_checkpoints", False)
        val_data_limit = params.pop("val_data_limit", 5000)
        max_epochs = params.pop("max_epochs", -1)
        min_epochs = params.pop("min_epochs", 0)
        dec_val_scale = params.pop("dec_val_scale", 100)
        training_data_fraction = params.pop("training_data_fraction", 1.0)
        online_coding_load_checkpoint = params.pop("online_coding_load_checkpoint")
        online_coding_portion_split = params.pop("online_coding_portion_split")
        online_coding_pre_shuffle = params.pop("online_coding_pre_shuffle", False)
        assert isinstance(online_coding_pre_shuffle, bool)
        online_coding_pre_shuffle_seed = int(params.pop("online_coding_pre_shuffle_seed", 1337))

        params.assert_empty(cls.__name__)
        return cls(
            model,
            patience=patience,
            val_interval=val_interval,
            max_vals=max_vals,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            lr_decay=lr_decay,
            min_lr=min_lr,
            keep_all_checkpoints=keep_all_checkpoints,
            val_data_limit=val_data_limit,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            dec_val_scale=dec_val_scale,
            training_data_fraction=training_data_fraction,
            online_coding_load_checkpoint=online_coding_load_checkpoint,
            online_coding_portion_split=online_coding_portion_split,
            online_coding_pre_shuffle=online_coding_pre_shuffle,
            online_coding_pre_shuffle_seed=online_coding_pre_shuffle_seed,
        )

    def _calculate_online_stat(
        self,
        task,
        batch_size,
        eval_data,
        name_suffix='',
    ):
        """
        Builds validation generator, evaluates on each task and produces validation metrics.

        Parameters
        ----------
        task: current task to get validation performance of
        batch_size: int, batch size to use for the tasks
        eval_data: something like task.val_data

        Returns
        -------
        all_val_metrics: dict with metrics, names with name_suffix
        """
        n_examples, batch_num = 0, 0
        max_data_points = eval_data.size
        val_generator = BasicIterator(batch_size, instances_per_epoch=max_data_points)(
            eval_data, num_epochs=1, shuffle=False
        )
        val_generator = move_to_device(val_generator, self._cuda_device)
        n_val_batches = math.ceil(eval_data.size / batch_size)

        all_val_metrics = {}
        all_val_metrics["{}_sumloss_{}".format(task.name, name_suffix)] = 0.0
        all_val_metrics["{}_num_examples_{}".format(task.name, name_suffix)] = 0.0
        all_val_metrics["{}_acc_{}".format(task.name, name_suffix)] = 0.0

        for batch in val_generator:
            num_examples = torch.sum((batch['span1s'][..., 0] != -1).long()).data.cpu().numpy()
            batch_num += 1
            with torch.no_grad():
                out = self._forward(batch, task=task)
            loss = out["loss"]
            all_val_metrics["{}_sumloss_{}".format(task.name, name_suffix)] += num_examples * loss.data.cpu().numpy()
            all_val_metrics["{}_num_examples_{}".format(task.name, name_suffix)] += num_examples
            all_val_metrics["{}_acc_{}".format(task.name, name_suffix)] += num_examples * out["acc"].data.cpu().numpy()
        all_val_metrics["{}_acc_{}".format(task.name, name_suffix)] /= all_val_metrics["{}_num_examples_{}".format(task.name, name_suffix)]

        return all_val_metrics


    def train(
        self,
        tasks,
        stop_metric,
        batch_size,
        weighting_method,
        scaling_method,
        train_params,
        optimizer_params,
        scheduler_params,
        load_model=1,
        phase="pretrain",
    ):
        """
        The main training loop.
        Training will stop if we run out of patience or hit the minimum learning rate.

        Parameters
        ----------
        tasks: a list of task objects to train on
        stop_metric: str, metric to use for early stopping
        batch_size: int, batch size to use for the tasks
        weighting_method: str, how to sample which task to use
        scaling_method:  str, how to scale gradients
        train_params: trainer config object
        optimizer_params: optimizer config object
        scheduler_params: scheduler config object
        load_model: bool, whether to restore and continue training if a checkpoint is found
        phase: str, usually 'pretrain' or 'target_train'

        Returns
        ----------
        Validation results
        """

        assert len(tasks) == 1, "Online coding regimen expects only one task. Be careful!"
        task = tasks[0]
        train_data = task.train_data
        if self.online_coding_pre_shuffle:
            log.info(f"Shuffling data with seed {self.online_coding_pre_shuffle_seed}...")
            train_data = train_data.shuffled(self.online_coding_pre_shuffle_seed)
            log.info("Done!")
        else:
            log.info("Not shuffling training data!")
        val_data = task.val_data

        portion_sizes = list(map(float, self.online_coding_portion_split.split(',')))
        portion_inds = [int(portion_size * task.train_data.size) for portion_size in portion_sizes]

        clean_serialization_dir = self._serialization_dir
        results_by_portion = []
        for i in range(len(portion_sizes)):
            train_portion = train_data.get_data_portion(begin=0, end=portion_inds[i])
            if i != len(portion_sizes) - 1:
                dev_portion_online = train_data.get_data_portion(begin=portion_inds[i], end=portion_inds[i + 1])
            else:
                dev_portion_online = val_data
            task.train_data = train_portion
            task.example_counts["train"] = train_portion.size

            self._serialization_dir = os.path.join(clean_serialization_dir, "online_portion_{}".format(i))
            self._TB_dir = os.path.join(self._serialization_dir, "tensorboard")
            self._TB_train_log = SummaryWriter(os.path.join(self._TB_dir, "train"))
            self._TB_validation_log = SummaryWriter(os.path.join(self._TB_dir, "val"))
            _ = super().train(tasks=[task],  # diff
                          stop_metric=stop_metric,
                          batch_size=batch_size,
                          weighting_method=weighting_method,
                          scaling_method=scaling_method,
                          train_params=train_params,
                          optimizer_params=optimizer_params,
                          scheduler_params=scheduler_params,
                          load_model=self.online_coding_load_checkpoint, # diff
                          phase=phase,)

            def find_best_ckpt(dir_):
                names = [el for el in os.listdir(dir_) if
                         el.startswith('model_state_pretrain_val') and el.endswith('.best.th')]
                inds = list(sorted([int(name.split('.')[0].split('_')[-1]) for name in names]))
                return inds[-1]

            ind = find_best_ckpt(self._serialization_dir)
            self._model.load_state_dict(torch.load(os.path.join(self._serialization_dir,
                                                      'model_state_pretrain_val_{}.best.th'.format(ind))),
                                        strict=False)

            print("Online stat: val")
            portion_result = self._calculate_online_stat(task, batch_size,
                                                              eval_data=val_data,
                                                              name_suffix='val')
            print("Online stat: test")
            portion_result.update(self._calculate_online_stat(task, batch_size,
                                                      eval_data=task.test_data,
                                                      name_suffix='test'))
            print("Online stat: dev_portion_online")
            portion_result.update(self._calculate_online_stat(task, batch_size,
                                                      eval_data=dev_portion_online,
                                                      name_suffix='dev_portion_online'))
            results_by_portion.append(portion_result)
            print("Portion result: ", portion_result)

            with open(os.path.join(self._serialization_dir, "online_coding.pkl"), 'wb') as f:
                pickle.dump(results_by_portion, f)



