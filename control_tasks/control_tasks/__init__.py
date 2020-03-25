from argparse import ArgumentParser
import os
import random
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import random
import numpy as np

from . import data, model, probe, regimen, reporter, task, loss

def choose_task_classes(args):
  """Chooses which task class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a task specification.
  """
  if args['probe']['task_name'] == 'parse-distance':
    task_class = task.ParseDistanceTask
    reporter_class = reporter.WordPairReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DistanceLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'parse-depth':
    task_class = task.ParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'part-of-speech':
    task_class = task.PartOfSpeechLabelTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'cross-entropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'rand-prefix-label':
    task_class = task.RandomPrefixLabelTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'cross-entropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'rand-word-label':
    task_class = task.RandomWordLabelTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'cross-entropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'corrupted-part-of-speech':
    task_class = task.CorruptedPartOfSpeechLabelTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'cross-entropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'corrupted-edge-labels':
    task_class = task.CorruptedEdgePositionTask
    reporter_class = reporter.WordPairLabelReporter
    if args['probe_training']['loss'] == 'cross-entropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
      #elif args['probe']['task_name'] == 'corrupted-parse-distance':
      #  task_class = task.CorruptedParseDistanceTask
      #  reporter_class = reporter.WordPairReporter
      #  if args['probe_training']['loss'] == 'L1':
      #    loss_class = loss.L1DistanceLoss
      #  else:
      #    raise ValueError("Unknown loss type for given probe type: {}".format(
      #      args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'rand-parse-depth':
    task_class = task.RandomParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'rand-linear-parse-depth':
    task_class = task.RandomLinearParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'balanced-distance':
    task_class = task.BalancedBinaryTreeDistanceTask
    reporter_class = reporter.WordPairReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DistanceLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'corrupted-parse-depth':
    task_class = task.CorruptedParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))

  else:
    raise ValueError("Unknown probing task type: {}".format(
      args['probe']['task_name']))
  return task_class, reporter_class, loss_class

def choose_dataset_class(args):
  """Chooses which dataset class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a dataset.
  """
  if args['model']['model_type'] in {'ELMo-disk', 'ELMo-random-projection', 'ELMo-decay'}:
    dataset_class = data.ELMoDataset
  elif args['model']['model_type'] == 'BERT-disk':
    dataset_class = data.BERTDataset
  else:
    raise ValueError("Unknown model type for datasets: {}".format(
      args['model']['model_type']))

  return dataset_class

def choose_probe_class(args):
  """Chooses which probe and reporter classes to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A probe_class to be instantiated.
  """
  if args['probe']['task_signature'] == 'word':
    if args['probe']['psd_parameters']:
      return probe.OneWordPSDProbe
    elif 'probe_spec' not in args['probe'] or args['probe']['probe_spec']['probe_hidden_layers'] == 0:
      return probe.OneWordNonPSDProbe
    else:
      return probe.OneWordNNDepthProbe
  elif args['probe']['task_signature'] == 'word_pair':
    if args['probe']['psd_parameters']:
      return probe.TwoWordPSDProbe
    else:
      return probe.TwoWordNonPSDProbe
  elif args['probe']['task_signature'] == 'word_label':
    if 'probe_spec' not in args['probe'] or args['probe']['probe_spec']['probe_hidden_layers'] == 0:
      return probe.OneWordLinearLabelProbe
    else:
      if 'type' not in args['probe'] or args['probe']['type'] == 'none':
        return probe.OneWordNNLabelProbe
      elif args['probe'] or args['probe']['type'] == 'bayes':
        return probe.OneWordNNLabelProbeBayesCompression
      else:
        raise ValueError("Unknown Probe type: not 'none' and not 'bayesian'")
  elif args['probe']['task_signature'] == 'word_pair_label':
    if args['probe']['probe_spec']['probe_type'] == 'linear':
      return probe.TwoWordLinearLabelProbe
    if args['probe']['probe_spec']['probe_type'] == 'bilinear':
      return probe.TwoWordBilinearLabelProbe
    if args['probe']['probe_spec']['probe_type'] == 'MLP':
      return probe.TwoWordNNLabelProbe
    else:
      raise ValueError("Unknown TwoWordLabel type")
  else:
    raise ValueError("Unknown probe type (probe function signature): {}".format(
      args['probe']['task_signature']))

def choose_model_class(args):
  """Chooses which reporesentation learner class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a model to supply word representations.
  """
  if args['model']['model_type'] == 'ELMo-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'BERT-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'ELMo-random-projection':
    return model.ProjectionModel
  elif args['model']['model_type'] == 'ELMo-decay':
    return model.DecayModel
  elif args['model']['model_type'] == 'pytorch_model':
    raise ValueError("Using pytorch models for embeddings not yet supported...")
  else:
    raise ValueError("Unknown model type: {}".format(
      args['model']['model_type']))

def run_train_probe(args, probe, dataset, model, loss, reporter, regimen):
  """Trains a structural probe according to args.

  Args:
    args: the global config dictionary built by yaml.
          Describes experiment settings.
    probe: An instance of probe.Probe or subclass.
          Maps hidden states to linguistic quantities.
    dataset: An instance of data.SimpleDataset or subclass.
          Provides access to DataLoaders of corpora. 
    model: An instance of model.Model
          Provides word representations.
    reporter: An instance of reporter.Reporter
          Implements evaluation and visualization scripts.
  Returns:
    None; causes probe parameters to be written to disk.
  """
  regimen.train_until_convergence(probe, model, loss,
      dataset.get_train_dataloader(), dataset.get_dev_dataloader())


def run_report_results(args, probe, dataset, model, loss, reporter, regimen):
  """
  Reports results from a structural probe according to args.
  By default, does so only for dev set.
  Requires a simple code change to run on the test set.
  """
  probe_params_path = os.path.join(args['reporting']['root'],args['probe']['params_path'])
  probe.load_state_dict(torch.load(probe_params_path))
  probe.eval()

  dev_dataloader = dataset.get_dev_dataloader()
  dev_predictions = regimen.predict(probe, model, dev_dataloader)
  reporter(dev_predictions, dev_dataloader, 'dev')
