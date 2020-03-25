"""Classes for training and running inference on probes."""
import os
import sys

from torch import optim
import torch
from tqdm import tqdm
import json

class ProbeRegimen:
  """Basic regimen for training and running inference on probes.
  
  Tutorial help from:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

  Attributes:
    optimizer: the optimizer used to train the probe
    scheduler: the scheduler used to set the optimizer base learning rate
  """

  def __init__(self, args, reporter):
    self.args = args
    
    self.reporter = reporter
    self.reports = []
    
    self.max_epochs = args['probe_training']['epochs']
    self.params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
    self.max_gradient_steps = args['probe_training']['max_gradient_steps'] if 'max_gradient_steps' in args['probe_training'] else sys.maxsize
    self.dev_eval_gradient_steps = args['probe_training']['eval_dev_every'] if 'eval_dev_every' in args['probe_training'] else -1

  def set_optimizer(self, probe):
    """Sets the optimizer and scheduler for the training regimen.
  
    Args:
      probe: the probe PyTorch model the optimizer should act on.
    """
    if 'weight_decay' in self.args['probe_training']:
      weight_decay = self.args['probe_training']['weight_decay']
    else:
      weight_decay = 0
    if 'scheduler_patience' in self.args['probe_training']:
      scheduler_patience = self.args['probe_training']['scheduler_patience']
    else:
      scheduler_patience = 0
    
    learning_rate = 0.001 if not 'learning_rate' in self.args['probe_training'] else\
                    self.args['probe_training']['learning_rate']
        
    scheduler_factor = 0.5 if not 'scheduler_factor' in self.args['probe_training'] else\
                    self.args['probe_training']['scheduler_factor']

    self.optimizer = optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                          mode='min',
                                                          factor=scheduler_factor,
                                                          patience=scheduler_patience)
    

  def seqlens_on_dataset(self, dataset):
    seqlens = []
    for batch in tqdm(dataset, desc='[training batch]'):
      observation_batch, label_batch, length_batch, _ = batch
      seqlens += list(length_batch.detach().cpu().numpy())
    return seqlens

  def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset, eval_datasets={}, test_dataset=None):
    """ Trains a probe until a convergence criterion is met.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    def loss_on_dataset(dataset, name=''):
      loss_ = 0
      num_targets = 0
      num_targets_real = 0
      num_examples = 0
      for batch in tqdm(dataset, desc='[eval batch{}]'.format(' ' + name)):
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        loss_ += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        num_targets += sum(length_batch.detach().cpu().numpy())
        num_targets_real += torch.sum(label_batch > -1).detach().cpu().numpy()
        num_examples += observation_batch.detach().cpu().numpy().shape[0]
      return {'loss{}'.format('_' + name): loss_,
              'num_targets{}'.format('_' + name): num_targets,
              'num_targets_real{}'.format('_' + name): num_targets_real,
              'num_examples{}'.format('_' + name): num_examples,}

    def num_targets(dataset):
        num_targets = 0
        for batch in tqdm(dataset):
            observation_batch, label_batch, length_batch, _ = batch
            num_targets += torch.sum(label_batch > -1).detach().cpu().numpy()
        return num_targets

    def eval_on_exit():
        probe.load_state_dict(torch.load(self.params_path))
        probe.eval()
        print("Evaling on exit...")
        result = {'train_targets': num_targets(train_dataset)}
        for name, dataset in eval_datasets.items():
            result.update(loss_on_dataset(dataset, name=name))
        return result
    
    self.set_optimizer(probe)
    min_dev_loss = sys.maxsize
    min_epoch_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    gradient_steps = 0
    eval_dev_every = self.dev_eval_gradient_steps if self.dev_eval_gradient_steps != -1 else (len(train_dataset))
    eval_index = 0
    min_dev_loss_eval_index = -1
    
    self.wait_without_improvement_for = self.args['probe_training'].get('wait_without_improvement_for', 4)
    
    if self.args['reporting'].get('report_ckpt', False):
      if not os.path.exists(os.path.join(self.args['reporting']['root'], 'checkpoint')):
        os.mkdir(os.path.join(self.args['reporting']['root'], 'checkpoint'))  
        
    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      epoch_train_loss = 0
      epoch_train_epoch_count = 0
      epoch_dev_epoch_count = 0
      epoch_train_loss_count = 0
      for batch in tqdm(train_dataset, desc='[training batch]'):
        probe.train()
        self.optimizer.zero_grad()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        batch_loss.backward()
        epoch_train_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_train_epoch_count += 1
        epoch_train_loss_count += count.detach().cpu().numpy()
        self.optimizer.step()
        gradient_steps += 1
        if gradient_steps % eval_dev_every == 0:
          eval_index += 1
          if gradient_steps >= self.max_gradient_steps:
            tqdm.write('Hit max gradient steps; stopping')
            with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
                json.dump(self.reports, f)
            return self.reports, eval_on_exit()
          epoch_dev_loss = 0
          epoch_dev_loss_count = 0
          for batch in tqdm(dev_dataset, desc='[dev batch]'):
            self.optimizer.zero_grad()
            probe.eval()
            observation_batch, label_batch, length_batch, _ = batch
            word_representations = model(observation_batch)
            predictions = probe(word_representations)
            batch_loss, count = loss(predictions, label_batch, length_batch)
            epoch_dev_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
            epoch_dev_loss_count += count.detach().cpu().numpy()
            epoch_dev_epoch_count += 1
          self.scheduler.step(epoch_dev_loss)
          tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index,
              epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))
          
          current_report = {'eval_step': eval_index,
                            'gradient_steps': gradient_steps,
                            'dev_loss': epoch_dev_loss/epoch_dev_loss_count,
                            'train_loss': epoch_train_loss/epoch_train_loss_count,
                            'min_dev_loss': min_dev_loss,
                           'min_epoch_dev_loss': min_epoch_dev_loss}
          dev_predictions = self.predict(probe, model, dev_dataset)
          report_from_reporter = self.reporter(dev_predictions, dev_dataset, 'dev', probe=probe, is_train=True)
          current_report.update(report_from_reporter)
          self.reports.append(current_report)
          if self.args['reporting'].get('report_ckpt', False):
             torch.save(probe.state_dict(),
                        self.args['reporting']['root'] + '/checkpoint/probe_state_dict_{}'.format(epoch_index))
             torch.save(probe,
                        self.args['reporting']['root'] + '/checkpoint/probe_{}.pth'.format(epoch_index))
          with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
            json.dump(self.reports, f)
            
          if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.001:
            torch.save(probe.state_dict(), self.params_path)
            torch.save(probe, self.params_path + '_whole_probe.pth')
            min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
            min_epoch_dev_loss = epoch_dev_loss
            min_dev_loss_epoch = epoch_index
            min_dev_loss_eval_index = eval_index
            tqdm.write('Saving probe parameters')
          elif min_dev_loss_eval_index < eval_index - self.wait_without_improvement_for:
            tqdm.write('Early stopping')
            with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
                json.dump(self.reports, f)
            return self.reports, eval_on_exit()
    with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
      json.dump(self.reports, f)
    return self.reports, eval_on_exit()

  def predict(self, probe, model, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      dataset: A pytorch.DataLoader object 

    Returns:
      A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    for batch in tqdm(dataset, desc='[predicting]'):
      observation_batch, label_batch, length_batch, _ = batch
      word_representations = model(observation_batch)
      predictions = probe(word_representations)
      predictions_by_batch.append(predictions.detach().cpu().numpy())
    return predictions_by_batch




class BayesRegimen:
#TODO: write description  
  """
  TODO: write description
  Basic regimen for training and running inference on probes.
  
  Tutorial help from:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

  Attributes:
    optimizer: the optimizer used to train the probe
    scheduler: the scheduler used to set the optimizer base learning rate
  """

  def __init__(self, args, reporter):
    self.args = args
    
    self.reporter = reporter
    self.reports = []
    
    self.max_epochs = args['probe_training']['epochs']
    self.params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
    self.max_gradient_steps = args['probe_training']['max_gradient_steps'] if 'max_gradient_steps' in args['probe_training'] else sys.maxsize
    self.dev_eval_gradient_steps = args['probe_training']['eval_dev_every'] if 'eval_dev_every' in args['probe_training'] else -1
    
  def seqlens_on_dataset(self, dataset):
    seqlens = []
    num_targets = 0
    for batch in tqdm(dataset, desc='[training batch]'):
      observation_batch, label_batch, length_batch, _ = batch
      seqlens += list(length_batch.detach().cpu().numpy())
      num_targets += torch.sum(label_batch > -1).detach().cpu().numpy()
    return seqlens, num_targets

  def set_optimizer(self, probe):
    """Sets the optimizer and scheduler for the training regimen.
  
    Args:
      probe: the probe PyTorch model the optimizer should act on.
    """
    if 'weight_decay' in self.args['probe_training']:
      weight_decay = self.args['probe_training']['weight_decay']
    else:
      weight_decay = 0
    if 'scheduler_patience' in self.args['probe_training']:
      scheduler_patience = self.args['probe_training']['scheduler_patience']
    else:
      scheduler_patience = 0
    
    learning_rate = 0.001 if not 'learning_rate' in self.args['probe_training'] else\
                    self.args['probe_training']['learning_rate']
        
    scheduler_factor = 0.5 if not 'scheduler_factor' in self.args['probe_training'] else\
                    self.args['probe_training']['scheduler_factor']

    self.optimizer = optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                          mode='min',
                                                          factor=scheduler_factor,
                                                          patience=scheduler_patience)

  def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset, test_dataset=None):
    """ Trains a probe until a convergence criterion is met.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    self.set_optimizer(probe)
    min_dev_loss = sys.maxsize
    min_epoch_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    gradient_steps = 0
    eval_dev_every = self.dev_eval_gradient_steps if self.dev_eval_gradient_steps != -1 else (len(train_dataset))
    eval_index = 0
    min_dev_loss_eval_index = -1
    
    self.wait_without_improvement_for = self.args['probe_training'].get('wait_without_improvement_for', 4)
    
    _, num_train_targets = self.seqlens_on_dataset(train_dataset)
    
    if self.args['reporting'].get('report_ckpt', False):
      if not os.path.exists(os.path.join(self.args['reporting']['root'], 'checkpoint')):
        os.mkdir(os.path.join(self.args['reporting']['root'], 'checkpoint'))  
        
    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      epoch_train_loss = 0
      epoch_train_epoch_count = 0
      epoch_dev_epoch_count = 0
      epoch_train_loss_count = 0
      for batch in tqdm(train_dataset, desc='[training batch]'):
        probe.train()
        self.optimizer.zero_grad()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        #----------------------- Lena diff begin
        batch_loss = batch_loss * count + probe.kl_divergence() * torch.sum(label_batch > -1) / num_train_targets
        #----------------------- Lena diff end
        batch_loss.backward()
        epoch_train_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_train_epoch_count += 1
        epoch_train_loss_count += count.detach().cpu().numpy()
        self.optimizer.step()
        gradient_steps += 1
        if gradient_steps % eval_dev_every == 0:
          eval_index += 1
          if gradient_steps >= self.max_gradient_steps:
            tqdm.write('Hit max gradient steps; stopping')
            with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
                json.dump(self.reports, f)
            return self.reports
          epoch_dev_loss = 0
          epoch_dev_loss_count = 0
          for batch in tqdm(dev_dataset, desc='[dev batch]'):
            self.optimizer.zero_grad()
            probe.eval()
            observation_batch, label_batch, length_batch, _ = batch
            word_representations = model(observation_batch)
            predictions = probe(word_representations)
            batch_loss, count = loss(predictions, label_batch, length_batch)
            epoch_dev_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
            epoch_dev_loss_count += count.detach().cpu().numpy()
            epoch_dev_epoch_count += 1
          train_xent = 0
          for batch in tqdm(train_dataset, desc='[EVAL: train batch]'):
            self.optimizer.zero_grad()
            probe.eval()
            observation_batch, label_batch, length_batch, _ = batch
            word_representations = model(observation_batch)
            predictions = probe(word_representations)
            batch_loss, count = loss(predictions, label_batch, length_batch)
            train_xent += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
            
          self.scheduler.step(epoch_dev_loss)
          tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index,
              epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))
            
          current_report = {'epoch': epoch_index,
                            'gradient_steps': gradient_steps,
                            'dev_xent': epoch_dev_loss/epoch_dev_loss_count,
                            'train_xent': train_xent,
                            'kl': float(probe.kl_divergence().detach().cpu().numpy()),
                            }
          dev_predictions = self.predict(probe, model, dev_dataset)
          report_from_reporter = self.reporter(dev_predictions, dev_dataset, 'dev', probe=probe, is_train=True)
          if test_dataset is not None:
            test_predictions = self.predict(probe, model, test_dataset)
            report_from_reporter = self.reporter(test_predictions, test_dataset, 'test', probe=probe, is_train=True)
          current_report.update(report_from_reporter)
          self.reports.append(current_report)
          if self.args['reporting'].get('report_ckpt', False):
             torch.save(probe.state_dict(),
                        self.args['reporting']['root'] + '/checkpoint/probe_state_dict_{}'.format(epoch_index))
             torch.save(probe,
                        self.args['reporting']['root'] + '/checkpoint/probe_{}.pth'.format(epoch_index))
          with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
            json.dump(self.reports, f)
                
        
          if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.001:
            torch.save(probe.state_dict(), self.params_path)
            torch.save(probe, self.params_path + '_whole_probe.pth')
            min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
            min_epoch_dev_loss = epoch_dev_loss
            min_dev_loss_epoch = epoch_index
            min_dev_loss_eval_index = eval_index
            tqdm.write('Saving probe parameters')
          elif min_dev_loss_eval_index < eval_index - self.wait_without_improvement_for:
            tqdm.write('Early stopping')
            with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
                json.dump(self.reports, f)
            return self.reports
    with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
      json.dump(self.reports, f)
    return self.reports

  def predict(self, probe, model, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      dataset: A pytorch.DataLoader object 

    Returns:
      A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    for batch in tqdm(dataset, desc='[predicting]'):
      observation_batch, label_batch, length_batch, _ = batch
      word_representations = model(observation_batch)
      predictions = probe(word_representations)
      predictions_by_batch.append(predictions.detach().cpu().numpy())
    return predictions_by_batch



