"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn
from tqdm import tqdm

class CrossEntropyLoss(nn.Module):
  """Custom cross-entropy loss"""
  def __init__(self, args):
    super(CrossEntropyLoss, self).__init__()
    tqdm.write('Constructing CrossEntropyLoss')
    self.args = args
    self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

  def forward(self, predictions, label_batch, length_batch):
    """
    Computes and returns CrossEntropyLoss.

    Ignores all entries where label_batch=-1
    Noralizes by the number of sentences in the batch.

    Args: 
      predictions: A pytorch batch of logits
      label_batch: A pytorch batch of label indices
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        cross_entropy_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    if len(label_batch.size()) == 2:
      batchlen, seqlen, class_count = predictions.size()
      total_sents = torch.sum((length_batch != 0)).float()
      predictions = predictions.view(batchlen*seqlen, class_count)
      label_batch = label_batch.view(batchlen*seqlen).long()
      cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch) / total_sents
    return cross_entropy_loss, total_sents


class L1DistanceLoss(nn.Module):
  """Custom L1 loss for distance matrices."""
  def __init__(self, args):
    super(L1DistanceLoss, self).__init__()
    tqdm.write('Constructing L1DistanceLoss')
    self.args = args
    self.word_pair_dims = (1,2)

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted distances
      label_batch: A pytorch batch of true distances
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents


class L1DepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(L1DepthLoss, self).__init__()
    tqdm.write('Constructing L1DepthLoss')
    self.args = args
    self.word_dim = 1

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    total_sents = torch.sum((length_batch != 0)).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
      normalized_loss_per_sent = loss_per_sent / length_batch.float()
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents
