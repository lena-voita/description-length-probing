import json
from error_analysis import load_observations, vocab, inv_vocab
import torch
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('name')
  argp.add_argument('conllx_filepath')
  argp.add_argument('filepaths', nargs='+')
  #argp.add_argument('')
  args = argp.parse_args()

  
  confusion_matrix = torch.zeros((45,45))
  for filepath in tqdm(args.filepaths):
    tqdm.write('reading\n' + str( filepath))
    observations = iter(load_observations(args.conllx_filepath))
    predictions = json.load(open(filepath))
    for prediction_batch in predictions:
      for prediction in prediction_batch:
        observation = next(observations)
        prediction = torch.argmax(torch.tensor(prediction),1)
        for predicted_pos_int, gold_pos_str in zip(prediction, observation.xpos_sentence):
          gold_pos_int = vocab[gold_pos_str]
          confusion_matrix[gold_pos_int][predicted_pos_int] += 1
  torch.save(confusion_matrix, args.name)

