"""Contains classes describing linguistic tasks of interest on annotated data."""

from collections import Counter, OrderedDict

import numpy as np
import sys
import torch
import math
import itertools

PTB_TRAIN_EMPIRICAL_POS_DISTRIBUTION = [0.00003789361998, 0.00006105083219, 0.0001021022538, 0.0001494692788, 0.0001768368932, 0.0002463085299, 0.0003894622053, 0.0004747228503, 0.0009083942789, 0.001437852358, 0.001448378364, 0.001860997781, 0.00204941328, 0.002255722989, 0.002487295111, 0.002802022677, 0.002813601283, 0.003408320597, 0.004519866783, 0.005023009848, 0.00728294324, 0.007465043136, 0.007759771291, 0.008849212865, 0.009158677428, 0.01031864324, 0.01314803353, 0.01562690784, 0.01835314328, 0.02107727351, 0.02281195923, 0.02353299061, 0.02520662549, 0.02782865347, 0.03146117799, 0.03259903919, 0.03849149709, 0.04155456471, 0.05129006724, 0.06300445882, 0.06443704817, 0.08614693462, 0.09627716236, 0.1037379951, 0.1399274548]

#PTB_DEV_EMPIRICAL_DEPTH_DISTRIBUTION = {14: 0.00009970835307, 13: 0.000373906324, 12: 0.0007228855597, 11: 0.001395916943, 10: 0.003938479946, 9: 0.007702470274, 8: 0.01570406561, 7: 0.02921454745, 0: 0.04237605005, 6: 0.05309469801, 5: 0.08729466311, 4: 0.1302440362, 3: 0.183563078, 1: 0.2192088142, 2: 0.22506668}
PTB_DEV_EMPIRICAL_DEPTH_DISTRIBUTION = [0.00009970835307, 0.000373906324, 0.0007228855597, 0.001395916943, 0.003938479946, 0.007702470274, 0.01570406561, 0.02921454745, 0.04237605005, 0.05309469801, 0.08729466311, 0.1302440362, 0.183563078, 0.2192088142, 0.22506668]

PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dict = OrderedDict({-44: 7.690651244347372e-06, -3: 0.047819370772888475, -2: 0.1088534777124927, -1: 0.277384211752194, 4: 0.035580248649741374, 1: 0.17205854563192982, 3: 0.06036172428795556, 2: 0.09961151224571411, -4: 0.02238199244997781, 15: 0.003433326448369362, 6: 0.01574166443271559, 7: 0.011697480542652352, 8: 0.009206808203947281, 11: 0.00579765237377444, -13: 0.0016556873464616411, -11: 0.002414864490725075, 5: 0.022290803299509117, -8: 0.004191404928169318, 19: 0.0021665663219790025, -7: 0.005423007791728375, -5: 0.012027079881695811, 9: 0.00793565341970301, 22: 0.0015447222356503435, -10: 0.0029543087422928688, -19: 0.0007163292301877837, -6: 0.00748410232521347, 12: 0.004976950019556227, 35: 0.0003317966679704152, 13: 0.004389164531595393, 18: 0.002396187194845945, -9: 0.0034783716913719684, 28: 0.0008723395840016876, 43: 0.00011865576205564516, -17: 0.0009151874980773372, -12: 0.0020545025467042263, 26: 0.0009964886683747236, 25: 0.0011404137130903674, -23: 0.0003471779704591099, -26: 0.00023731152411129032, 20: 0.001866630923449455, 34: 0.00038343389775389035, 10: 0.006666695964385693, 36: 0.0002955407406756347, -22: 0.00042518314736606183, -15: 0.0012920294090503583, -21: 0.0005306549358599686, 16: 0.0030652738531041666, 17: 0.0026005387850528898, -16: 0.001105256450259065, 14: 0.003947501417277158, 23: 0.001423869144667742, -20: 0.0005767988433260529, 21: 0.0017677511217364173, 32: 0.00048780702178431896, 38: 0.0002647781356982452, 37: 0.0002450021753556377, 50: 4.834123639304062e-05, 46: 6.042654549130078e-05, 31: 0.0005910814813512694, -14: 0.0015601035381390383, 27: 0.0009470487675182048, 45: 0.00010107713063999403, 24: 0.0012953254024407929, 42: 0.00013623439347129629, 29: 0.000745993170701695, 40: 0.00020654891913390083, 41: 0.00013953038686173087, 47: 5.49332231739098e-05, 30: 0.0006273374086460499, -18: 0.0008174063608277777, 56: 1.7578631415651135e-05, -35: 4.1749249612171444e-05, -27: 0.0001658983339852076, 39: 0.00019885826788955345, 33: 0.0004647350680512769, -31: 8.789315707825567e-05, 57: 2.1973289269563917e-05, 61: 1.867729587912933e-05, -30: 0.00011975442651912336, 44: 8.239983476086469e-05, -24: 0.00028455409604085275, -29: 0.000106570452957385, -25: 0.0002614821423078106, 65: 8.789315707825568e-06, 49: 4.834123639304062e-05, 51: 3.186126944086768e-05, 62: 1.0986644634781959e-05, 90: 1.098664463478196e-06, -36: 3.405859836782407e-05, -28: 0.00013953038686173087, -38: 2.1973289269563917e-05, -33: 6.921586119912634e-05, 52: 2.3071953733042113e-05, 55: 1.867729587912933e-05, 72: 4.394657853912784e-06, 73: 3.295993390434588e-06, 77: 2.197328926956392e-06, 85: 1.098664463478196e-06, 48: 5.603188763738799e-05, 68: 5.493322317390979e-06, -32: 6.482120334521356e-05, -40: 1.4282638025216547e-05, 53: 2.417061819652031e-05, 54: 2.5269282659998507e-05, 100: 1.098664463478196e-06, -34: 6.372253888173536e-05, -39: 2.3071953733042113e-05, -48: 3.295993390434588e-06, -37: 2.3071953733042113e-05, -67: 1.098664463478196e-06, -64: 2.197328926956392e-06, -63: 1.098664463478196e-06, -59: 1.098664463478196e-06, -41: 9.887980171303763e-06, 58: 1.2085309098260154e-05, -47: 3.295993390434588e-06, 59: 9.887980171303763e-06, 60: 9.887980171303763e-06, 63: 1.0986644634781959e-05, 67: 3.295993390434588e-06, 79: 3.295993390434588e-06, 64: 6.591986780869176e-06, 69: 2.197328926956392e-06, -43: 5.493322317390979e-06, 80: 1.098664463478196e-06, 81: 1.098664463478196e-06, -58: 1.098664463478196e-06, -56: 1.098664463478196e-06, -42: 5.493322317390979e-06, -49: 1.098664463478196e-06, 74: 4.394657853912784e-06, 75: 3.295993390434588e-06, 117: 1.098664463478196e-06, -62: 1.098664463478196e-06, 76: 1.098664463478196e-06, 78: 2.197328926956392e-06, -53: 2.197328926956392e-06, -65: 1.098664463478196e-06, -61: 1.098664463478196e-06, 127: 1.098664463478196e-06, -45: 4.394657853912784e-06, -46: 1.098664463478196e-06, -50: 1.098664463478196e-06, -77: 1.098664463478196e-06, -74: 1.098664463478196e-06, 70: 2.197328926956392e-06, 66: 1.098664463478196e-06, -55: 1.098664463478196e-06, -54: 2.197328926956392e-06, -66: 1.098664463478196e-06, 71: 2.197328926956392e-06, 83: 1.098664463478196e-06, 87: 1.098664463478196e-06, 86: 1.098664463478196e-06})
PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dists = list(sorted(PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dict.keys()))
PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_probs = [PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dict[x] for x in PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dists]

class Task:
  """Abstract class representing a linguistic task mapping texts to labels."""

  def __init__(self, args):
    args['ignore_punct'] = True 

  def _register_observation(self):
    """
    For labeling tasks that require a label vocabulary, keep state
    that determines how future observations' labels should be encoded
    as integers, etc.
    """
    pass

  def prepare(self, train_obs, dev_obs, test_obs):
    """Prepares task with corpus-specific information.

    If the distribution of a certain quantity in the dataset must be known
    for the definition of the task, this computes the necessary
    statistics and stores them in the state of the task for future
    use in label assignment.

    A noop if no statistics are needed.

    Args:
      observations: the (training) observations of a dataset
    """
    pass

  def labels(self, observation):
    """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class ParseDistanceTask(Task):
  """Maps observations to dependency parse distances between words."""

  @staticmethod
  def labels(observation):
    """Computes the distances between all pairs of words; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length, sentence_length) of distances
      in the parse tree as specified by the observation annotation.
    """
    sentence_length = len(observation[0]) #All observation fields must be of same length
    distances = torch.zeros((sentence_length, sentence_length))
    for i in range(sentence_length):
      for j in range(i,sentence_length):
        i_j_distance = ParseDistanceTask.distance_between_pairs(observation, i, j)
        distances[i][j] = i_j_distance
        distances[j][i] = i_j_distance
    return distances

  @staticmethod
  def distance_between_pairs(observation, i, j, head_indices=None):
    '''Computes path distance between a pair of words

    TODO: It would be (much) more efficient to compute all pairs' distances at once;
          this pair-by-pair method is an artefact of an older design, but
          was unit-tested for correctness... 

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: one of the two words to compute the distance between.
      j: one of the two words to compute the distance between.
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer distance d_path(i,j)
    '''
    if i == j:
      return 0
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    i_path = [i+1]
    j_path = [j+1]
    i_head = i+1
    j_head = j+1
    while True:
      if not (i_head == 0 and (i_path == [i+1] or i_path[-1] == 0)):
        i_head = head_indices[i_head - 1]
        i_path.append(i_head)
      if not (j_head == 0 and (j_path == [j+1] or j_path[-1] == 0)):
        j_head = head_indices[j_head - 1]
        j_path.append(j_head)
      if i_head in j_path:
        j_path_length = j_path.index(i_head)
        i_path_length = len(i_path) - 1
        break
      elif j_head in i_path:
        i_path_length = i_path.index(j_head)
        j_path_length = len(j_path) - 1
        break
      elif i_head == j_head:
        i_path_length = len(i_path) - 1
        j_path_length = len(j_path) - 1
        break
    total_length = j_path_length + i_path_length
    return total_length

#class CorruptedParseDistanceTask(Task):
#  """Unfinished..."""
#
#  def __init__(self, args):
#    args['ignore_punct'] = False # Global setting to make sure the tree doesn't get messed up at test time
#    self.target_corrupted_token_percent = args['probe']['misc']['corrupted_token_percent']
#    #self.REPLACE_TOKEN = '<RPL>'
#    self.dist = PTB_DEV_EMPIRICAL_DEPTH_DISTRIBUTION
#    self.args = args
#    self.rand_root_type_vocab = set()
#    self.rand_type_vocab = {}
#
#  def labels(self, observation):
#    sentence_length = len(observation[0]) #All observation fields must be of same length
#    #distances = torch.zeros((sentence_length, sentence_length))
#    depths = torch.tensor([self.rand_type_vocab[x] for x in observation.sentence]).float()
#    depth_differences = depths.repeat(sentence_length, 1) - depths.repeat(sentence_length, 1).t()
#    positions = torch.tensor(list(range(sentence_length))).float()
#    position_differences = (positions.repeat(sentence_length, 1) - positions.repeat(sentence_length, 1).t()) * .25
#    #print(position_differences)
#    #print(depth_differences)
#    distances = torch.abs(depth_differences) + torch.abs(position_differences)
#    #print(torch.abs(depth_differences))
#    #print(torch.abs(position_differences))
#    #print(distances)
#    #print()
#    #print()
#    #print()
#    return distances
#
#  def _register_type(self, string):
#    if string not in self.rand_type_vocab:
#      ints = list(range(5))
#      self.rand_type_vocab[string] = int(np.random.choice(ints))
#      #self.rand_type_vocab[string] = np.random.random()
#    return self.rand_type_vocab[string]
#
#  def prepare(self, train_obs, dev_obs, test_obs):
#    """Chooses the word types to be part-of-speech-corrupted in all datasets.
#    """
#    np.random.seed(self.args['seed'])
#    root_type_counter = Counter()
#    root_intmapping = {}
#
#    type_counter = Counter()
#    intmapping = {}
#
#    for observation in itertools.chain(train_obs, dev_obs, test_obs):
#      type_counter.update(observation.sentence)
#      for string, head_index in zip(observation.sentence, observation.head_indices):
#
#        if string not in intmapping:
#          intmapping[string] = len(intmapping)
#
#        if int(head_index) == 0: # Only consider the root of each sent
#          root_type_counter.update([string])
#          if string not in root_intmapping:
#            root_intmapping[string] = len(root_intmapping)
#
#    root_counts = [root_type_counter[string] for string in sorted(root_intmapping, key=lambda x: root_intmapping[x])]
#    root_strings = [string for string in sorted(root_intmapping, key=lambda x: root_intmapping[x])]
#    root_count_sum = sum(root_counts)
#    root_probs = [x/root_count_sum for x in root_counts]
#
#    corrupted_token_percent = 0
#    while corrupted_token_percent < self.target_corrupted_token_percent - .00000001:
#      remaining_strings = list(filter(lambda x: x, root_strings))
#      string = np.random.choice(remaining_strings)
#      prob = root_probs[root_intmapping[string]]
#      root_strings[root_intmapping[string]] = None
#      if string not in self.rand_root_type_vocab:
#        self.rand_root_type_vocab.add(string)
#        self._register_type(string)
#        corrupted_token_percent += prob
#
#    for string in intmapping:
#      if string not in self.rand_type_vocab:
#        self._register_type(string)
#    print('CORRUPTED', self.rand_type_vocab)
#    print('CORRUPTED', self.rand_root_type_vocab)
#    print('CORRUPTED', corrupted_token_percent)
#
#  def get_head_indices(observation):
#    for index, string in enumerate(observation.sentence):
#      pass
    

#class EdgeLabelTask(Task):
#
#  @staticmethod
#  def labels(observation):
#    """Computes the distances between all pairs of words; returns them as a torch tensor.
#
#    Args:
#      observation: a single Observation class for a sentence:
#    Returns:
#      A torch tensor of shape (sentence_length, sentence_length) of distances
#      in the parse tree as specified by the observation annotation.
#    """
#    sentence_length = len(observation[0]) #All observation fields must be of same length
#    labels = torch.zeros((sentence_length, sentence_length))
#    modified_head_indices = [int(x)-1 if x != '0' else 0 for x in observation.head_indices]
#    for i, word_i in enumerate(observation.sentence):
#      for j_prime, word_j in enumerate(observation.sentence[i:]):
#        j = j_prime + i
#        i_j_label = int(modified_head_indices[i] == j) #or modified_head_indices[j] == i)
#        labels[i][j] = i_j_label
#        #labels[j][i] = i_j_label
#    return labels

class CorruptedEdgePositionTask(Task):

  def __init__(self, args):
    self.label_dict = {}
    self.strings = set()
    args['ignore_punct'] = True
    self.args = args
    self.target_corrupted_token_percent = args['probe']['misc']['corrupted_token_percent']

  def _register_type(self, tup):
    if tup not in self.label_dict:
      #a = torch.tensor(np.random.choice(PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dists, p=PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_probs))
      #a = torch.tensor(int(np.random.choice([-1,0,1,2], p=[0.25,0.25,0.25,0.25])))
      a = torch.tensor(int(np.random.choice([0,1,2], p=[1/3,1/3,1/3])))
      self.label_dict[tup] = a
    return self.label_dict[tup]

  def prepare(self, train_obs, dev_obs, test_obs):
    tuple_counter = Counter()
    for observation in itertools.chain(train_obs, dev_obs, test_obs):
      for word1 in observation.sentence:
        tuple_counter.update([word1])

    np.random.seed(self.args['seed'])
    seen_tuples = set()
    all_tuples = list(tuple_counter.keys())
    np.random.shuffle(all_tuples)
    tuple_count_sum = sum(tuple_counter.values())
    corrupted_pair_percent = 0
    index = 0
    while corrupted_pair_percent < self.target_corrupted_token_percent - 0.00000001:
      next_tuple = all_tuples[index]
      if next_tuple not in seen_tuples:
        seen_tuples.add(next_tuple)
        tuple_probability_mass = tuple_counter[next_tuple] / tuple_count_sum
        corrupted_pair_percent += tuple_probability_mass 
        self._register_type(next_tuple)
      index += 1

    #print('CORRUPTED', self.label_dict)
    print('CORRUPTED', corrupted_pair_percent)

  def labels(self, observation):

    sentence_length = len(observation[0]) #All observation fields must be of same length
    labels = torch.zeros(sentence_length)
    modified_head_indices = torch.tensor([int(x)-1 if x != '0' else -1 for x in observation.head_indices])
    root_index = observation.head_indices.index('0')

    for i, word_i in enumerate(observation.sentence):
      if word_i in self.label_dict:
        #modified_head_indices[i] = max(min(i + self.label_dict[word_i], len(observation.sentence)-1),0)
        #if self.label_dict[word_i] == -1:
        #   modified_head_indices[i] = root_index
        if self.label_dict[word_i] == 0:
           modified_head_indices[i] = i
        elif self.label_dict[word_i] == 1:
           modified_head_indices[i] = 0
        elif self.label_dict[word_i] == 2:
           modified_head_indices[i] = len(observation.sentence) - 1
        else:
           raise ValueError("Fix this")

        
    return modified_head_indices

class CorruptedEdgeLabelTask(Task):

  def __init__(self, args):
    self.label_dict = {}
    self.strings = set()
    args['ignore_punct'] = True
    self.args = args
    self.target_corrupted_token_percent = args['probe']['misc']['corrupted_token_percent']

  def _register_type(self, tup):
    if tup not in self.label_dict:
      ints = list(range(2))
      #probs = [0.25, 0.75]
      probs = [0.5,0.5]
      #probs = [0,1]
      label1 = int(np.random.choice(ints,p=probs))
      label2 = int(np.random.choice(ints,p=probs))
      self.label_dict[tup] = label1
      self.label_dict[(tup[1], tup[0])] = label2
    return self.label_dict[tup]

  def prepare(self, train_obs, dev_obs, test_obs):
    tuple_counter = Counter()
    for observation in itertools.chain(train_obs, dev_obs, test_obs):
      for word1 in observation.sentence:
        for word2 in observation.sentence:
          if (word1, word2) not in self.label_dict:
            tuple_counter.update([(word1, word2), (word2, word1)])

    np.random.seed(self.args['seed'])
    seen_tuples = set()
    all_tuples = list(tuple_counter.keys())
    np.random.shuffle(all_tuples)
    tuple_count_sum = sum(tuple_counter.values())
    corrupted_pair_percent = 0
    index = 0
    while corrupted_pair_percent < self.target_corrupted_token_percent - 0.00000001:
      next_tuple = all_tuples[index]
      if next_tuple not in seen_tuples:
        seen_tuples.add(next_tuple)
        tuple_probability_mass = tuple_counter[next_tuple] / tuple_count_sum
        corrupted_pair_percent += tuple_probability_mass 
        self._register_type(next_tuple)
      index += 1

    #print('CORRUPTED', self.label_dict)
    print('CORRUPTED', corrupted_pair_percent)

  def labels(self, observation):

    sentence_length = len(observation[0]) #All observation fields must be of same length
    labels = torch.zeros(sentence_length)
    modified_head_indices = torch.tensor([int(x)-1 if x != '0' else -1 for x in observation.head_indices])

    for i, word_i in enumerate(observation.sentence):
      tups = [(x, word_i) for x in observation.sentence]
      scores = [self.label_dict[tup] if tup in self.label_dict else -1 for tup in tups]
      closest_score = sys.maxsize
      closest = -1
      for index, score in enumerate(scores):
        if score == 1:
          diff = abs(i - index)
          if diff != 0 and diff < closest_score:
            closest = index
            closest_score = diff
      if closest != -1:
        modified_head_indices[i]= closest
        
    return modified_head_indices


class BalancedBinaryTreeDistanceTask:

  def __init__(self, args):
    self.distance_dict = {}
    args['ignore_punct'] = False # Global setting to make sure the tree doesn't get messed up at test time

  def labels(self, observation):
    sentence_length = len(observation[0]) #All observation fields must be of same length
    if sentence_length in self.distance_dict:
      return self.distance_dict[sentence_length]
    distances = torch.zeros((sentence_length, sentence_length))
    head_indices = BalancedBinaryTreeDistanceTask.get_head_indices(sentence_length)
    for i in range(sentence_length):
      for j in range(i,sentence_length):
        i_j_distance = ParseDistanceTask.distance_between_pairs(None, i, j, head_indices)
        distances[i][j] = i_j_distance
        distances[j][i] = i_j_distance
    self.distance_dict[sentence_length] = distances
    return distances

  @staticmethod
  def get_head_indices(sentence_length):
    head_indices = [-1 for x in range(sentence_length)]
    root_index = int(sentence_length/2) # Even or odd, doesn't matter

    BalancedBinaryTreeDistanceTask._assign(head_indices, 0, root_index - 1, root_index)
    BalancedBinaryTreeDistanceTask._assign(head_indices, root_index + 1, sentence_length-1, root_index)

    head_indices = [x+1 for x in head_indices]
    return head_indices

  @staticmethod
  def _assign(array, start_index, end_index, parent_index):
    if ((start_index < 0 or end_index > len(array) - 1) or
        (end_index < 0 or start_index > len(array) - 1)):
      return
    # Base case -- single node
    if start_index == end_index:
      array[start_index] = parent_index
      return
    # Choose the child index
    if (end_index - start_index) % 2 == 0: # Odd # of elts
      child_index = int((end_index + start_index)/2)
    else:
      right_child_candidate = math.ceil((end_index + start_index)/2)
      left_child_candidate = math.floor((end_index + start_index)/2)
      if abs(right_child_candidate - parent_index) > abs(left_child_candidate - parent_index):
        child_index = right_child_candidate
      elif abs(left_child_candidate - parent_index) > abs(right_child_candidate - parent_index):
        child_index = left_child_candidate
      else:
        raise ValueError("Something's going on with child indices you don't understand.")
    # Assign child to parent
    array[child_index] = parent_index
    # Call new functions for newly made subdivisions
    if child_index != start_index:
      BalancedBinaryTreeDistanceTask._assign(array, start_index, child_index-1, child_index)
    if child_index != end_index:
      BalancedBinaryTreeDistanceTask._assign(array, child_index+1, end_index, child_index)


class ParseDepthTask(Task):
  """Maps observations to a depth in the parse tree for each word"""

  @staticmethod
  def labels(observation):
    """Computes the depth of each word; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length,) of depths
      in the parse tree as specified by the observation annotation.
    """
    sentence_length = len(observation[0]) #All observation fields must be of same length
    depths = torch.zeros(sentence_length)
    for i in range(sentence_length):
      depths[i] = ParseDepthTask.get_ordering_index(observation, i)
    return depths

  @staticmethod
  def get_ordering_index(observation, i, head_indices=None):
    '''Computes tree depth for a single word in a sentence

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: the word in the sentence to compute the depth of
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer depth in the tree of word i
    '''
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
      i_head = head_indices[i_head - 1]
      if i_head != 0:
        length += 1
      else:
        return length

class RandomParseDepthTask(Task):
  """Maps observations to a random sample from depths in the parse tree"""
  
  def __init__(self, args):
    self.vocab = {}
    self.args = args
    self.dist = PTB_DEV_EMPIRICAL_DEPTH_DISTRIBUTION
    args['ignore_punct'] = True 
    np.random.seed(args['seed'])

  def get_label(self):
    ints = list(range(15))
    return int(np.random.choice(ints, p=self.dist))

  def _register_observation(self, observation):
    for string in observation.sentence:
      if string not in self.vocab:
        self.vocab[string] = self.get_label()

  def labels(self, observation):
    self._register_observation(observation)
    sentence_length = len(observation[0]) 
    labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      labels[i] = self.vocab[observation.sentence[i]]
    return labels

class CorruptedParseDepthTask(ParseDepthTask):

  def __init__(self, args):
    self.distance_dict = {}
    args['ignore_punct'] = True # Global setting to make sure the tree doesn't get messed up at test time
    self.args = args
    self.rand_type_vocab = {}
    self.target_corrupted_token_percent = args['probe']['misc']['corrupted_token_percent']

  def labels(self, observation):
    sentence_length = len(observation[0]) #All observation fields must be of same length
    distances = torch.zeros((sentence_length, sentence_length))

  def _register_type(self, string):
    if string not in self.rand_type_vocab:
      #self.rand_type_vocab[string] = int(np.random.choice(ints, p=self.dist))
      ints = list(range(5))
      self.rand_type_vocab[string] = int(np.random.choice(ints))
    return self.rand_type_vocab[string]

  def prepare(self, train_obs, dev_obs, test_obs):
    """Chooses the word types to be part-of-speech-corrupted in all datasets.
    """
    np.random.seed(self.args['seed'])
    type_counter = Counter()
    intmapping = {}
    for observation in itertools.chain(train_obs, dev_obs, test_obs):
      type_counter.update(observation.sentence)
      for string in observation.sentence:
        if string not in intmapping:
          intmapping[string] = len(intmapping)
    counts = [type_counter[string] for string in sorted(intmapping, key=lambda x: intmapping[x])]
    strings = [string for string in sorted(intmapping, key=lambda x: intmapping[x])]
    count_sum = sum(counts)
    probs = [x/count_sum for x in counts]

    corrupted_token_percent = 0
    while corrupted_token_percent < self.target_corrupted_token_percent - .00000001:
      remaining_strings = list(filter(lambda x: x, strings))
      string = np.random.choice(remaining_strings)
      prob = probs[intmapping[string]]
      strings[intmapping[string]] = None
      if string not in self.rand_type_vocab:
        self._register_type(string)
        corrupted_token_percent += prob
    #print('CORRUPTED', self.rand_type_vocab)
    print('CORRUPTED', corrupted_token_percent)

  def labels(self, observation):
    labels = super(CorruptedParseDepthTask, self).labels(observation)
    for index, string in enumerate(observation.sentence):
      if string in self.rand_type_vocab:
        labels[index] = self.rand_type_vocab[string]
    return labels


class RandomLinearParseDepthTask(Task):
  """Maps observations to a random sample from depths in the parse tree
  plus their linear position in the sequence."""
  
  def __init__(self, args):
    self.vocab = {}
    self.args = args
    self.dist = PTB_DEV_EMPIRICAL_DEPTH_DISTRIBUTION
    args['ignore_punct'] = True 

  def get_label(self):
    ints = list(range(15))
    return int(np.random.choice(ints, p=self.dist))

  def _register_observation(self, observation):
    for string in observation.sentence:
      if string not in self.vocab:
        self.vocab[string] = self.get_label()

  def labels(self, observation):
    self._register_observation(observation)
    sentence_length = len(observation[0]) 
    labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      labels[i] = self.vocab[observation.sentence[i]] + i
    return labels


class PartOfSpeechLabelTask(Task):
  """
  Computes the POS of the word in the sentence.
  Requires the pos_sentence field.
  """
  def __init__(self, args):
    self.vocab = {}
    self.args = args

  def _register_observation(self, observation):
    for string in observation.xpos_sentence:
      if string not in self.vocab:
        self.vocab[string] = len(self.vocab)
    self.args['probe']['label_space_size'] = len(self.vocab)
    self.args['probe']['label_space'] = self.vocab

  def labels(self, observation):
    self._register_observation(observation)
    sentence_length = len(observation[0]) 
    labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      labels[i] = self.vocab[observation.xpos_sentence[i]]
    return labels

class CorruptedPartOfSpeechLabelTask(PartOfSpeechLabelTask):

  def __init__(self, args):
    super(CorruptedPartOfSpeechLabelTask, self).__init__(args)
    self.rand_type_vocab = {}
    self.dist = PTB_TRAIN_EMPIRICAL_POS_DISTRIBUTION
    self.target_corrupted_token_percent = args['probe']['misc']['corrupted_token_percent']
    np.random.seed(args['seed'])

  def prepare(self, train_obs, dev_obs, test_obs):
    """Chooses the word types to be part-of-speech-corrupted in all datasets.
    """
    type_counter = Counter()
    intmapping = {}
    for observation in itertools.chain(train_obs, dev_obs, test_obs):
      type_counter.update(observation.sentence)
      for string in observation.sentence:
        if string not in intmapping:
          intmapping[string] = len(intmapping)
    counts = [type_counter[string] for string in sorted(intmapping, key=lambda x: intmapping[x])]
    strings = [string for string in sorted(intmapping, key=lambda x: intmapping[x])]
    count_sum = sum(counts)
    probs = [x/count_sum for x in counts]

    np.random.shuffle(strings)
    index = 0
    corrupted_token_percent = 0
    while corrupted_token_percent < self.target_corrupted_token_percent - .00000001:
      #remaining_strings = list(filter(lambda x: x, strings))
      #string = np.random.choice(remaining_strings)
      string = strings[index]
      index += 1
      prob = probs[intmapping[string]]
      #strings[intmapping[string]] = None
      if string not in self.rand_type_vocab:
        self._register_type(string)
        corrupted_token_percent += prob
    #print('CORRUPTED', self.rand_type_vocab)
    print('CORRUPTED', corrupted_token_percent)

  def _register_type(self, string):
    if string not in self.rand_type_vocab:
      ints = list(range(45))
      self.rand_type_vocab[string] = int(np.random.choice(ints, p=self.dist))
    return self.rand_type_vocab[string]

  def labels(self, observation):
    labels = super(CorruptedPartOfSpeechLabelTask, self).labels(observation)
    for index, string in enumerate(observation.sentence):
      if string in self.rand_type_vocab:
        labels[index] = self.rand_type_vocab[string]
      #if random.random() < 0.2:
      #  labels[index] = self._register_type(string)
    self.args['probe']['label_space_size'] = 45
    return labels

class RandomPrefixLabelTask(Task):
  """
  Computes the POS of the word in the sentence.
  Requires the pos_sentence field.
  """
  def __init__(self, args):
    self.vocab = {}
    self.args = args
    self.condition_length = args['probe']['misc']['rand_label_condition_length']
    self.dist = PTB_TRAIN_EMPIRICAL_POS_DISTRIBUTION

  def get_label(self):
    ints = list(range(45))
    return int(np.random.choice(ints, p=self.dist))

  def _register_observation(self, observation):
    prefix = ()
    for string in observation.sentence:
      prefix = prefix + (string,)
      if prefix[-1 - self.condition_length:] not in self.vocab:
        self.vocab[prefix[-1- self.condition_length:]] = self.get_label()
    self.args['probe']['label_space_size'] = 45

  def labels(self, observation):
    self._register_observation(observation)
    sentence_length = len(observation[0]) 
    labels = torch.zeros(sentence_length)
    prefix = ()
    for i in range(sentence_length):
      prefix = prefix + (observation.sentence[i],)
      labels[i] = self.vocab[prefix[-1- self.condition_length:]]
    return labels

class RandomWordLabelTask(Task):
  """
  Computes the POS of the word in the sentence.
  Requires the pos_sentence field.
  """
  def __init__(self, args):
    self.vocab = {}
    self.args = args
    self.dist = PTB_TRAIN_EMPIRICAL_POS_DISTRIBUTION

  def get_label(self):
    ints = list(range(45))
    return int(np.random.choice(ints, p=self.dist))


  def _register_observation(self, observation):
    for string in observation.sentence:
      if string not in self.vocab:
        self.vocab[string] = self.get_label()
    self.args['probe']['label_space_size'] = 45
    #self.args['probe']['label_space'] = self.vocab

  def labels(self, observation):
    self._register_observation(observation)
    sentence_length = len(observation[0]) 
    labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      labels[i] = self.vocab[observation.sentence[i]]
    return labels
