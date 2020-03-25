python3 scripts/convert_conll_to_raw.py ptb3-wsj-test.conllx > raw.ptb3.test.txt
python3 scripts/convert_conll_to_raw.py ptb3-wsj-train.conllx > raw.ptb3.train.txt
python3 scripts/convert_conll_to_raw.py ptb3-wsj-dev.conllx > raw.ptb3.dev.txt

allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512..5B_weights.hdf5 --cuda-device 0 raw.ptb3.dev.txt raw.dev.elmo-layers.hdf5
allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512..5B_weights.hdf5 --cuda-device 0 raw.ptb3.test.txt raw.test.elmo-layers.hdf5
allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512..5B_weights.hdf5 --cuda-device 0 raw.ptb3.train.txt raw.train.elmo-layers.hdf5

