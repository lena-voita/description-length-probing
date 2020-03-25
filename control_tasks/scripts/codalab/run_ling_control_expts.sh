# This script modified from 
# https://github.com/codalab/worksheets-examples/blob/master/01-nli/train.sh

# This is an example script to start a CodaLab run. There are often several
# things to configure, including the docker image, compute resources, bundle
# dependencies (code and data), and custom arguments to pass to the command.
# Factoring all this into a script makes it easier to run and track different
# configurations.

### CodaLab arguments
CODALAB_ARGS="cl run"

# Name of bundle (can customize however you want)
CODALAB_ARGS="$CODALAB_ARGS --name run-ling-control"
# Docker image (default: codalab/default-cpu)
CODALAB_ARGS="$CODALAB_ARGS --request-docker-image codalab/default-gpu"
# Explicitly ask for a worker with at least one GPU
CODALAB_ARGS="$CODALAB_ARGS --request-gpus 1"
# Control the amount of RAM your run needs
#CODALAB_ARGS="$CODALAB_ARGS --request-memory 5g"
# Kill job after this many days (default: 1 day)
CODALAB_ARGS="$CODALAB_ARGS --request-time 2d"

# Bundle dependencies
CODALAB_ARGS="$CODALAB_ARGS :src"                              # Code
CODALAB_ARGS="$CODALAB_ARGS :SNLI"                             # Dataset
CODALAB_ARGS="$CODALAB_ARGS word-vectors.txt:glove.840B.300d"  # Word vectors

### Command to execute (these flags can be overridden) from the command-line
CMD="python src/train_nli.py"
# Read in the dataset
CMD="$CMD --nlipath SNLI"
# Runs can feel free to output to the current directory, which is in the bundle
CMD="$CMD --outputdir ."
# Use the word vectors
CMD="$CMD --word_emb_path word-vectors.txt"
# Train it on a small fraction of the data
CMD="$CMD --train_frac 0.1"
# Number of epochs to train for
CMD="$CMD --n_epochs 3"
# Pass the command-line arguments through to override the above
if [ -n "$1" ]; then
  CMD="$CMD $@"
fi

# Create the run on CodaLab!
FINAL_COMMAND="$CODALAB_ARGS '$CMD'"
echo $FINAL_COMMAND
exec bash -c "$FINAL_COMMAND"
