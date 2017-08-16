# WaveNet implementation in Keras
Based on https://deepmind.com/blog/wavenet-generative-model-raw-audio/ and https://arxiv.org/pdf/1609.03499.pdf.
Forked from https://github.com/basveeling/wavenet on 14/08/2017

Please see description at the original repo

Changes:

1. Adjusted to work with Python3
2. Theano backend extended with Tensorflow
3. Added Nadam optimiser
4. Fixed import of wav file
5. Added a workaround to prevent os.listdir to catch ".DS_Store" as a dir name on Mac OS. It seems to be a python's bug.


### Options:
Train with different configurations:
```$ KERAS_BACKEND=tensorflow python3 wavenet.py with 'option=value' 'option2=value'```
Available options:
```
  batch_size = 16
  data_dir = 'data'
  data_dir_structure = 'flat'
  debug = False
  desired_sample_rate = 4410
  dilation_depth = 9
  early_stopping_patience = 20
  fragment_length = 1152
  fragment_stride = 128
  keras_verbose = 1
  learn_all_outputs = True
  nb_epoch = 1000
  nb_filters = 256
  nb_output_bins = 256
  nb_stacks = 1
  predict_initial_input = ''
  predict_seconds = 1
  predict_use_softmax_as_input = False
  random_train_batches = False
  randomize_batch_order = True
  run_dir = None
  sample_argmax = False
  sample_temperature = 1
  seed = 173213366
  test_factor = 0.1
  train_only_in_receptive_field = True
  use_bias = False
  use_skip_connections = True
  use_ulaw = True
  optimizer:
    decay = 0.0
    epsilon = None
    lr = 0.001
    momentum = 0.9
    nesterov = True
    optimizer = 'sgd'
```

## Using your own training data:
- Create a new data directory with a train and test folder in it. All wave files in these folders will be used as data.
    - Caveat: Make sure your wav files are supported by scipy.io.wavefile.read(): e.g. don't use 24bit wav and remove meta info.
- Run with: `$ python3 wavenet.py with 'data_dir=your_data_dir_name'`
- Test preprocessing results with: `$ python3 wavenet.py test_preprocess with 'data_dir=your_data_dir_name'`



## Uncertainties from paper:
- It's unclear if the model is trained to predict t+1 samples for every input sample, or only for the outputs for which which $t-receptive_field$ was in the input. Right now the code does the latter.
- There is no mention of weight decay, batch normalization in the paper. Perhaps this is not needed given enough data?

## Note on computational cost:
The Wavenet model is quite expensive to train and sample from. We can however trade computation cost with accuracy and fidility by lowering the sampling rate, amount of stacks and the amount of channels per layer.

For a downsampled model 4000 sampling rate, 256 filters, 1 stacks:
- GTX 1080 Ti needs around 50 sec to generate one second of audio.


Kirill Makukhin

