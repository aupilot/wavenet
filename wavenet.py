from __future__ import absolute_import, division, print_function

import datetime
import json
import os
import re
import wave

import keras.backend as K
import numpy as np
import scipy.io.wavfile
import scipy.signal
# import theano
import tensorflow as tf
from keras import layers, regularizers
from keras import metrics
from keras import objectives
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, TerminateOnNaN
from keras.engine import Input
from keras.engine import Model
from keras.optimizers import Adam, SGD, Nadam
from keras.regularizers import l2
from keras.utils import plot_model
from sacred import Experiment
from sacred.commands import print_config
from tqdm import tqdm

import dataset
from wavenet_utils import categorical_mean_squared_error

ex = Experiment('wavenet')


@ex.config
def config():
    data_dir = 'data'
    data_dir_structure = 'flat'  # Or 'vctk' for a speakerdir structure
    test_factor = 0.1  # For 'vctk' structure, take test_factor amount of sequences for test set.
    nb_epoch = 1000
    run_dir = None
    early_stopping_patience = 20
    desired_sample_rate = 4410
    batch_size = 16
    nb_output_bins = 256
    nb_filters = 256
    dilation_depth = 9
    nb_stacks = 1
    use_bias = True
    use_ulaw = True
    res_l2 = 0.000
    final_l2 = 0.000
    fragment_length = 128 + compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks)[0]
    fragment_stride = 128
    use_skip_connections = True
    optimizer = {
        'optimizer': 'sgd',
        'lr': 0.001,
        'momentum': 0.9,
        'decay': 0.,
        'nesterov': True,
        'epsilon': 1e-8
    }

    learn_all_outputs = True
    random_train_batches = False
    randomize_batch_order = True  # Only effective if not using random train batches
    train_with_soft_target_stdev = 0.5  # float to make targets a gaussian with stdev. # None

    # The temporal-first outputs are computed from zero-padding. Setting below to True ignores these inputs:
    train_only_in_receptive_field = True

    keras_verbose = 1
    debug = False


@ex.named_config
def book():
    desired_sample_rate = 4000
    data_dir = 'data_book'
    dilation_depth = 8
    nb_stacks = 1
    fragment_length = 2 ** 10
    nb_filters = 256
    batch_size = 16
    fragment_stride = compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks)[0]


@ex.named_config
def small():
    desired_sample_rate = 4800 #4410
    nb_filters = 16
    dilation_depth = 8
    nb_stacks = 1
    fragment_length = 128 + (compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks)[0])
    fragment_stride = int(desired_sample_rate / 10)


@ex.named_config
def soft_targets():
    train_with_soft_target_stdev = 0.5
    # TODO: smooth decay of stdev per epoch.


@ex.named_config
def vctkdata():
    assert os.path.isdir(os.path.join('vctk', 'VCTK-Corpus')), "Please download vctk by running vctk/download_vctk.sh."
    desired_sample_rate = 4000 #4000
    data_dir = 'vctk/VCTK-Corpus/wav48'
    data_dir_structure = 'vctk'
    test_factor = 0.01


@ex.named_config
def vctkmod(desired_sample_rate):
    nb_filters = 32
    dilation_depth = 7
    nb_stacks = 4
    fragment_length = 1 + (compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks)[0])
    fragment_stride = int(desired_sample_rate / 10)
    random_train_batches = True

@ex.named_config
def length32(desired_sample_rate, dilation_depth, nb_stacks):
    fragment_length = 32 + (compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks)[0])


@ex.named_config
def adam():
    optimizer = {
        'optimizer': 'adam',
        'lr': 0.001,
        'decay': 0.,
        'epsilon': 1e-8
    }

@ex.named_config
def adam2():
    optimizer = {
        'optimizer': 'adam',
        'lr': 0.01,
        'decay': 0.,
        'epsilon': 1e-10
    }

@ex.named_config
def nadam():
        optimizer = {
        'optimizer': 'nadam',
        'lr': 0.0005,
        'epsilon': 1e-8
    }


@ex.config
def predict_config():
    predict_seconds = 1
    sample_argmax = False
    sample_temperature = 1.0  # Temperature for sampling. > 1.0 for more exploring, < 1.0 for conservative samples.
    predict_use_softmax_as_input = False  # Uses the softmax rather than the argmax as in input for the next step.
    predict_initial_input = None


@ex.named_config
def batch_run():
    keras_verbose = 2


def skip_out_of_receptive_field(func):
    # TODO: consider using keras masking for this?
    receptive_field, _ = compute_receptive_field()

    def wrapper(y_true, y_pred):
        y_true = y_true[:, receptive_field - 1:, :]
        y_pred = y_pred[:, receptive_field - 1:, :]
        return func(y_true, y_pred)

    wrapper.__name__ = func.__name__

    return wrapper


# def print_t(tensor, label):
#     tensor.name = label
#     tensor = teano.printing.Print(tensor.name, attrs=('__str__', 'shape'))(tensor)
#     return tensor

# kir
def print_t(tensor, label):
    K.print_tensor(tensor, label)
    return tensor


@ex.capture
def make_soft(y_true, fragment_length, nb_output_bins, train_with_soft_target_stdev, with_prints=False):
    receptive_field, _ = compute_receptive_field()
    n_outputs = fragment_length - receptive_field + 1

    # Make a gaussian kernel.
    kernel_v = scipy.signal.gaussian(9, std=train_with_soft_target_stdev)
    print(kernel_v)

    if K.backend() == 'theano':
        kernel_v = np.reshape(kernel_v, [1, 1, -1, 1])
    else:
        kernel_v = np.reshape(kernel_v, [1, -1, 1, 1])  # tf order

    kernel = K.variable(kernel_v)

    if with_prints:
        y_true = print_t(y_true, 'y_true initial')

    # y_true: [batch, timesteps, input_dim]
    y_true = K.reshape(y_true, (-1, 1, nb_output_bins, 1))  # Same filter for all output; combine with batch.
    # y_true: [batch*timesteps, n_channels=1, input_dim, dummy]

    if K.backend() != 'theano':
        y_true = tf.cast(y_true, 'float32')
        y_true = K.conv2d(tf.transpose(y_true, (0, 2, 3, 1)), kernel, padding='same')

    y_true = K.reshape(y_true, (-1, n_outputs, nb_output_bins))  # Same filter for all output; combine with batch.

    # y_true: [batch, timesteps, input_dim]
    y_true /= K.sum(y_true, axis=-1, keepdims=True)

    if with_prints:
        y_true = print_t(y_true, 'y_true after')
    return y_true


def make_targets_soft(func):
    """Turns one-hot into gaussian distributed."""

    def wrapper(y_true, y_pred):
        y_true = make_soft(y_true)
        y_pred = y_pred
        return func(y_true, y_pred)

    wrapper.__name__ = func.__name__

    return wrapper


@ex.capture()
def build_model(fragment_length, nb_filters, nb_output_bins, dilation_depth, nb_stacks, use_skip_connections,
                learn_all_outputs, _log, desired_sample_rate, use_bias, res_l2, final_l2):

    def residual_block(x):
        original_x = x

        tanh_out = layers.Conv1D(nb_filters, kernel_size=2,
                                 padding='causal',
                                 use_bias=use_bias,
                                 # activation='linear', <-- does not work!
                                 # activation='selu',
                                 activation='tanh',
                                 dilation_rate=2 ** i,
                                 activity_regularizer = l2(res_l2),
                                 name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)

        sigm_out = layers.Conv1D(nb_filters, kernel_size=2,
                                 padding='causal',
                                 use_bias=use_bias,
                                 activation='sigmoid',
                                 dilation_rate=2 ** i,
                                 activity_regularizer=l2(res_l2),
                                 name='dilated_conv_%d_sigm_s%d' % (2 ** i, s))(x)

        x = layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x  = layers.Conv1D(nb_filters, 1, padding='same', use_bias=use_bias, activity_regularizer=l2(res_l2))(x)
        skip_x = layers.Conv1D(nb_filters, 1, padding='same', use_bias=use_bias, activity_regularizer=l2(res_l2))(x)

        res_x = layers.Add()([original_x, res_x])

        return res_x, skip_x

    input = Input(shape=(fragment_length, nb_output_bins), name='input_part')

    out = input
    skip_connections = []

    out = layers.Conv1D(nb_filters, kernel_size=2, padding='causal', dilation_rate=1, name='initial_causal_conv')(out)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = residual_block(out)
            skip_connections.append(skip_out)

    if use_skip_connections:
        out = layers.Add()(skip_connections)

    out = layers.Activation('selu')(out)
    out = layers.Conv1D(nb_output_bins, 1, padding='same', activity_regularizer=l2(final_l2))(out)
    out = layers.Activation('selu')(out)
    out = layers.Conv1D(nb_output_bins, 1, padding='same')(out)

    if not learn_all_outputs:
        raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
        out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(
            out)  # Based on gif in deepmind blog: take last output?

    out = layers.Activation('softmax', name="output_softmax")(out)
    model = Model(input, out)

    receptive_field, receptive_field_ms = compute_receptive_field()

    _log.info('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    return model


@ex.capture
def compute_receptive_field(desired_sample_rate, dilation_depth, nb_stacks):
    return compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks)


def compute_receptive_field_(desired_sample_rate, dilation_depth, nb_stacks):
    receptive_field = nb_stacks * (2 ** dilation_depth * 2) - (nb_stacks - 1)
    receptive_field_ms = (receptive_field * 1000) / desired_sample_rate
    return receptive_field, receptive_field_ms


@ex.capture(prefix='optimizer')
def make_optimizer(optimizer, lr, momentum, decay, nesterov, epsilon):
    if optimizer == 'sgd':
        optim = SGD(lr, momentum, decay, nesterov)
    elif optimizer == 'adam':
        optim = Adam(lr=lr, decay=decay, epsilon=epsilon)
    elif optimizer == 'nadam':
        optim = Nadam(lr=lr, epsilon=epsilon)
    else:
        raise ValueError('Invalid config for optimizer.optimizer: ' + optimizer)
    return optim


@ex.command
def predict(desired_sample_rate, fragment_length, _log, seed, _seed, _config, predict_seconds, data_dir, batch_size,
            fragment_stride, nb_output_bins, learn_all_outputs, run_dir, predict_use_softmax_as_input, use_ulaw,
            predict_initial_input,
            **kwargs):
    fragment_length = compute_receptive_field()[0]
    _config['fragment_length'] = fragment_length

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
    epoch = int(re.match(r'checkpoint\.(\d+?)-.*', last_checkpoint).group(1))
    _log.info('Using checkpoint from epoch: %s' % epoch)

    sample_dir = os.path.join(run_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    sample_name = make_sample_name(epoch)
    sample_filename = os.path.join(sample_dir, sample_name)

    _log.info('Saving to "%s"' % sample_filename)

    sample_stream = make_sample_stream(desired_sample_rate, sample_filename)

    model = build_model()
    model.load_weights(os.path.join(checkpoint_dir, last_checkpoint))
    model.summary()

    if predict_initial_input is None:
        outputs = list(dataset.one_hot(np.zeros(fragment_length) + nb_output_bins / 2))
    elif predict_initial_input != '':
        _log.info('Taking first %d (%.2fs) from \'%s\' as initial input.' % (
            fragment_length, fragment_length / desired_sample_rate, predict_initial_input))
        wav = dataset.process_wav(desired_sample_rate, predict_initial_input, use_ulaw)
        outputs = list(dataset.one_hot(wav[0:fragment_length]))
    else:
        _log.info('Taking sample from test dataset as initial input.')
        data_generators, _ = get_generators()
        outputs = list(data_generators['test'].next()[0][-1])

    # write_samples(sample_stream, outputs)
    warned_repetition = False
    for i in tqdm(range(int(desired_sample_rate * predict_seconds))):
        if not warned_repetition:
            if np.argmax(outputs[-1]) == np.argmax(outputs[-2]) and np.argmax(outputs[-2]) == np.argmax(outputs[-3]):
                warned_repetition = True
                _log.warning('Last three predicted outputs where %d' % np.argmax(outputs[-1]))
            else:
                warned_repetition = False
        prediction_seed = np.expand_dims(np.array(outputs[i:i + fragment_length]), 0)
        output = model.predict(prediction_seed)
        output_dist = output[0][-1]
        output_val = draw_sample(output_dist)
        if predict_use_softmax_as_input:
            outputs.append(output_dist)
        else:
            outputs.append(output_val)
        write_samples(sample_stream, [output_val])

    sample_stream.close()

    _log.info("Done!")


@ex.capture
def make_sample_name(epoch, predict_seconds, predict_use_softmax_as_input, sample_argmax, sample_temperature, seed):
    sample_str = ''
    if predict_use_softmax_as_input:
        sample_str += '_soft-in'
    if sample_argmax:
        sample_str += '_argmax'
    else:
        sample_str += '_sample'
        if sample_temperature:
            sample_str += '-temp-%s' % sample_temperature
    sample_name = 'sample_epoch-%05d_%02ds_%s_seed-%d.wav' % (epoch, int(predict_seconds), sample_str, seed)
    return sample_name


@ex.capture
def write_samples(sample_file, out_val, use_ulaw):
    s = np.argmax(out_val, axis=-1).astype('uint8')
    # print out_val,
    if use_ulaw:
        s = dataset.ulaw2lin(s)
    # print s,
    s = bytearray(list(s))
    # print s[0]
    sample_file.writeframes(s)
    sample_file._file.flush()


@ex.capture
def get_generators(batch_size, data_dir, desired_sample_rate, fragment_length, fragment_stride, learn_all_outputs,
                   nb_output_bins, use_ulaw, test_factor, data_dir_structure, randomize_batch_order, _rnd,
                   random_train_batches):
    if data_dir_structure == 'flat':
        return dataset.generators(data_dir, desired_sample_rate, fragment_length, batch_size,
                                  fragment_stride, nb_output_bins, learn_all_outputs, use_ulaw, randomize_batch_order,
                                  _rnd, random_train_batches)

    elif data_dir_structure == 'vctk':
        return dataset.generators_vctk(data_dir, desired_sample_rate, fragment_length, batch_size,
                                       fragment_stride, nb_output_bins, learn_all_outputs, use_ulaw, test_factor,
                                       randomize_batch_order, _rnd, random_train_batches)
    else:
        raise ValueError('data_dir_structure must be "flat" or "vctk", is %s' % data_dir_structure)


@ex.command
def test_make_soft(_log, train_with_soft_target_stdev, _config):
    if train_with_soft_target_stdev is None:
        _config['train_with_soft_target_stdev'] = 1
    y_true = K.reshape(K.eye(512)[:129, :256], (2, 129, 256))
    y_soft = make_soft(y_true)
    f = K.function([], y_soft)
    _log.info('Output of soft:')
    f1 = f([])

    _log.info(f1[0, 0])
    _log.info(f1[-1, -1])


@ex.command
def test_preprocess(desired_sample_rate, batch_size, use_ulaw):
    sample_dir = os.path.join('preprocess_test')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    ulaw_str = '_ulaw' if use_ulaw else ''
    sample_filename = os.path.join(sample_dir, 'test1%s.wav' % ulaw_str)
    sample_stream = make_sample_stream(desired_sample_rate, sample_filename)

    data_generators, _ = get_generators()
    outputs = data_generators['test'].next()[0][1].astype('uint8')

    write_samples(sample_stream, outputs)
    scipy.io.wavfile.write(os.path.join(sample_dir, 'test2%s.wav' % ulaw_str), desired_sample_rate,
                           np.argmax(outputs, axis=-1).astype('uint8'))


def make_sample_stream(desired_sample_rate, sample_filename):
    sample_file = wave.open(sample_filename, mode='w')
    sample_file.setnchannels(1)
    sample_file.setframerate(desired_sample_rate)
    sample_file.setsampwidth(1)
    return sample_file


def softmax(x, temp, mod=np):
    x = mod.log(x) / temp
    e_x = mod.exp(x - mod.max(x, axis=-1))
    return e_x / mod.sum(e_x, axis=-1)


@ex.capture
def draw_sample(output_dist, sample_temperature, sample_argmax, _rnd):
    if sample_argmax:
        output_dist = np.eye(256)[np.argmax(output_dist, axis=-1)]
    else:
        if sample_temperature is not None:
            output_dist = softmax(output_dist, sample_temperature)
        output_dist = output_dist / np.sum(output_dist + 1e-7)
        output_dist = _rnd.multinomial(1, output_dist)
    return output_dist



@ex.automain
def main(run_dir, data_dir, nb_epoch, early_stopping_patience, desired_sample_rate, fragment_length, batch_size,
         fragment_stride, nb_output_bins, keras_verbose, _log, seed, _config, debug, learn_all_outputs,
         train_only_in_receptive_field, _run, use_ulaw, train_with_soft_target_stdev):

    # class TensorBoardWrapper(TensorBoard):
    #     '''Sets the self.validation_data property for use with TensorBoard callback.'''
    #
    #     def __init__(self, batch_gen, nb_steps, **kwargs):
    #         super().__init__(**kwargs)
    #         self.batch_gen = batch_gen  # The generator.
    #         self.nb_steps = nb_steps  # Number of times to call next() on the generator.
    #
    #     def on_epoch_end(self, epoch, logs):
    #         # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
    #         # Below is an example that yields images and classification tags.
    #         # After it's filled in, the regular on_epoch_end method has access to the validation_data.
    #         imgs, tags = None, None
    #         for s in range(self.nb_steps):
    #             ib, tb = next(self.batch_gen)
    #             if imgs is None and tags is None:
    #                 imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
    #                 tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
    #             imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
    #             tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
    #         self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
    #         return super().on_epoch_end(epoch, logs)

    if run_dir is None:
        if not os.path.exists("models"):
            os.mkdir("models")
        run_dir = os.path.join('models', datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S'))
        _config['run_dir'] = run_dir

    print_config(_run)

    _log.info('Running with seed %d' % seed)

    if not debug:
        if os.path.exists(run_dir):
            raise EnvironmentError('Run with seed %d already exists' % seed)
        os.mkdir(run_dir)
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        json.dump(_config, open(os.path.join(run_dir, 'config.json'), 'w'))

    _log.info('Loading data...')
    data_generators, nb_examples = get_generators()


    # test generator
    # aaa, bbb = next(data_generators['train'])

    # test razmazyvatel'
    # y = make_soft(bbb, _config['fragment_length'], _config['nb_output_bins'], _config['train_with_soft_target_stdev'], with_prints=True)


    _log.info('Building model...')
    model = build_model(fragment_length)
    _log.info(model.summary())

    optim = make_optimizer()
    # optim = Nadam(lr=0.001)

    _log.info('Compiling Model...')

    loss = objectives.categorical_crossentropy
    all_metrics = [
        metrics.categorical_accuracy,
        categorical_mean_squared_error
    ]
    if train_with_soft_target_stdev:
        loss = make_targets_soft(loss)
    if train_only_in_receptive_field:
        loss = skip_out_of_receptive_field(loss)
        all_metrics = [skip_out_of_receptive_field(m) for m in all_metrics]

    model.compile(optimizer=optim, loss=loss, metrics=all_metrics)

    plot_model(model, to_file='model.pdf', show_shapes=True)

    # TODO: Consider gradient weighting making last outputs more important.


    # Save the model
    checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5'),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)

    callbacks = [
        ReduceLROnPlateau(patience=early_stopping_patience / 2, cooldown=early_stopping_patience / 4, verbose=1),
        EarlyStopping(patience=early_stopping_patience, verbose=1),
    ]

    if not debug:
        callbacks.extend([
            checkpoint,
            # TerminateOnNaN,
            CSVLogger(os.path.join(run_dir, 'history.csv')),
        ])
        if K.backend() == "tensorflow":
            # tensorboard = TensorBoardWrapper(data_generators['test'], 8,  log_dir='./logs/' + datetime.datetime.now().strftime('run_%Y-%m-%d_%H-%M-%S'),
            #                           histogram_freq=1,
            #                           write_grads=True,
            #                           write_graph=False,
            #                           write_images=True)
            tensorboard = TensorBoard(log_dir='./logs/' + datetime.datetime.now().strftime('run_%Y-%m-%d_%H-%M-%S'),
                                      histogram_freq=0,
                                      write_grads=True,
                                      write_graph=False,
                                      write_images=True)
            callbacks.extend([
                tensorboard
            ])

    if not debug:
        os.mkdir(checkpoint_dir)
        _log.info('Starting Training...')

    model.fit_generator(data_generators['train'],
                        nb_examples['train'] / batch_size,
                        epochs=nb_epoch,
                        validation_data=data_generators['test'],
                        validation_steps=nb_examples['test'] / batch_size,
                        callbacks=callbacks,
                        verbose=keras_verbose)
