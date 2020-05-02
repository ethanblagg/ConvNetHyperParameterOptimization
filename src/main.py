


import tensorflow as tf
from Ethanet_0 import Ethanet
import datetime
import os
import numpy as np
import argparse
import pathlib                      # glob

# ==================== Fetch args =========================================
parser = argparse.ArgumentParser(description='Provide optional inputs for scripting')

parser.add_argument('-r', '--learning_rate', type=float, default = 3e-3, help='Learning Rate for training')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of training Epochs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('-f', '--init_filter_size', type=int, default=12, help='First Conv Layer Filter Size')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose information')

loss_group = parser.add_mutually_exclusive_group()
loss_group.add_argument('-p', '--poisson', action='store_true')
loss_group.add_argument('-c', '--categorical_cross_entropy', action='store_true')
loss_group.add_argument('-m', '--mean_squared_error', action='store_true')

args = parser.parse_args();          # x & y lengths for images to be resized to\



# ==================== Constants / Variables ==============================
if args.poisson:
    loss_type = tf.keras.losses.Poisson()
elif args.categorical_cross_entropy:
    loss_type = tf.keras.losses.CategoricalCrossentropy()
elif args.mean_squared_error:
    loss_type = tf.keras.losses.MeanSquaredError()

learning_rate = args.learning_rate
num_epochs = args.epochs
batch_size = args.batch_size
loss = loss_type = tf.keras.losses.CategoricalCrossentropy()
filt_size = args.init_filter_size
verbose = args.verbose
image_size = 100
IMG_WIDTH = image_size
IMG_HEIGHT = image_size

param_str = "_lr={}_e={}".format(learning_rate, num_epochs)
model_time_identifier = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + param_str
log_dir = '../log/' + model_time_identifier                  # Directory where logs are saved
data_dir = '../data/'
train_data_dir_str = data_dir + 'training_data_fruit/'
test_data_dir_str = data_dir + 'testing_data_fruit/'
train_cache_str = data_dir + 'training_data_cache.tfcache'
test_cache_str = data_dir + 'testing_data_cache.tfcache'

train_data_dir = pathlib.Path(train_data_dir_str)
test_data_dir = pathlib.Path(test_data_dir_str)

# ==================== Setup ===============================================
print('\nRunning in {}'.format(os.getcwd()))
print('Log identifier:  {}'.format(model_time_identifier))
print('Training epochs: {}'.format(num_epochs))
print('Learning rate:   {}'.format(learning_rate))
print('')


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
            
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    #ds = ds.repeat()
    
    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return ds


CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt"])
if verbose:
    print(CLASS_NAMES)
    print(len(CLASS_NAMES))


train_list_ds = tf.data.Dataset.list_files(train_data_dir_str + '*/*')
test_list_ds = tf.data.Dataset.list_files(test_data_dir_str + '*/*')
if verbose:
    for f in train_list_ds.take(5):
        print(f.numpy())

    for f in test_list_ds.take(5):
        print(f.numpy())

print("Number of training records: ", tf.data.experimental.cardinality(train_list_ds))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_labeled_ds  = test_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = prepare_for_training(train_labeled_ds, cache=train_cache_str)
test_ds = prepare_for_training(test_labeled_ds, cache=test_cache_str)


if verbose:
    for image, label in train_labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())


print('\nCompiling Ethanet')
model = Ethanet(filt_size=filt_size, num_logits=len(CLASS_NAMES))                                           # Create the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),                                 # Compile the model
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits='true'),
              metrics=['accuracy'])

print('Connecting to Tensorboard')
tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=log_dir,                    # tensorboard callback
                                                 histogram_freq=1,                  # How often to log histogram visualizations
                                                 embeddings_freq=1,                 # How often to log embedding visualizations
                                                 update_freq='epoch')               # How often to write logs (default: once per epoch)

print('Training model')
history = model.fit(train_ds, #train_ds.batch(batch_size),                                               # train the model
                    epochs=num_epochs,
                    #validation_data=,
                    shuffle=True,
                    callbacks=[tensorboard_cbk])


print('Evaluating model')
print('final_output_line:  ', end='')
model.evaluate(test_ds,
               callbacks=[tensorboard_cbk])


print('\n\n\n')


