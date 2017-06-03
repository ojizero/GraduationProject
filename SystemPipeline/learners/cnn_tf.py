import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')

import numpy as np
import tensorflow as tf

# data structuring
# CNN handles images, we don't have images

# one solution is to restructure the data as an image
# # two paths for restructuring while preserving correlations
# # # -> height, width = streams, features per stream
# # # -> height, width = streams*windows, features per window

# no need for feature selection here, let and let be

from utils.dataset_handler import DatasetHandler
from features.structure_transformer import StructureTransformer

def vector_maker (row, names):
	'''
	Reads one row of data (from an already existing CSV dataset)
	and generates a restructured feature vector.

	Suitable for this case only, not for generator a CSV file
	from the dataset later on, needs more work in other words on the
	DatasetHandler class
	'''

	label = row[0]

	# kwargs of restructure method adjusted to our specific use case
	return (label,) + StructureTransformer.restructure(names, row[1:], windows_count=5, streams_count=6, dtype=np.float32)

def cnn_model (features, labels, mode):
	height, width = features[0].shape
	output_size = len(np.unique(labels))

	# Input layer of CNN
	# shape is 4D -> [batch_size, image_width, image_height, channels]
	# # batch_size number of elements in features array
	# # channels number of colour channels in features (as images)
	# # width and height are already done in our case by the StructureTransformer
	input_layer = tf.reshape(features, [-1, width, height, 1])

	# First convolution layer
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 147, 30, 1]
	# Output Tensor Shape: [batch_size, 147, 30, 32]
	conv_layer_1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding='same', # how to make this wrap around ?!
		activation=tf.nn.relu
	)

	# First pooling layer
	# Input Tensor Shape: [batch_size, 147, 30, 32]
	# Output Tensor Shape: [batch_size, 28, 29, 32]
	pool_layer_1 = tf.layers.max_pooling2d(
		inputs=conv_layer_1,
		pool_size=[8, 2],
		strides=[5, 1] # stride assures overlapping between filters
	)

	# Second convolution layer
	# Computes 48 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28, 29, 32]
	# Output Tensor Shape: [batch_size, 28, 29, 48]
	conv_layer_2 = tf.layers.conv2d(
		inputs=pool_layer_1,
		filters=48,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu
	)

	# Second pooling layer
	# Input Tensor Shape: [batch_size, 28, 29, 48]
	# Output Tensor Shape: [batch_size, 9, 9, 48]
	pool_layer_2 = tf.layers.max_pooling2d(
		inputs=conv_layer_2,
		pool_size=[3, 3],
		strides=3
	)

	# Third convolution layer
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 9, 9, 48]
	# Output Tensor Shape: [batch_size, 9, 9, 64]
	conv_layer_3 = tf.layers.conv2d(
		inputs=pool_layer_2,
		filters=64,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu
	)

	# Third pooling layer
	# Output Tensor Shape: [batch_size, 9, 9, 64]
	# Output Tensor Shape: [batch_size, 3, 3, 64]
	pool_layer_3 = tf.layers.max_pooling2d(
		inputs=conv_layer_3,
		pool_size=[3, 3],
		strides=3
	)

	# Flatten third pooling layer
	flattened_pool = tf.reshape(pool_layer_3, [-1, 3 * 3 * 64])

	# First fully connected layer
	# Densely connected layer with 400 neurons
	# Input Tensor Shape: [batch_size, 3 * 3 * 64]
	# Output Tensor Shape: [batch_size, 400]
	connected_layer_1 = tf.layers.dense(
		inputs=flattened_pool,
		units=400,
		activation=tf.nn.relu
	)

	# Dropout
	# Add dropout operation; 0.6 probability that element will be kept
	dropout_1 = tf.layers.dropout(
		inputs=connected_layer_1,
		rate=0.4,
		training=mode == learn.ModeKeys.TRAIN
	)

	# Second fully connected layer
	# Densely connected layer with 200 neurons
	# Input Tensor Shape: [batch_size, 400]
	# Output Tensor Shape: [batch_size, 200]
	connected_layer_2 = tf.layers.dense(
		inputs=dropout_1,
		units=200,
		activation=tf.nn.relu
	)

	# Dropout
	# Add dropout operation; 0.67 probability that element will be kept
	dropout_2 = tf.layers.dropout(
		inputs=connected_layer_2,
		rate=0.3,
		training=mode == learn.ModeKeys.TRAIN
	)

	# Output layer
	logits = tf.layers.dense(
		inputs=dropout_2,
		units=output_size,
		activation=tf.nn.relu
	)

	## Print shapes of layers
	print('h', height, 'w', width)
	print('input'    , input_layer.shape)
	print('conv_1'   , conv_layer_1.shape)
	print('pool_1'   , pool_layer_1.shape)
	print('conv_2'   , conv_layer_2.shape)
	print('pool_2'   , pool_layer_2.shape)
	print('conv_3'   , conv_layer_3.shape)
	print('pool_3'   , pool_layer_3.shape)
	print('flat_pool', flattened_pool.shape)
	print('fully_1'  , connected_layer_1.shape)
	print('drop_1'   , dropout_1.shape)
	print('fully_2'  , connected_layer_2.shape)
	print('drop_2'   , dropout_2.shape)
	print('logits'   , logits.shape)

	loss = None
	train_op = None

	# Calculate loss for both TRAIN and EVAL modes
	# measures how closely the model's predictions match the target classes
	if mode != tf.contrib.learn.ModeKeys.INFER:
		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=output_size)
		loss = tf.losses.softmax_cross_entropy(
			onehot_labels=onehot_labels,
			logits=logits
		)

	# Configure the Training Op (for TRAIN mode)
	if mode == learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.contrib.framework.get_global_step(),
			learning_rate=0.001,
			optimizer='SGD'
		)

	# Generate Predictions
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(
		mode=mode,
		predictions=predictions,
		loss=loss,
		train_op=train_op
	)



# get file name
path = '/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/floats.dataset.accel.only.withnative.dump.csv'

dataset = DatasetHandler.from_csv_file(path, vector_maker=vector_maker, dtype=np.float32)

labels, features = dataset.as_arrays()

# convert labels to be indices
# because fuck tensorflow :|
uniques = np.unique(labels)
for index, unique in enumerate(uniques):
	labels[labels == unique] = index

cnn_model(features, labels, 0)
