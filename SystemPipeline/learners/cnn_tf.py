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
	print('h', height, 'w', width)

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
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28, 29, 32]
	# Output Tensor Shape: [batch_size, 28, 29, 64]
	conv_layer_2 = tf.layers.conv2d(
		inputs=pool_layer_1,
		filters=64,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu
	)

	# Second pooling layer
	# Pooling Layer #2
	# Input Tensor Shape: [batch_size, 28, 29, 64]
	# Output Tensor Shape: [batch_size, 14, 14, 64]
	pool_layer_2 = tf.layers.max_pooling2d(
		inputs=conv_layer_2,
		pool_size=[2, 2],
		strides=2
	)

	# Flatten second pooling layer

	# First fully connected layer

	# Second fully connected layer

	# Output layer

	print('input', input_layer.shape)
	print('conv_1', conv_layer_1.shape)
	print('pool_1', pool_layer_1.shape)
	print('conv_2', conv_layer_2.shape)
	print('pool_2', pool_layer_2.shape)

	exit(10)


path = '/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/floats.dataset.accel.only.withnative.dump.csv' # get file name

dataset = DatasetHandler.from_csv_file(path, vector_maker=vector_maker, dtype=np.float32)

labels, features = dataset.as_arrays()


cnn_model(features, 0, 0)
