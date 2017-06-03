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

def vector_maker (row):
	'''
	Reads one row of data (from an already existing CSV dataset)
	and generates a restructured feature vector.

	Suitable for this case only, not for generator a CSV file
	from the dataset later on, needs more work in other words on the
	DatasetHandler class
	'''

	label = row[0]

	# cannot reconstrucut features names here
	names = [str(i) for i in range(len(row))]

	# kwargs of restructure method adjusted to our specific use case
	return (label,) + StructureTransformer.restructure(names, row[1:], windows_count=5, streams_count=6)

def cnn_model (features, labels, mode):
	width, height = features[0].shape

	# Input layer of CNN
	# shape is 4D -> [batch_size, image_width, image_height, channels]
	# # batch_size number of elements in features array
	# # channels number of colour channels in features (as images)
	# # width and height are already done in our case by the StructureTransformer
	input_layer = tf.reshape(features, [-1, width, height, 1])

	# First convolution layer

	# First pooling layer

	# Second convolution layer

	# Second pooling layer

	# First fully connected layer

	# Second fully connected layer

	# Output layer

path = '' # get file name

dataset = DatasetHandler.from_csv_file(path, vector_maker=vector_maker)

labels, features = dataset.as_arrays()
