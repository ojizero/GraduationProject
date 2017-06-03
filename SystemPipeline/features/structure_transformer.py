import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')

import numpy as np

from utils.decorators import classinstancemethod
from features.features_transformer import FeaturesTransformer

class StructureTransformer (FeaturesTransformer):
	'''
	Restructures the feature vector in an image like format
	'''
	@classinstancemethod
	def transform (obj, **kwargs):
		names, values = super().transform(**kwargs)
		names, values = np.array(names), np.array(values)

		return obj.restructure(names, values, **kwargs)

	@classinstancemethod
	def restructure (obj, names, values, **kwargs):
		windows_count = kwargs.get('windows_count', obj._windows_count)
		streams_count = kwargs.get('streams_count', 6)

		height = len(names) // streams_count // windows_count
		width  = streams_count * windows_count

		names  = np.reshape(names,  (width, height))
		values = np.reshape(values, (width, height))

		return names, values
