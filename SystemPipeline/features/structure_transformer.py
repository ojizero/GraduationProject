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
	dtype = np.float

	def __init__ (self, extractor, dtype=np.float):
		self._extractor = extractor if extractor is not None else self._extractor
		self.dtype = dtype

	@classinstancemethod
	def transform (obj, **kwargs):
		names, values = super().transform(**kwargs)
		names, values = np.array(names), np.array(values, dtype=obj.dtype)

		return obj.restructure(names, values, **kwargs)

	@classinstancemethod
	def restructure (obj, names, values, **kwargs):
		windows_count = kwargs.get('windows_count', obj._windows_count)
		streams_count = kwargs.get('streams_count', 6)

		width  = len(names) // streams_count // windows_count
		height = streams_count * windows_count

		names = np.reshape(names,  (height, width))
		# having names as a tuple makes life easier (laziness)
		names = tuple([tuple(elem) for elem in names])

		values = np.reshape(values, (height, width))

		return names, values.astype(kwargs.get('dtype', obj.dtype))
