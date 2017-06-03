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

		step = len(names) // streams_count // windows_count
		end  = len(names)

		names  = np.array([names[pivot:pivot+step]  for pivot in range(0, end, step)])
		values = np.array([values[pivot:pivot+step] for pivot in range(0, end, step)])

		return names, values
