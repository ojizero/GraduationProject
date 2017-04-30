import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import numpy as np
import scipy.stats as st

from utils.decorators import classinstancemethod

class Extractor:
	_WINDOW_SIZE = 10

	@classinstancemethod
	def extract (obj, data, **kwargs):
		## introspection manipulation
		if isinstance(obj, type):
			obj = obj.__name__
		else:
			obj = 'obj'

		## parameters and their default values
		if not kwargs.get('multi', True):
			data = np.array([data])

		window_size = kwargs.get('window_size', eval(obj)._WINDOW_SIZE)

		overlap = kwargs('overlap', 0.0)
		assert 0.0 <= overlap < 1

		## method logic
		begin = border = window_size // 2
		step  = window_size - round(overlap * window_size)
		end   = data.shape[1] - window_size // 2

		# window the data
		data_windowed = np.array([data[:,pivot-border:pivot+border,...] for pivot in range(begin, end, step)])

		readings = data_windowed.shape[-1]
		# returns R -> R[sensor][window][reading]['feature_method_name']
		return np.array([eval(obj)._extract_features(data_windowed[...,col]) for col in range(readings)])

	@classinstancemethod
	def _extract_features (obj, data_column):
		# this condition is used to manipulate the introspective part when calculating features
		if isinstance(obj, type):
			obj = obj.__name__
		else:
			obj = 'obj'

		# perform each method ending with '_feature' from given class or instance on given data
		return {feature: eval('%s.%s' % (obj, feature))(data_column) for feature in dir(eval(obj)) if feature.endswith('_feature')}

	@staticmethod
	def _generic_feature_applier (data_streams, feature_function):
		return np.array(np.nan_to_num([
			[feature_function(window) for window in stream]
				for stream in data_streams
		]))

